"""Server transfer helpers for the manual alignment tool.

Provides download (make_manual_align_package output from server) and upload
(manual transforms back to server) via SCP, using connection details
parsed from a subject's nextflow.config.

Also exposes:

- ``ScpWorker`` — thin QThread wrapper for SCP operations.
- ``RemoteSliceReader`` — persistent SSH+Python process per slice that loads
  an OME-Zarr volume once and serves cross-section requests via stdin/stdout.
- ``SliceReaderWorker`` — QThread that opens a ``RemoteSliceReader``.
- ``CrossSectionWorker`` — QThread that fetches one 2-D cross-section image
  from an open ``RemoteSliceReader``.
"""

from __future__ import annotations

import base64
import io
import logging
import os
import re
import subprocess
import threading
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from qtpy.QtCore import QThread, Signal

logger = logging.getLogger(__name__)


class ScpWorker(QThread):
    """Background QThread for SCP download/upload operations.

    Emits ``finished(ok, message)`` when the transfer completes.
    """

    finished = Signal(bool, str)

    def __init__(self, func: object, args: tuple) -> None:
        super().__init__()
        self._func = func
        self._args = args

    def run(self) -> None:
        ok, msg = self._func(*self._args)
        self.finished.emit(ok, msg)


# Path to the Python interpreter in the linumpy virtualenv on the server.
# Managed by uv; update here if the server layout ever changes.
_REMOTE_PYTHON = "/home/frans/code/linumpy/.venv/bin/python"

# Server-side script uploaded once per reader session.
# The script deletes itself immediately on startup so /tmp never accumulates
# stale files, even if the widget crashes.  The kernel keeps the process
# alive from its file buffer after deletion.
_CS_SERVER_SCRIPT = """\
import os, sys, numpy as np, base64, io
from linumpy.io.zarr import read_omezarr

try:
    os.unlink(sys.argv[0])   # self-delete; kernel keeps process alive
except OSError:
    pass  # already deleted by a concurrent reader (harmless)

zarr_path = sys.argv[1]
level = int(sys.argv[2])
vol, sc = read_omezarr(zarr_path, level=level)
arr = np.asarray(vol)        # (Z, Y, X)
nz, ny, nx = arr.shape
scale_str = ",".join(str(s) for s in sc)
print(f"ready {nz} {ny} {nx} {scale_str}", flush=True)

for line in sys.stdin:
    line = line.strip()
    if not line:
        continue
    axis, pos = line.split(",")
    pos = int(pos)
    if axis == "xz":
        img = arr[:, pos, :][::-1]   # (Z, X), flip Z
    else:
        img = arr[:, :, pos][::-1]   # (Z, Y), flip Z
    buf = io.BytesIO()
    np.savez_compressed(buf, img=img.astype("float32"), scale=np.array(sc, dtype="float64"))
    print(base64.b64encode(buf.getvalue()).decode(), flush=True)
"""


@dataclass
class RemoteSliceReader:
    """Persistent SSH+Python process serving cross-section requests for one slice.

    After construction (via ``open_remote_slice_reader``), each call to
    ``request(axis, pos)`` sends one line to the remote process and returns
    a 2-D numpy array in <100 ms.  The process holds the full zarr volume
    in RAM on the server, so the first open takes ~2 s (zarr load) and all
    subsequent requests are fast.
    """

    slice_id: int
    shape: tuple[int, int, int]  # (Z, Y, X) at the requested level
    scale: list[float]
    _process: subprocess.Popen = field(repr=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)
    _host: str = field(default="", repr=False)

    def request(self, axis: str, pos: int) -> tuple[np.ndarray, list[float]]:
        """Return one cross-section image (blocks until result arrives).

        Parameters
        ----------
        axis : {"xz", "yz"}
            "xz" fixes Y = *pos*, returning shape (Z, X).
            "yz" fixes X = *pos*, returning shape (Z, Y).
        pos : int
            Column/row index in the volume at the pyramid level used on open.

        Returns
        -------
        img : np.ndarray
            2-D float32 image, Z-axis flipped so depth increases downward.
        scale : list[float]
            2-element scale list [z_scale, lateral_scale].
        """
        assert self._process.stdin is not None and self._process.stdout is not None
        with self._lock:
            self._process.stdin.write(f"{axis},{pos}\n".encode())
            self._process.stdin.flush()
            data_line = self._process.stdout.readline().strip()
        buf = io.BytesIO(base64.b64decode(data_line))
        d = np.load(buf)
        return d["img"], d["scale"].tolist()

    def close(self) -> None:
        """Terminate the SSH process (server script already self-deleted)."""
        try:
            self._process.terminate()
            self._process.wait(timeout=3)
        except Exception:
            pass


def open_remote_slice_reader(
    server: ServerConfig,
    remote_zarr_path: str,
    level: int = 0,
    reader_id: int = 0,
) -> RemoteSliceReader:
    """Open a persistent SSH+Python process for *remote_zarr_path*.

    Uploads the server-side script to ``/tmp/cs_server_{pid}_{reader_id}.py``,
    starts the process, reads the ``"ready Z Y X scale"`` handshake, and
    returns a ready-to-use ``RemoteSliceReader``.

    *reader_id* (typically the slice ID) is used to give each concurrent
    reader a unique script path so simultaneous uploads don't collide.

    Raises ``RuntimeError`` if the handshake fails.
    """
    pid = os.getpid()
    remote_script = f"/tmp/cs_server_{pid}_{reader_id}.py"

    # Upload the server-side script via SSH echo (single round-trip, avoids SCP)
    upload_cmd = ["ssh", server.host, f"cat > {remote_script}"]
    upload = subprocess.run(
        upload_cmd,
        input=_CS_SERVER_SCRIPT.encode(),
        capture_output=True,
        timeout=15,
    )
    if upload.returncode != 0:
        raise RuntimeError(f"Failed to upload server script: {upload.stderr.decode().strip()}")

    # Start the persistent process
    proc = subprocess.Popen(
        ["ssh", server.host, f"{_REMOTE_PYTHON} {remote_script} {remote_zarr_path} {level}"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    # Read the "ready Z Y X s0,s1,s2" handshake (up to 30 s for zarr load)
    assert proc.stdout is not None and proc.stderr is not None
    try:
        line = proc.stdout.readline().decode().strip()
    except Exception as exc:
        proc.terminate()
        raise RuntimeError(f"SSH reader process failed before handshake: {exc}") from exc

    if not line.startswith("ready"):
        # Process exited before handshake — read stderr for the actual error.
        try:
            proc.wait(timeout=5)
            stderr = proc.stderr.read().decode(errors="replace").strip()
        except subprocess.TimeoutExpired:
            proc.terminate()
            stderr = "(process still running — check zarr path)"
        raise RuntimeError(
            f"Server process exited before handshake.\n  zarr: {remote_zarr_path}\n  stdout: {line!r}\n  stderr: {stderr}"
        )

    parts = line.split()
    # parts: ["ready", Z, Y, X, "s0,s1,s2"]  (scale is comma-joined by the script)
    nz, ny, nx = int(parts[1]), int(parts[2]), int(parts[3])
    scale = [float(s) for s in parts[4].split(",")]

    return RemoteSliceReader(
        slice_id=-1,  # caller sets this
        shape=(nz, ny, nx),
        scale=scale,
        _process=proc,
        _host=server.host,
    )


class SliceReaderWorker(QThread):
    """QThread that opens a ``RemoteSliceReader`` in the background.

    Emits ``ready(slice_id, reader)`` on success or
    ``failed(slice_id, message)`` on error.
    """

    ready = Signal(int, object)  # (slice_id, RemoteSliceReader)
    failed = Signal(int, str)  # (slice_id, error_message)

    def __init__(
        self,
        server: ServerConfig,
        slice_id: int,
        remote_zarr_path: str,
        level: int = 0,
    ) -> None:
        super().__init__()
        self._server = server
        self._slice_id = slice_id
        self._remote_zarr_path = remote_zarr_path
        self._level = level

    def run(self) -> None:
        try:
            reader = open_remote_slice_reader(self._server, self._remote_zarr_path, self._level, reader_id=self._slice_id)
            reader.slice_id = self._slice_id
            self.ready.emit(self._slice_id, reader)
        except Exception as exc:
            self.failed.emit(self._slice_id, str(exc))


class CrossSectionWorker(QThread):
    """QThread that fetches one 2-D cross-section from an open ``RemoteSliceReader``.

    Emits ``finished(slice_id, axis, pos, img)`` where *img* is a float32
    ndarray, or ``failed(slice_id, axis, pos, message)`` on error.
    """

    finished = Signal(int, str, int, object)  # (slice_id, axis, pos, img)
    failed = Signal(int, str, int, str)  # (slice_id, axis, pos, msg)

    def __init__(
        self,
        reader: RemoteSliceReader,
        axis: str,
        pos: int,
    ) -> None:
        super().__init__()
        self._reader = reader
        self._axis = axis
        self._pos = pos

    def run(self) -> None:
        try:
            img, _scale = self._reader.request(self._axis, self._pos)
            self.finished.emit(self._reader.slice_id, self._axis, self._pos, img)
        except Exception as exc:
            self.failed.emit(self._reader.slice_id, self._axis, self._pos, str(exc))


@dataclass
class ServerConfig:
    """Connection details parsed from nextflow.config + update_files.sh conventions."""

    host: str
    remote_output: str  # e.g. /scratch/workspace/sub-22/output
    subject_id: str  # e.g. sub-22
    config_path: Path | None = None


def parse_server_config(config_path: Path, *, host: str = "132.207.157.41") -> ServerConfig | None:
    """Parse server connection info from a local nextflow.config.

    Extracts the output directory pattern and derives the subject ID
    from the config file's parent directory name (e.g. ~/Downloads/sub-22/).

    Parameters
    ----------
    config_path : Path
        Path to a local copy of the subject's nextflow.config.
    host : str
        Server hostname or IP. Defaults to the LINUM lab server.

    Returns
    -------
    ServerConfig or None
        Parsed config, or None if parsing fails.
    """
    config_path = Path(config_path)
    if not config_path.exists():
        logger.error(f"Config file not found: {config_path}")
        return None

    # Derive subject ID from parent directory name
    subject_id = config_path.parent.name  # e.g. "sub-22"
    if not re.match(r"sub-\d+", subject_id):
        logger.warning(f"Parent directory '{subject_id}' doesn't look like a subject ID")

    # Server host (from parameter — default matches update_files.sh convention)

    # Remote workspace path follows convention: /scratch/workspace/{subject_id}/
    remote_workspace = f"/scratch/workspace/{subject_id}"
    remote_output = f"{remote_workspace}/output"

    return ServerConfig(host=host, remote_output=remote_output, subject_id=subject_id, config_path=config_path)


def _run_scp(args: list[str], description: str) -> tuple[bool, str]:
    """Run an scp command and return (success, message)."""
    cmd = ["scp", *args]
    logger.info(f"Running: {' '.join(cmd)}")
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if result.returncode == 0:
            return True, f"{description}: OK"
        return False, f"{description}: FAILED\n{result.stderr.strip()}"
    except subprocess.TimeoutExpired:
        return False, f"{description}: TIMEOUT (>300s)"
    except FileNotFoundError:
        return False, f"{description}: scp not found"


def download_manual_align_package(
    server: ServerConfig,
    local_dir: Path,
    level: int = 1,
) -> tuple[bool, str]:
    """Download the manual_align data package from the server.

    Downloads:
    - AIPs from output/make_manual_align_package/manual_align_package/aips/
    - Transforms from output/make_manual_align_package/manual_align_package/transforms/
    - Metadata JSON

    Parameters
    ----------
    server : ServerConfig
        Server connection details.
    local_dir : Path
        Local directory to download into (will be created).
    level : int
        Expected pyramid level (for logging only).

    Returns
    -------
    tuple[bool, str]
        (success, status_message)
    """
    local_dir = Path(local_dir)
    local_dir.mkdir(parents=True, exist_ok=True)

    remote_pkg = f"{server.remote_output}/make_manual_align_package/manual_align_package"

    # Download entire package recursively
    ok, msg = _run_scp(
        ["-r", f"{server.host}:{remote_pkg}/", str(local_dir) + "/"],
        "Download manual align package",
    )
    if not ok:
        return False, msg

    # Verify we got the expected files
    aips_dir = local_dir / "manual_align_package" / "aips"
    if not aips_dir.exists():
        # scp -r may place contents directly or in a subdirectory
        aips_dir = local_dir / "aips"

    n_aips = len(list(aips_dir.glob("*.npz"))) if aips_dir.exists() else 0
    return True, f"Downloaded {n_aips} AIPs to {local_dir}"


def upload_manual_transforms(
    server: ServerConfig,
    local_transforms_dir: Path,
) -> tuple[bool, str]:
    """Upload manual transforms to the server.

    Uploads each slice_z##/ subdirectory to:
    {remote_output}/manual_transforms/

    Parameters
    ----------
    server : ServerConfig
        Server connection details.
    local_transforms_dir : Path
        Local directory containing slice_z##/ subdirs with .tfm files.

    Returns
    -------
    tuple[bool, str]
        (success, status_message)
    """
    local_transforms_dir = Path(local_transforms_dir)
    if not local_transforms_dir.exists():
        return False, f"Local transforms directory not found: {local_transforms_dir}"

    # Find all slice directories
    slice_dirs = sorted(local_transforms_dir.glob("slice_z*"))
    if not slice_dirs:
        return False, "No slice_z* directories found to upload"

    remote_dest = f"{server.remote_output}/manual_transforms/"

    # First create the remote directory
    mkdir_cmd = ["ssh", server.host, f"mkdir -p {remote_dest}"]
    try:
        subprocess.run(mkdir_cmd, capture_output=True, text=True, timeout=30, check=True)
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
        return False, f"Failed to create remote directory: {e}"

    # Upload each slice directory
    ok, msg = _run_scp(
        ["-r"] + [str(d) for d in slice_dirs] + [f"{server.host}:{remote_dest}"],
        f"Upload {len(slice_dirs)} manual transforms",
    )
    if not ok:
        return ok, msg

    return True, f"Uploaded {len(slice_dirs)} transforms to {server.host}:{remote_dest}"
