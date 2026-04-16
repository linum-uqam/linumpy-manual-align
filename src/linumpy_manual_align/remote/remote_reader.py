"""SSH-backed OME-Zarr slice reader for remote cross-section requests."""

from __future__ import annotations

import base64
import io
import os
import subprocess
import threading
from dataclasses import dataclass, field

import numpy as np

from linumpy_manual_align.remote.cs_script import _cs_server_script, _remote_python
from linumpy_manual_align.remote.server_config import ServerConfig


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
        input=_cs_server_script().encode(),
        capture_output=True,
        timeout=15,
    )
    if upload.returncode != 0:
        raise RuntimeError(f"Failed to upload server script: {upload.stderr.decode().strip()}")

    # Start the persistent process
    proc = subprocess.Popen(
        ["ssh", server.host, f"{_remote_python()} {remote_script} {remote_zarr_path} {level}"],
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
