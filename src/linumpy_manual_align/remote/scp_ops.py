"""SCP download/upload of manual-align packages and transforms."""

from __future__ import annotations

import logging
import subprocess
from pathlib import Path

from qtpy.QtCore import QThread, Signal

from linumpy_manual_align.remote.server_config import ServerConfig

logger = logging.getLogger(__name__)


class ScpWorker(QThread):
    """Background QThread for SCP download/upload operations.

    Emits ``finished(ok, message)`` when the transfer completes.
    """

    transfer_done = Signal(bool, str)

    def __init__(self, func: object, args: tuple) -> None:
        super().__init__()
        self._func = func
        self._args = args

    def run(self) -> None:
        """Execute the wrapped callable and emit the transfer-done signal."""
        ok, msg = self._func(*self._args)
        self.transfer_done.emit(ok, msg)


def _run_scp(args: list[str], description: str) -> tuple[bool, str]:
    """Run an scp command and return (success, message)."""
    cmd = ["scp", *args]
    logger.info("Running: %s", " ".join(cmd))
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
    _level: int = 1,
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


def _dirs_within_root(dirs: list[Path], root: Path) -> Path | None:
    """Return the first dir not contained under root, or None if all are contained."""
    resolved_root = root.resolve()
    for d in dirs:
        resolved = Path(d).resolve()
        if resolved != resolved_root and not resolved.is_relative_to(resolved_root):
            return Path(d)
    return None


def upload_manual_transforms(
    server: ServerConfig,
    local_transforms_dir: Path,
    slice_dirs: list[Path] | None = None,
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
    slice_dirs : list[Path] | None, optional
        Explicit slice directories to upload. When None, all ``slice_z*``
        subdirectories under ``local_transforms_dir`` are discovered via glob.

    Returns
    -------
    tuple[bool, str]
        (success, status_message)
    """
    local_transforms_dir = Path(local_transforms_dir)
    if not local_transforms_dir.exists():
        return False, f"Local transforms directory not found: {local_transforms_dir}"

    slice_dirs = (
        sorted(local_transforms_dir.glob("slice_z*"))
        if slice_dirs is None
        else [Path(d) for d in slice_dirs]
    )

    if not slice_dirs:
        return False, "No slice directories to upload"

    offending = _dirs_within_root(slice_dirs, local_transforms_dir)
    if offending is not None:
        return (
            False,
            f"Refusing to upload directory outside {local_transforms_dir}: {offending}",
        )

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
