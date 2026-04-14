"""Server transfer helpers for the manual alignment tool.

Provides download (make_manual_align_package output from server) and upload
(manual transforms back to server) via SCP, using connection details
parsed from a subject's nextflow.config.

Also exposes ``ScpWorker``, a thin ``QThread`` wrapper that runs a transfer
function on a background thread so the Qt event loop stays responsive.
"""

from __future__ import annotations

import logging
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

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
