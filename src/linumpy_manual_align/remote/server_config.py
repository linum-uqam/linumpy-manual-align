"""Parse server connection details from a local nextflow.config."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class ServerConfig:
    """Connection details parsed from nextflow.config + update_files.sh conventions."""

    host: str
    remote_output: str  # e.g. /scratch/workspace/sub-22/output
    subject_id: str  # e.g. sub-22
    config_path: Path | None = None


def parse_server_config(config_path: Path, *, host: str = "") -> ServerConfig | None:
    """Parse server connection info from a local nextflow.config.

    Extracts the output directory pattern and derives the subject ID
    from the config file's parent directory name (e.g. ~/Downloads/sub-22/).

    Parameters
    ----------
    config_path : Path
        Path to a local copy of the subject's nextflow.config.
    host : str
        Server hostname or IP. Usually from :data:`settings` or the dock Host field.

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

    # Remote workspace path follows convention: /scratch/workspace/{subject_id}/
    remote_workspace = f"/scratch/workspace/{subject_id}"
    remote_output = f"{remote_workspace}/output"

    return ServerConfig(host=host, remote_output=remote_output, subject_id=subject_id, config_path=config_path)
