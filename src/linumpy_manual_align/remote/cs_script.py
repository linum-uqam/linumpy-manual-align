"""Load the server-side cross-section script for SSH upload."""

from __future__ import annotations

from importlib import resources
from pathlib import Path

from linumpy_manual_align.settings import settings


def _remote_python() -> str:
    """Return the remote Python path from user settings."""
    return str(settings.get("server/remote_python"))


def _cs_server_script() -> str:
    """Return the content of the server-side cross-section script.

    Loaded via :mod:`importlib.resources` so the file is found when the package
    is installed from a wheel (not only in editable mode).  Falls back to a
    path next to this module for unusual layouts.
    """
    pkg = resources.files("linumpy_manual_align")
    script = pkg.joinpath("remote", "cs_server.py")
    try:
        return script.read_text(encoding="utf-8")
    except OSError:
        return (Path(__file__).resolve().parent / "cs_server.py").read_text(encoding="utf-8")
