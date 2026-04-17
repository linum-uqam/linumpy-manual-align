"""Load the server-side cross-section script for SSH upload."""

from __future__ import annotations

import os
from importlib import resources
from pathlib import Path

from linumpy_manual_align.settings import settings


def _remote_python() -> str:
    """Return the remote Python interpreter for the SSH remote command line.

    Precedence:

    #. Environment variable :envvar:`LINUMPY_MANUAL_ALIGN_REMOTE_PYTHON` (non-empty).
    #. :class:`~linumpy_manual_align.settings.AppSettings` key ``server/remote_python``
       (non-empty after strip).

    If both are unset, :exc:`RuntimeError` is raised. There is no default to
    ``python3``: the server-side script needs linumpy, zarr, ome-zarr, etc.; a
    system interpreter almost never has those.

    The uploaded script is run as ``<this> /tmp/cs_server_….py …`` (not as a
    standalone executable).
    """
    env = os.environ.get("LINUMPY_MANUAL_ALIGN_REMOTE_PYTHON", "").strip()
    if env:
        return env
    path = str(settings.get("server/remote_python")).strip()
    if path:
        return path
    raise RuntimeError(
        "Remote Python is not configured. The SSH cross-section script needs linumpy, "
        "zarr, ome-zarr, and related packages — use the same interpreter you use on the "
        "server (e.g. …/linumpy/.venv/bin/python). Set it in Manual Align → Settings → "
        "Server → Remote Python path, or set LINUMPY_MANUAL_ALIGN_REMOTE_PYTHON to that "
        "path."
    )


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
