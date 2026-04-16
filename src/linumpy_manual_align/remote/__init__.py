"""Server SCP, remote slice readers, and cross-section workers.

Public API is re-exported here; implementation lives in sibling modules
(``server_config``, ``scp_ops``, ``remote_reader``, ``qt_workers``, ``cs_script``).
"""

from __future__ import annotations

from linumpy_manual_align.remote.cs_script import _cs_server_script
from linumpy_manual_align.remote.qt_workers import CrossSectionWorker, SliceReaderWorker
from linumpy_manual_align.remote.remote_reader import RemoteSliceReader, open_remote_slice_reader
from linumpy_manual_align.remote.scp_ops import (
    ScpWorker,
    download_manual_align_package,
    upload_manual_transforms,
)
from linumpy_manual_align.remote.server_config import ServerConfig, parse_server_config

__all__ = [
    "CrossSectionWorker",
    "RemoteSliceReader",
    "ScpWorker",
    "ServerConfig",
    "SliceReaderWorker",
    "_cs_server_script",
    "download_manual_align_package",
    "open_remote_slice_reader",
    "parse_server_config",
    "upload_manual_transforms",
]
