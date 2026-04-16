"""Qt threads for remote slice readers and cross-section fetches."""

from __future__ import annotations

from qtpy.QtCore import QThread, Signal

from linumpy_manual_align.remote.remote_reader import RemoteSliceReader, open_remote_slice_reader
from linumpy_manual_align.remote.server_config import ServerConfig


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

    Emits ``result(slice_id, axis, pos, img)`` where *img* is a float32
    ndarray, or ``failed(slice_id, axis, pos, message)`` on error.

    Note: the custom signal is named ``result`` (not ``finished``) to avoid
    shadowing QThread's built-in ``finished`` signal, which is used for safe
    thread-lifetime management.
    """

    result = Signal(int, str, int, object)  # (slice_id, axis, pos, img)
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
            self.result.emit(self._reader.slice_id, self._axis, self._pos, img)
        except Exception as exc:
            self.failed.emit(self._reader.slice_id, self._axis, self._pos, str(exc))
