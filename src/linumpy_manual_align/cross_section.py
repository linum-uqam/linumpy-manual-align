"""Interactive cross-section manager for the manual alignment widget.

Encapsulates all state and background-thread plumbing related to the
remote OME-Zarr cross-section feature:

- Metadata loading from ``manual_align_metadata.json``
- Lazy ``RemoteSliceReader`` lifecycle (open on demand, keep open, close on exit)
- Cross-section request / cache / prefetch logic
- Eviction of stale cache entries to keep memory bounded

The manager is a ``QObject`` with Qt signals so that the widget can
connect to ``reader_ready``, ``reader_failed``, ``cross_section_ready``,
and ``cross_section_failed`` without needing to own the worker threads.
"""

from __future__ import annotations

import contextlib
import json
import logging
from pathlib import Path

import numpy as np
from qtpy.QtCore import QObject, Signal

from linumpy_manual_align.server_transfer import (
    CrossSectionWorker,
    RemoteSliceReader,
    ServerConfig,
    SliceReaderWorker,
)

logger = logging.getLogger(__name__)

# Number of positions to pre-fetch on each side of the current position.
_PREFETCH_STEPS = 5
# Step size in volume pixels between pre-fetched positions.
_PREFETCH_STEP_SIZE = 10
# Maximum cached positions per (slice, axis) before eviction.
_CACHE_EVICT_RADIUS = 15 * _PREFETCH_STEP_SIZE  # ±150 px


class CrossSectionManager(QObject):
    """Manages remote OME-Zarr readers, cross-section cache, and prefetch.

    Signals
    -------
    reader_ready(slice_id, reader)
        Emitted when a ``RemoteSliceReader`` finishes opening successfully.
    reader_failed(slice_id, message)
        Emitted when a ``SliceReaderWorker`` fails to open a reader.
    cross_section_ready(slice_id, axis, pos, img)
        Emitted when a fetched cross-section image is available in the cache.
    cross_section_failed(slice_id, axis, pos, message)
        Emitted when a cross-section fetch fails.
    """

    reader_ready = Signal(int, object)  # (slice_id, RemoteSliceReader)
    reader_failed = Signal(int, str)  # (slice_id, error_message)
    cross_section_ready = Signal(int, str, int, object)  # (sid, axis, pos, img)
    cross_section_failed = Signal(int, str, int, str)  # (sid, axis, pos, msg)

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)

        # Populated from manual_align_metadata.json
        self.slices_remote_dir: str | None = None
        self.cs_level: int = 0
        self.slice_filenames: dict[int, str] = {}

        # One persistent SSH reader per slice ID — opened lazily, kept alive.
        self._readers: dict[int, RemoteSliceReader] = {}
        # SliceReaderWorker threads currently opening a reader.
        self._reader_workers: dict[int, SliceReaderWorker] = {}
        # CrossSectionWorker threads with in-flight requests.
        self._cs_workers: list[CrossSectionWorker] = []
        # Cache: sid → axis → pos → float32 image array
        self._cache: dict[int, dict[str, dict[int, np.ndarray]]] = {}

    # ------------------------------------------------------------------
    # Metadata
    # ------------------------------------------------------------------

    def load_metadata(self, pkg_root: Path) -> None:
        """Populate remote-reader settings from ``manual_align_metadata.json``.

        Tries *pkg_root* first, then *pkg_root.parent* to handle both flat
        and nested package layouts.
        """
        for candidate in (
            pkg_root / "manual_align_metadata.json",
            pkg_root.parent / "manual_align_metadata.json",
        ):
            if candidate.exists():
                try:
                    meta = json.loads(candidate.read_text())
                    self.slices_remote_dir = meta.get("slices_remote_dir")
                    self.cs_level = int(meta.get("cross_section_level", meta.get("pyramid_level", 0)))
                    self.slice_filenames = {int(k): v for k, v in meta.get("slice_filenames", {}).items()}
                    if self.slices_remote_dir:
                        logger.info(
                            "Interactive XZ/YZ enabled: remote zarr at %s level %d",
                            self.slices_remote_dir,
                            self.cs_level,
                        )
                except Exception as exc:
                    logger.warning("Could not read package metadata: %s", exc)
                break

    # ------------------------------------------------------------------
    # Reader lifecycle
    # ------------------------------------------------------------------

    def ensure_reader(self, server_config: ServerConfig, slice_id: int) -> None:
        """Start a background worker to open a ``RemoteSliceReader`` for *slice_id*.

        Does nothing if a reader or an in-progress worker already exists.
        ``self.slices_remote_dir`` must be set (from :meth:`load_metadata`) before calling.
        """
        if self.slices_remote_dir is None:
            return
        if slice_id in self._readers or slice_id in self._reader_workers:
            return
        zarr_name = self.slice_filenames.get(slice_id, f"slice_z{slice_id:02d}.ome.zarr")
        remote_path = f"{self.slices_remote_dir}/{zarr_name}"
        worker = SliceReaderWorker(server_config, slice_id, remote_path, self.cs_level)
        worker.ready.connect(self._handle_reader_ready)
        worker.failed.connect(self._handle_reader_failed)
        self._reader_workers[slice_id] = worker
        worker.start()

    def reader_shape(self, slice_id: int) -> tuple[int, int, int] | None:
        """Return the ``(Z, Y, X)`` shape of the reader for *slice_id*, or *None*."""
        reader = self._readers.get(slice_id)
        return reader.shape if reader is not None else None

    def has_reader(self, slice_id: int) -> bool:
        """Return True if an open reader exists for *slice_id*."""
        return slice_id in self._readers

    # ------------------------------------------------------------------
    # Cache access
    # ------------------------------------------------------------------

    def get_cached(self, slice_id: int, axis: str, pos: int) -> np.ndarray | None:
        """Return a cached cross-section image, or *None* if not yet fetched."""
        return self._cache.get(slice_id, {}).get(axis, {}).get(pos)

    # ------------------------------------------------------------------
    # Request and prefetch
    # ------------------------------------------------------------------

    def request(self, slice_id: int, axis: str, pos: int) -> None:
        """Fetch cross-section *axis*[*pos*] for *slice_id*, using cache if available.

        If the image is already cached the ``cross_section_ready`` signal is
        emitted immediately on the next event-loop iteration via the normal
        signal path (the worker would have emitted it otherwise).  If not
        cached, a :class:`CrossSectionWorker` is spawned.
        """
        if self.get_cached(slice_id, axis, pos) is not None:
            # Already have it — emit so the widget refreshes display.
            self.cross_section_ready.emit(slice_id, axis, pos, self._cache[slice_id][axis][pos])
            return
        reader = self._readers.get(slice_id)
        if reader is None:
            return
        worker = CrossSectionWorker(reader, axis, pos)
        worker.finished.connect(self._handle_cs_ready)
        worker.failed.connect(self._handle_cs_failed)
        self._cs_workers.append(worker)
        worker.finished.connect(lambda *_: self._cs_workers.remove(worker) if worker in self._cs_workers else None)
        worker.start()

    def prefetch_around(self, slice_id: int, axis: str, pos: int) -> None:
        """Pre-fetch cross-sections at ±steps around *pos* and evict stale entries.

        Spawns up to ``_PREFETCH_STEPS * 2`` workers for positions not yet
        cached.  Also evicts cache entries further than ``_CACHE_EVICT_RADIUS``
        pixels from *pos* to keep per-(slice, axis) memory bounded.
        """
        reader = self._readers.get(slice_id)
        if reader is None:
            return
        max_pos = reader.shape[2] if axis == "yz" else reader.shape[1]
        for i in range(1, _PREFETCH_STEPS + 1):
            offset = i * _PREFETCH_STEP_SIZE
            for candidate in (pos + offset, pos - offset):
                if 0 <= candidate < max_pos and self.get_cached(slice_id, axis, candidate) is None:
                    self.request(slice_id, axis, candidate)

        # Evict positions outside the sliding window.
        axis_cache = self._cache.get(slice_id, {}).get(axis, {})
        if axis_cache:
            for cached_pos in list(axis_cache.keys()):
                if abs(cached_pos - pos) > _CACHE_EVICT_RADIUS:
                    del axis_cache[cached_pos]

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def close_all(self) -> None:
        """Terminate all open readers and in-progress workers."""
        for reader in list(self._readers.values()):
            reader.close()
        self._readers.clear()
        for w in list(self._reader_workers.values()):
            with contextlib.suppress(Exception):
                w.quit()
                w.wait(2000)
        self._reader_workers.clear()
        for w in list(self._cs_workers):
            with contextlib.suppress(Exception):
                w.quit()
                w.wait(2000)
        self._cs_workers.clear()

    # ------------------------------------------------------------------
    # Internal Qt slots
    # ------------------------------------------------------------------

    def _handle_reader_ready(self, slice_id: int, reader: RemoteSliceReader) -> None:
        self._reader_workers.pop(slice_id, None)
        self._readers[slice_id] = reader
        self.reader_ready.emit(slice_id, reader)

    def _handle_reader_failed(self, slice_id: int, msg: str) -> None:
        self._reader_workers.pop(slice_id, None)
        logger.warning("RemoteSliceReader failed for z%02d: %s", slice_id, msg)
        self.reader_failed.emit(slice_id, msg)

    def _handle_cs_ready(self, slice_id: int, axis: str, pos: int, img: object) -> None:
        img_arr = np.asarray(img)
        self._cache.setdefault(slice_id, {}).setdefault(axis, {})[pos] = img_arr
        self.cross_section_ready.emit(slice_id, axis, pos, img_arr)

    def _handle_cs_failed(self, slice_id: int, axis: str, pos: int, msg: str) -> None:
        logger.warning("Cross-section fetch failed z%02d %s[%d]: %s", slice_id, axis, pos, msg)
        self.cross_section_failed.emit(slice_id, axis, pos, msg)
