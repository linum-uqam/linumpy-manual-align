"""Tests for CrossSectionManager: cache, metadata loading, prefetch."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from linumpy_manual_align.cross_section import CrossSectionManager


@pytest.fixture()
def mgr(qapp) -> CrossSectionManager:
    """Create a fresh manager for each test (requires a QApplication)."""
    return CrossSectionManager()


# ---------------------------------------------------------------------------
# Metadata loading
# ---------------------------------------------------------------------------


class TestLoadMetadata:
    def test_loads_from_pkg_root(self, tmp_path: Path, qapp) -> None:
        meta = {
            "slices_remote_dir": "/remote/slices",
            "cross_section_level": 2,
            "slice_filenames": {"0": "slice_z00_norm.ome.zarr"},
        }
        (tmp_path / "manual_align_metadata.json").write_text(json.dumps(meta))
        mgr = CrossSectionManager()
        mgr.load_metadata(tmp_path)
        assert mgr.slices_remote_dir == "/remote/slices"
        assert mgr.cs_level == 2
        assert mgr.slice_filenames[0] == "slice_z00_norm.ome.zarr"

    def test_loads_from_parent(self, tmp_path: Path, qapp) -> None:
        sub = tmp_path / "aips"
        sub.mkdir()
        meta = {"slices_remote_dir": "/other", "cross_section_level": 0}
        (tmp_path / "manual_align_metadata.json").write_text(json.dumps(meta))
        mgr = CrossSectionManager()
        mgr.load_metadata(sub)
        assert mgr.slices_remote_dir == "/other"

    def test_prefers_pkg_root_over_parent(self, tmp_path: Path, qapp) -> None:
        sub = tmp_path / "aips"
        sub.mkdir()
        (sub / "manual_align_metadata.json").write_text(json.dumps({"slices_remote_dir": "LOCAL"}))
        (tmp_path / "manual_align_metadata.json").write_text(json.dumps({"slices_remote_dir": "PARENT"}))
        mgr = CrossSectionManager()
        mgr.load_metadata(sub)
        assert mgr.slices_remote_dir == "LOCAL"

    def test_missing_metadata_leaves_defaults(self, tmp_path: Path, qapp) -> None:
        mgr = CrossSectionManager()
        mgr.load_metadata(tmp_path)  # no file
        assert mgr.slices_remote_dir is None
        assert mgr.cs_level == 0

    def test_broken_json_leaves_defaults(self, tmp_path: Path, qapp) -> None:
        (tmp_path / "manual_align_metadata.json").write_text("not valid json {{{")
        mgr = CrossSectionManager()
        mgr.load_metadata(tmp_path)
        assert mgr.slices_remote_dir is None

    def test_pyramid_level_fallback_key(self, tmp_path: Path, qapp) -> None:
        meta = {"slices_remote_dir": "/r", "pyramid_level": 3}
        (tmp_path / "manual_align_metadata.json").write_text(json.dumps(meta))
        mgr = CrossSectionManager()
        mgr.load_metadata(tmp_path)
        assert mgr.cs_level == 3


# ---------------------------------------------------------------------------
# Cache operations
# ---------------------------------------------------------------------------


class TestCache:
    def test_get_cached_returns_none_when_empty(self, mgr: CrossSectionManager) -> None:
        assert mgr.get_cached(0, "xz", 50) is None

    def test_cross_section_ready_updates_cache(self, mgr: CrossSectionManager) -> None:
        img = np.ones((32, 32), dtype=np.float32)
        mgr._handle_cs_ready(0, "xz", 50, img)
        cached = mgr.get_cached(0, "xz", 50)
        assert cached is not None
        np.testing.assert_array_equal(cached, img)

    def test_cache_keyed_by_slice_axis_pos(self, mgr: CrossSectionManager) -> None:
        img_a = np.zeros((8, 8), dtype=np.float32)
        img_b = np.ones((8, 8), dtype=np.float32)
        mgr._handle_cs_ready(0, "xz", 10, img_a)
        mgr._handle_cs_ready(0, "yz", 10, img_b)
        assert np.all(mgr.get_cached(0, "xz", 10) == 0)
        assert np.all(mgr.get_cached(0, "yz", 10) == 1)

    def test_cross_section_ready_emits_signal(self, mgr: CrossSectionManager) -> None:
        received: list = []
        mgr.cross_section_ready.connect(lambda sid, ax, pos, img: received.append((sid, ax, pos)))
        mgr._handle_cs_ready(1, "yz", 30, np.zeros((4, 4)))
        assert received == [(1, "yz", 30)]

    def test_prefetch_around_evicts_stale_entries(self, mgr: CrossSectionManager) -> None:
        reader = MagicMock()
        reader.shape = (10, 500, 500)
        mgr._readers[0] = reader

        # Prime cache with entries far from future position.
        for pos in range(0, 100, 10):
            mgr._handle_cs_ready(0, "xz", pos, np.zeros((4, 4)))

        # Simulate prefetch around pos=400 — entries near 0 should be evicted.
        with patch.object(mgr, "request"):
            mgr.prefetch_around(0, "xz", 400)

        # Entries near 0 are > 150 px away from 400 → evicted.
        assert mgr.get_cached(0, "xz", 0) is None

    def test_prefetch_around_no_reader_is_noop(self, mgr: CrossSectionManager) -> None:
        """Should not raise if no reader is open for the slice."""
        mgr.prefetch_around(99, "xz", 100)  # no reader for sid=99


# ---------------------------------------------------------------------------
# Reader lifecycle
# ---------------------------------------------------------------------------


class TestReaderLifecycle:
    def test_has_reader_false_initially(self, mgr: CrossSectionManager) -> None:
        assert not mgr.has_reader(0)

    def test_has_reader_true_after_ready(self, mgr: CrossSectionManager) -> None:
        reader = MagicMock()
        mgr._handle_reader_ready(0, reader)
        assert mgr.has_reader(0)

    def test_reader_shape_none_without_reader(self, mgr: CrossSectionManager) -> None:
        assert mgr.reader_shape(0) is None

    def test_reader_shape_returns_shape(self, mgr: CrossSectionManager) -> None:
        reader = MagicMock()
        reader.shape = (10, 200, 300)
        mgr._handle_reader_ready(0, reader)
        assert mgr.reader_shape(0) == (10, 200, 300)

    def test_reader_ready_removes_worker(self, mgr: CrossSectionManager) -> None:
        worker = MagicMock()
        mgr._reader_workers[5] = worker
        mgr._handle_reader_ready(5, MagicMock())
        assert 5 not in mgr._reader_workers

    def test_reader_failed_removes_worker(self, mgr: CrossSectionManager) -> None:
        worker = MagicMock()
        mgr._reader_workers[3] = worker
        mgr._handle_reader_failed(3, "connection refused")
        assert 3 not in mgr._reader_workers

    def test_reader_ready_emits_signal(self, mgr: CrossSectionManager) -> None:
        received: list = []
        mgr.reader_ready.connect(lambda sid, r: received.append(sid))
        reader = MagicMock()
        mgr._handle_reader_ready(7, reader)
        assert received == [7]

    def test_reader_failed_emits_signal(self, mgr: CrossSectionManager) -> None:
        received: list = []
        mgr.reader_failed.connect(lambda sid, msg: received.append(sid))
        mgr._handle_reader_failed(2, "error")
        assert received == [2]

    def test_ensure_reader_skips_if_already_open(self, mgr: CrossSectionManager) -> None:
        mgr._readers[0] = MagicMock()
        mgr.slices_remote_dir = "/remote"
        server_cfg = MagicMock()
        with patch("linumpy_manual_align.cross_section.SliceReaderWorker") as MockW:
            mgr.ensure_reader(server_cfg, 0)
            MockW.assert_not_called()

    def test_ensure_reader_skips_if_no_remote_dir(self, mgr: CrossSectionManager) -> None:
        mgr.slices_remote_dir = None
        server_cfg = MagicMock()
        with patch("linumpy_manual_align.cross_section.SliceReaderWorker") as MockW:
            mgr.ensure_reader(server_cfg, 0)
            MockW.assert_not_called()


# ---------------------------------------------------------------------------
# Close
# ---------------------------------------------------------------------------


class TestCloseAll:
    def test_close_all_clears_readers(self, mgr: CrossSectionManager) -> None:
        reader = MagicMock()
        mgr._readers[0] = reader
        mgr.close_all()
        reader.close.assert_called_once()
        assert len(mgr._readers) == 0

    def test_close_all_clears_workers(self, mgr: CrossSectionManager) -> None:
        worker = MagicMock()
        mgr._reader_workers[0] = worker
        mgr.close_all()
        assert len(mgr._reader_workers) == 0
