"""Headless napari pair-switch memory benchmark (NAPI-04).

Git-based before/after procedure (D-18)
--------------------------------------
1. Check out the pre-Phase-9 baseline commit (last commit before napari_layers
   extraction), e.g. the parent of the first ``09-02`` commit.
2. Run::

       uv run pytest tests/test_napari_benchmark.py -m napari_benchmark -s

   Record the printed snapshot line (baseline_rss_mb, peak_rss_mb, rss_delta_mb,
   mean_switch_s).
3. Check out HEAD (post-Phase-9) and repeat step 2.
4. Compare peak RSS delta and mean switch time — relative growth cap is the gate
   (D-21); no committed golden numbers required.
"""

from __future__ import annotations

import contextlib
import gc
import logging
import time
from pathlib import Path

import numpy as np
import psutil
import pytest

from linumpy_manual_align.io.image_utils import ENHANCE_NONE, OVERLAY_COLOR
from linumpy_manual_align.io.transform_io import discover_aips
from linumpy_manual_align.state import AlignmentState, UndoStack
from linumpy_manual_align.ui import napari_layers
from linumpy_manual_align.ui.widget_overlay import OverlayStateMixin
from linumpy_manual_align.ui.widget_pair_loading import PairLoadingMixin
from linumpy_manual_align.ui.widget_ui import UiHelpersMixin

# Relative RSS growth cap — first green run calibrates; not an SLA (D-21).
RSS_DELTA_THRESHOLD_MB = 50.0


class _SpinStub:
    def __init__(self, value: int = 0) -> None:
        self._value = value
        self._min = 0
        self._max = 100

    def value(self) -> int:
        return self._value

    def setValue(self, value: int) -> None:
        self._value = value

    def minimum(self) -> int:
        return self._min

    def maximum(self) -> int:
        return self._max


class _ComboStub:
    def setCurrentIndex(self, _idx: int) -> None:
        pass


class _BenchmarkWidget(UiHelpersMixin, PairLoadingMixin, OverlayStateMixin):
    pass


def _make_benchmark_widget(viewer, pkg: Path, output_dir: Path) -> _BenchmarkWidget:
    widget = object.__new__(_BenchmarkWidget)
    widget.viewer = viewer
    widget.output_dir = output_dir
    widget.level = 1
    widget.pairs = [(0, 1), (1, 2), (2, 3)]
    widget.current_pair_idx = 0
    widget._projection_mode = "xy"
    widget._overlay_mode = OVERLAY_COLOR
    widget._enhance_mode = ENHANCE_NONE
    widget.aips_dir = pkg / "aips"
    widget.slice_paths = discover_aips(widget.aips_dir)
    widget.pair_paths_xy = {}
    widget.pair_paths_xz = {}
    widget.pair_paths_yz = {}
    widget.slice_paths_xz = {}
    widget.slice_paths_yz = {}
    widget.existing_transforms = {}
    widget.saved_pairs = set()
    widget.unsaved_changes = set()
    widget.undo_stacks = {}
    widget._current_offsets = {}
    widget.pair_centers = {}
    widget.fixed_layer = None
    widget.moving_layer = None
    widget._composite_layer = None
    widget._raw_fixed_aip = None
    widget._raw_moving_aip = None
    widget._original_fixed_aip = None
    widget._original_moving_aip = None
    widget._moving_scale_yx = [1.0, 1.0]
    widget._content_bbox_wl = None
    widget._crop_rc = (0, 0)
    widget.spin_tile = _SpinStub(32)
    widget.spin_tx = type("S", (), {"value": lambda self: 0.0, "setValue": lambda self, v: None})()
    widget.spin_ty = type("S", (), {"value": lambda self: 0.0, "setValue": lambda self, v: None})()
    widget.spin_rot = type("S", (), {"value": lambda self: 0.0, "setValue": lambda self, v: None})()
    widget.rot_slider = type("S", (), {"setValue": lambda self, v: None})()
    widget.spin_fixed_z = _SpinStub(0)
    widget.spin_moving_z = _SpinStub(0)
    widget.pair_combo = _ComboStub()
    widget._cs_mgr = type("CS", (), {"slices_remote_dir": None, "slice_remote_paths": {}})()
    widget._update_status = lambda: None
    widget._update_z_relative_label = lambda: None
    widget._set_cs_sliders_visible = lambda _visible: None
    return widget


@pytest.mark.napari_benchmark
def test_pair_switch_memory_benchmark(
    make_napari_viewer,
    fake_multi_pair_package: Path,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """50 XY pair switches through real _load_pair incremental path (NAPI-04)."""
    viewer = make_napari_viewer(show=False)
    widget = _make_benchmark_widget(viewer, fake_multi_pair_package, tmp_path)
    num_pairs = len(widget.pairs)

    incremental_hits = 0
    original_gate = widget._can_incremental_update

    def _counting_gate(**kwargs: object) -> bool:
        nonlocal incremental_hits
        result = original_gate(**kwargs)  # type: ignore[arg-type]
        if result:
            incremental_hits += 1
        return result

    widget._can_incremental_update = _counting_gate  # type: ignore[method-assign]

    gc.collect()
    process = psutil.Process()
    baseline_rss_mb = process.memory_info().rss / (1024 * 1024)
    peak_rss_mb = baseline_rss_mb
    switch_times: list[float] = []

    with caplog.at_level(logging.WARNING):
        for i in range(50):
            idx = i % num_pairs
            t0 = time.perf_counter()
            widget._load_pair(idx)
            switch_times.append(time.perf_counter() - t0)
            napari_layers.assert_layer_inventory(
                viewer,
                projection_mode=widget._projection_mode,
                overlay_mode=widget._overlay_mode,
                overlay_color=OVERLAY_COLOR,
            )
            gc.collect()
            rss_mb = process.memory_info().rss / (1024 * 1024)
            peak_rss_mb = max(peak_rss_mb, rss_mb)

    assert incremental_hits > 0, "benchmark must exercise incremental update_pair_data path"
    assert len(viewer.layers) in (2, 3)
    assert not caplog.records, "inventory guard should stay silent in stable color mode"

    rss_delta_mb = peak_rss_mb - baseline_rss_mb
    mean_switch_s = sum(switch_times) / len(switch_times)

    print(
        f"NAPI-04 snapshot: baseline_rss_mb={baseline_rss_mb:.1f} "
        f"peak_rss_mb={peak_rss_mb:.1f} rss_delta_mb={rss_delta_mb:.1f} "
        f"mean_switch_s={mean_switch_s:.4f} incremental_hits={incremental_hits}"
    )

    assert rss_delta_mb < RSS_DELTA_THRESHOLD_MB
    assert mean_switch_s < 5.0
