"""Mixins extracted from :class:`~linumpy_manual_align.ui.widget.ManualAlignWidget` for clearer structure."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

from linumpy_manual_align.io.transform_io import get_metric, load_pairwise_metrics


class _PairNavHost(Protocol):
    """Structural type for :class:`ManualAlignWidget` attributes used by :class:`PairNavigationMixin`."""

    pairs: list[tuple[int, int]]
    slice_ids: list[int]
    _filter_slices: list[int] | None
    existing_transforms: dict[int, Path]
    current_pair_idx: int

    def _load_pair_preserve_camera(self, idx: int) -> None: ...


class PairNavigationMixin:
    """Slice-pair list construction and prev/next navigation."""

    def _build_pairs(self: _PairNavHost) -> None:
        """Populate ``self.pairs`` from ``self.slice_ids``, respecting any active slice filter."""
        self.pairs = [
            (self.slice_ids[i], self.slice_ids[i + 1])
            for i in range(len(self.slice_ids) - 1)
            if self._filter_slices is None or self.slice_ids[i + 1] in self._filter_slices
        ]

    def _pair_label(self: _PairNavHost, fid: int, mid: int) -> str:
        """Return the display label for a pair (fid → mid), including metrics if available."""
        label = f"z{fid:02d} → z{mid:02d}"
        _cs_mgr = getattr(self, "_cs_mgr", None)
        _interp = getattr(_cs_mgr, "interpolated_slice_ids", set())
        if fid in _interp or mid in _interp:
            label += "  [interp]"
        if mid in self.existing_transforms:
            metrics_path = self.existing_transforms[mid] / "pairwise_registration_metrics.json"
            metrics = load_pairwise_metrics(metrics_path)
            mag = get_metric(metrics, "translation_magnitude")
            if mag is not None:
                label += f"  ({mag:.0f}px)"
        return label

    def _prev_pair(self: _PairNavHost) -> None:
        if self.current_pair_idx > 0:
            self._load_pair_preserve_camera(self.current_pair_idx - 1)

    def _next_pair(self: _PairNavHost) -> None:
        if self.current_pair_idx < len(self.pairs) - 1:
            self._load_pair_preserve_camera(self.current_pair_idx + 1)
