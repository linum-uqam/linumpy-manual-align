"""Interactive XZ/YZ cross-sections (remote OME-Zarr)."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from linumpy_manual_align.io.image_utils import enhance_aip, normalize_aip
from linumpy_manual_align.ui.widget_typing import ManualAlignWidget

logger = logging.getLogger(__name__)


class CrossSectionMixin:
    def _update_initial_cs_position(self: ManualAlignWidget, fid: int, mid: int) -> None:
        """Resolve the cross-section slider position for this pair.

        All values stored in ``_cross_section_y``/``_cross_section_x`` and
        ``_cs_positions`` are **slider values** (moving-frame).  The moving
        offset is applied only in the NPZ-centroid path (priority 3), which
        converts the fixed-frame centroid into a moving-frame slider value.

        Priority:
        1. ``_cs_positions[mid]`` — in-memory (set when the user or code moves
           the slider; survives XY↔Z switches within a session).
        2. ``cs_position.txt`` on disk — persists across reloads.
        3. NPZ tissue centroid + moving offset — first-time default.
        4. 0 (fallback).
        """
        pair_key = (fid, mid)
        axis = self._projection_mode  # "xz" or "yz"
        pair_store = self.pair_paths_xz if axis == "xz" else self.pair_paths_yz
        per_slice_store = self.slice_paths_xz if axis == "xz" else self.slice_paths_yz

        npz_path: Path | None = None
        if pair_key in pair_store and "fixed" in pair_store[pair_key]:
            npz_path = pair_store[pair_key]["fixed"]
        elif fid in per_slice_store:
            npz_path = per_slice_store[fid]

        # Always read center_pos from the fixed NPZ — it is a fixed property of the
        # data (where the static cross-section was extracted) and must be loaded
        # regardless of which priority path resolves the moving slider position.
        if npz_path is not None:
            try:
                data = np.load(str(npz_path))
                if "center_pos" in data:
                    cp = int(data["center_pos"])
                    if axis == "xz":
                        self._fixed_cs_pos = (cp, self._fixed_cs_pos[1])
                    else:
                        self._fixed_cs_pos = (self._fixed_cs_pos[0], cp)
            except Exception:
                pass

        self._cross_section_y = 0
        self._cross_section_x = 0

        # Priority 1: in-memory (already a slider value)
        if mid in self._cs_positions:
            self._cross_section_y, self._cross_section_x = self._cs_positions[mid]
            return

        # Priority 2: disk (already a slider value)
        if self.output_dir is not None:
            cs_path = self.output_dir / f"slice_z{mid:02d}" / "cs_position.txt"
            if cs_path.exists():
                try:
                    vals = np.loadtxt(str(cs_path), dtype=int)
                    if vals.size >= 2:
                        self._cross_section_y = int(vals[0])
                        self._cross_section_x = int(vals[1])
                        self._cs_positions[mid] = (self._cross_section_y, self._cross_section_x)
                        return
                except Exception:
                    pass

        # Priority 3: NPZ centroid (fixed-frame) → convert to slider value
        if npz_path is None:
            return

        try:
            data = np.load(str(npz_path))
            if "center_pos" not in data:
                return
            cp = int(data["center_pos"])
            slider_val = cp + self._cs_moving_offset(axis)
            if axis == "xz":
                self._cross_section_y = slider_val
            else:
                self._cross_section_x = slider_val
        except Exception:
            pass

    def _ensure_readers_for_pair(self: ManualAlignWidget) -> None:
        """Start a SliceReaderWorker for the moving slice of the current pair.

        The fixed slice is always shown from the static downloaded NPZ — no reader needed.
        """
        if not self.pairs or self.server_config is None:
            return
        _fid, mid = self.pairs[self.current_pair_idx]
        if self._cs_mgr.has_reader(mid):
            return
        self._cs_mgr.ensure_reader(self.server_config, mid)
        self._cs_loading_label.setText("Opening remote reader for moving slice…")

    def _init_cs_slider_from_reader(self: ManualAlignWidget, mid: int) -> None:
        """Initialise the active cross-section slider from the already-open reader for *mid*.

        ``_cross_section_y``/``_cross_section_x`` already hold the target
        slider value (moving-frame), set earlier by ``_update_initial_cs_position``.
        This method just configures the slider range, sets the value, and fires
        the first cross-section request.
        """
        shape = self._cs_mgr.reader_shape(mid)
        if shape is None:
            return
        self._cs_loading_label.setText("")
        _nz, ny, nx = shape
        axis = self._projection_mode  # "xz" or "yz"

        # Scale brain content bounds from working-level pixels to cs_level pixels
        # so the slider spans only the tissue region rather than the full dark border.
        cs_scale = 2 ** (self.level - self._cs_mgr.cs_level)
        if self._content_bbox_wl is not None:
            br1, bc1, br2, bc2 = self._content_bbox_wl
            y_lo = max(0, round(br1 * cs_scale))
            y_hi = min(ny - 1, round((br2 - 1) * cs_scale))
            x_lo = max(0, round(bc1 * cs_scale))
            x_hi = min(nx - 1, round((bc2 - 1) * cs_scale))
        else:
            y_lo, y_hi = 0, ny - 1
            x_lo, x_hi = 0, nx - 1

        # Fixed reference slider position (working-level → cs_level)
        fixed_y_cs = max(y_lo, min(y_hi, round(self._fixed_cs_pos[0] * cs_scale)))
        fixed_x_cs = max(x_lo, min(x_hi, round(self._fixed_cs_pos[1] * cs_scale)))

        if axis == "xz":
            # Fixed reference
            self.slider_fixed_y.setMinimum(y_lo)
            self.slider_fixed_y.setMaximum(y_hi)
            self.slider_fixed_y.setValue(fixed_y_cs)
            self._lbl_fixed_y.setText(str(fixed_y_cs))
            # Moving slider
            init = max(y_lo, min(y_hi, self._cross_section_y))
            self.slider_cs_y.blockSignals(True)
            self.slider_cs_y.setMinimum(y_lo)
            self.slider_cs_y.setMaximum(y_hi)
            self.slider_cs_y.setValue(init)
            self.slider_cs_y.blockSignals(False)
            self._lbl_cs_y.setText(str(init))
            self._cross_section_y = init
        else:
            # Fixed reference
            self.slider_fixed_x.setMinimum(x_lo)
            self.slider_fixed_x.setMaximum(x_hi)
            self.slider_fixed_x.setValue(fixed_x_cs)
            self._lbl_fixed_x.setText(str(fixed_x_cs))
            # Moving slider
            init = max(x_lo, min(x_hi, self._cross_section_x))
            self.slider_cs_x.blockSignals(True)
            self.slider_cs_x.setMinimum(x_lo)
            self.slider_cs_x.setMaximum(x_hi)
            self.slider_cs_x.setValue(init)
            self.slider_cs_x.blockSignals(False)
            self._lbl_cs_x.setText(str(init))
            self._cross_section_x = init

        # Persist so that XY↔Z round-trips and pair switches restore this value
        # even when the user never manually moves the slider.
        self._cs_positions[mid] = (self._cross_section_y, self._cross_section_x)

        self._cs_mgr.request(mid, axis, init)

    def _on_reader_ready(self: ManualAlignWidget, sid: int, reader: object) -> None:
        """Called when ``CrossSectionManager`` finishes opening a reader."""
        if not self.pairs:
            return
        _fid, mid = self.pairs[self.current_pair_idx]
        if sid != mid:
            return
        self._init_cs_slider_from_reader(mid)

    def _on_reader_failed(self: ManualAlignWidget, sid: int, msg: str) -> None:
        """Called when a remote reader fails to open."""
        self._cs_loading_label.setText(f"<span style='color:red;'>Reader failed for z{sid:02d}</span>")

    def _on_cross_section_ready(self: ManualAlignWidget, sid: int, axis: str, pos: int, img: object) -> None:
        """Refresh display when a cross-section image arrives in the manager cache."""
        if not self.pairs:
            return
        fid, mid = self.pairs[self.current_pair_idx]
        if sid not in (fid, mid):
            return
        if self._projection_mode != axis:
            return

        # Enable the appropriate slider now that we have real data
        if axis == "xz":
            self.slider_cs_y.setEnabled(True)
        else:
            self.slider_cs_x.setEnabled(True)

        # Only replace the moving layer when this completion matches the slider.
        # Prefetch and out-of-order SSH responses must not overwrite the static
        # NPZ (or a prior correct frame) with a different column.
        if sid != mid:
            return
        slider_pos = self.slider_cs_y.value() if axis == "xz" else self.slider_cs_x.value()
        if pos != slider_pos:
            return
        if self.moving_layer is not None:
            self.moving_layer.data = enhance_aip(normalize_aip(np.asarray(img, dtype=np.float32)), self._enhance_mode)

    def _on_cross_section_failed(self: ManualAlignWidget, sid: int, axis: str, pos: int, msg: str) -> None:
        logger.warning(f"Cross-section fetch failed z{sid:02d} {axis}[{pos}]: {msg}")

    def _cs_moving_offset(self: ManualAlignWidget, axis: str) -> int:
        """Pixel offset to add to the fixed slider position to show matching tissue in the moving slice.

        The moving slice has been shifted by (tx, ty) relative to the fixed slice (at working
        resolution self.level).  To see the same anatomical tissue in both cross-sections we must
        fetch the moving slice at a different column:
          XZ view (Y slider):  moving_y = fixed_y - ty * scale
          YZ view (X slider):  moving_x = fixed_x - tx * scale
        where scale = 2 ** (self.level - cs_level) converts from working-res pixels to
        cross-section-level pixels.
        """
        state = self._current_state()
        scale = 2 ** (self.level - self._cs_mgr.cs_level)
        if axis == "xz":
            return -round(state.ty * scale)
        return -round(state.tx * scale)

    def _on_cs_slider_settled(self: ManualAlignWidget) -> None:
        """Called after the debounce timer fires — fetch moving slice at the new position."""
        if not self.pairs or self._projection_mode == "xy":
            return
        _fid, mid = self.pairs[self.current_pair_idx]
        axis = self._projection_mode
        # Use slider value directly: it is already in moving-frame.
        pos = self.slider_cs_y.value() if axis == "xz" else self.slider_cs_x.value()
        self._cs_mgr.request(mid, axis, pos)
        self._cs_mgr.prefetch_around(mid, axis, pos)
        self._save_cs_position(mid)

    def _save_cs_position(self: ManualAlignWidget, mid: int) -> None:
        """Persist the cross-section position for *mid* to disk so it survives reloads."""
        if self.output_dir is None:
            return
        cs_y, cs_x = self._cs_positions.get(mid, (self._cross_section_y, self._cross_section_x))
        out_dir = self.output_dir / f"slice_z{mid:02d}"
        out_dir.mkdir(parents=True, exist_ok=True)
        np.savetxt(str(out_dir / "cs_position.txt"), [cs_y, cs_x], fmt="%d")

    def _nudge_cs_position(self: ManualAlignWidget, delta: int) -> None:
        """Shift the active cross-section slider (moving Y in XZ, moving X in YZ) by *delta* pixels.

        Bound to Alt+, / Alt-. (coarse step) and Ctrl+, / Ctrl-. (fine step from settings).
        """
        if self._projection_mode == "xz":
            new_val = max(0, min(self.slider_cs_y.maximum(), self.slider_cs_y.value() + delta))
            self.slider_cs_y.setValue(new_val)
        elif self._projection_mode == "yz":
            new_val = max(0, min(self.slider_cs_x.maximum(), self.slider_cs_x.value() + delta))
            self.slider_cs_x.setValue(new_val)

    def _load_remote_cs_metadata(self: ManualAlignWidget, pkg_root: Path) -> None:
        """Delegate metadata loading to the CrossSectionManager."""
        self._cs_mgr.load_metadata(pkg_root)
