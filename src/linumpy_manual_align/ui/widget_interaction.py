"""Keybindings, display mode handlers, nudges, camera, and widget close cleanup."""

from __future__ import annotations

import numpy as np
from qtpy.QtCore import QTimer

from linumpy_manual_align.io.image_utils import (
    ENHANCE_CLAHE,
    ENHANCE_EDGES,
    ENHANCE_NONE,
    ENHANCE_SHARPEN,
    OVERLAY_CHECKER,
    OVERLAY_COLOR,
    OVERLAY_DIFF,
    enhance_aip,
    normalize_aip,
)
from linumpy_manual_align.settings_runtime import keybindings_from_settings
from linumpy_manual_align.ui.widget_typing import ManualAlignWidget


class InteractionMixin:
    def _install_keybindings(self: ManualAlignWidget) -> None:
        for key, method_name, args in keybindings_from_settings():
            method = getattr(self, method_name)
            self.viewer.bind_key(key, lambda _v, m=method, a=args: m(*a), overwrite=True)

        # Undo/redo use dedicated macOS-aware labels, so keep them explicit
        self.viewer.bind_key("Control-z", lambda _v: self._undo(), overwrite=True)
        self.viewer.bind_key("Control-Shift-z", lambda _v: self._redo(), overwrite=True)

    def _on_overlay_mode_changed(self: ManualAlignWidget, _: int = 0) -> None:
        self._overlay_mode = [OVERLAY_COLOR, OVERLAY_DIFF, OVERLAY_CHECKER][self.combo_overlay.currentIndex()]
        is_checker = self._overlay_mode == OVERLAY_CHECKER
        self._tile_row_label.setVisible(is_checker)
        self.spin_tile.setVisible(is_checker)
        self._rebuild_layer_visibility()
        if self._overlay_mode != OVERLAY_COLOR:
            state = self._current_state()
            self._refresh_composite(state)

    def _on_tile_size_changed(self: ManualAlignWidget, _: int = 0) -> None:
        if self._overlay_mode == OVERLAY_CHECKER:
            state = self._current_state()
            self._refresh_composite(state)

    def _on_enhance_changed(self: ManualAlignWidget, _: int = 0) -> None:
        """Re-enhance both AIPs from the stored raw data and refresh the display."""
        modes = [ENHANCE_NONE, ENHANCE_EDGES, ENHANCE_CLAHE, ENHANCE_SHARPEN]
        self._enhance_mode = modes[self.combo_enhance.currentIndex()]

        if self._raw_fixed_aip is None or self._raw_moving_aip is None:
            return

        self._original_fixed_aip = enhance_aip(self._raw_fixed_aip, self._enhance_mode)
        self._original_moving_aip = enhance_aip(self._raw_moving_aip, self._enhance_mode)

        if self.fixed_layer is not None:
            self.fixed_layer.data = self._original_fixed_aip

        if self.moving_layer is not None:
            if self._projection_mode == "xy":
                # XY: bake rotation + translation into pixel data
                state = self._current_state()
                self._apply_state(state, push=False)
            else:
                # XZ/YZ moving layer: if the remote cross-section is loaded use that
                # (enhanced), otherwise fall back to the static NPZ slice.
                if self.pairs:
                    _fid, mid = self.pairs[self.current_pair_idx]
                    axis = self._projection_mode
                    slider_pos = self.slider_cs_y.value() if axis == "xz" else self.slider_cs_x.value()
                    cached = self._cs_mgr.get_cached(mid, axis, slider_pos)
                    if cached is not None:
                        self.moving_layer.data = enhance_aip(normalize_aip(cached.astype(np.float32)), self._enhance_mode)
                    else:
                        self.moving_layer.data = self._original_moving_aip

        # Refresh composite overlay (Difference / Checkerboard) if active
        if self._overlay_mode != OVERLAY_COLOR and self._composite_layer is not None:
            state = self._current_state()
            self._refresh_composite(state)

    def _nudge_translate(self: ManualAlignWidget, dx: float, dy: float) -> None:
        if self._projection_mode != "xy" and dy != 0:
            # In Z alignment mode Up/Down adjusts the moving Z-overlap voxel index.
            # Pressing Up (dy=-1) decreases moving_z, which decreases dz_display
            # (= moving_z - fixed_z), which moves the layer translate upward.
            self._nudge_z_offset(int(dy))
            return
        state = self._current_state()
        if self._projection_mode == "yz":
            # In YZ mode the horizontal axis maps to ty (Y), not tx (X).
            state.ty += dx
        else:
            state.tx += dx
            state.ty += dy
        self._apply_state(state, push=True)
        self._update_status()

    def _nudge_z_offset(self: ManualAlignWidget, delta: int) -> None:
        """Adjust the Z-overlap by *delta* voxels (Z alignment mode only).

        The visual overlap is ``dz_display = (moving_z - fixed_z) / scale``.
        Pressing Up (delta=-1) should move the moving layer up, i.e. decrease
        dz_display, so we *increase* fixed_z.

        We adjust **fixed_z first** because it controls where the overlap
        starts in the fixed volume (``overlap = nz_fixed - fixed_z``).
        ``moving_z`` represents the noisy-data skip in the moving volume and
        should stay at its pipeline-assigned value.  Only when fixed_z hits a
        bound does the overflow spill into moving_z.
        """
        if not self.pairs:
            return
        new_fixed = self.spin_fixed_z.value() - delta
        lo, hi = self.spin_fixed_z.minimum(), self.spin_fixed_z.maximum()
        if new_fixed < lo:
            overflow = lo - new_fixed
            self.spin_fixed_z.setValue(lo)
            new_moving = max(self.spin_moving_z.minimum(), self.spin_moving_z.value() + overflow)
            self.spin_moving_z.setValue(new_moving)
        elif new_fixed > hi:
            overflow = new_fixed - hi
            self.spin_fixed_z.setValue(hi)
            new_moving = min(self.spin_moving_z.maximum(), self.spin_moving_z.value() - overflow)
            self.spin_moving_z.setValue(new_moving)
        else:
            self.spin_fixed_z.setValue(new_fixed)

    def _restore_camera(self: ManualAlignWidget, zoom: float, center: tuple) -> None:
        """Re-apply a previously snapshotted camera state."""
        self.viewer.camera.zoom = zoom
        self.viewer.camera.center = center

    def _load_pair_preserve_camera(self: ManualAlignWidget, idx: int) -> None:
        """Load a pair while keeping the current zoom level and position."""
        zoom = self.viewer.camera.zoom
        center = tuple(self.viewer.camera.center)
        self._load_pair(idx, preserve_camera=True)
        QTimer.singleShot(0, lambda: self._restore_camera(zoom, center))

    def _toggle_z_proj(self: ManualAlignWidget) -> None:
        """Switch between XZ and YZ views (Z alignment mode only)."""
        if self._projection_mode == "xy":
            return
        # idClicked only fires on real mouse clicks; block signals and call
        # the handler directly so the display is updated on keyboard use too.
        if self._projection_mode == "xz":
            self._proj_btn_group.blockSignals(True)
            self._btn_proj_yz.setChecked(True)
            self._proj_btn_group.blockSignals(False)
            self._on_z_proj_changed(1)
        else:
            self._proj_btn_group.blockSignals(True)
            self._btn_proj_xz.setChecked(True)
            self._proj_btn_group.blockSignals(False)
            self._on_z_proj_changed(0)

    def _toggle_alignment_mode(self: ManualAlignWidget) -> None:
        """Toggle between XY Alignment and Z Alignment modes."""
        # Block signals on both buttons to prevent re-entrant toggled calls,
        # set the visual state, then invoke the handler once directly.
        if self._projection_mode == "xy":
            if not self._btn_mode_z.isEnabled():
                return
            self._btn_mode_xy.blockSignals(True)
            self._btn_mode_z.blockSignals(True)
            self._btn_mode_xy.setChecked(False)
            self._btn_mode_z.setChecked(True)
            self._btn_mode_xy.blockSignals(False)
            self._btn_mode_z.blockSignals(False)
            self._on_mode_btn_toggled("z", True)
        else:
            self._btn_mode_xy.blockSignals(True)
            self._btn_mode_z.blockSignals(True)
            self._btn_mode_xy.setChecked(True)
            self._btn_mode_z.setChecked(False)
            self._btn_mode_xy.blockSignals(False)
            self._btn_mode_z.blockSignals(False)
            self._on_mode_btn_toggled("xy", True)

    def _nudge_rotate(self: ManualAlignWidget, delta_deg: float) -> None:
        state = self._current_state()
        state.rotation += delta_deg
        self._apply_state(state, push=True)
        self._update_status()

    def closeEvent(self: ManualAlignWidget, event: object) -> None:
        """Terminate all persistent SSH readers on widget close."""
        self._cs_mgr.close_all()
        super().closeEvent(event)  # type: ignore[arg-type]
