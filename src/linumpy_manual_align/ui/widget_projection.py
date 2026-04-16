"""Spinbox, pair combo, projection mode, and Z-offset handlers."""

from __future__ import annotations

from linumpy_manual_align.state import AlignmentState
from linumpy_manual_align.ui.widget_typing import ManualAlignWidget


class ProjectionEventMixin:
    def _on_pair_changed(self: ManualAlignWidget, idx: int) -> None:
        if idx >= 0 and idx != self.current_pair_idx:
            self._load_pair_preserve_camera(idx)

    def _on_spinbox_changed(self: ManualAlignWidget) -> None:
        if self._suppress_spinbox_event:
            return
        state = self._current_state()
        self._apply_state(state, push=True)
        self._update_status()

    def _on_rotation_changed(self: ManualAlignWidget) -> None:
        if self._suppress_spinbox_event:
            return
        state = self._current_state()
        with self._suppress_events():
            self.rot_slider.setValue(int(state.rotation * 10))
        self._apply_state(state, push=True)
        self._update_status()

    def _on_rotation_slider_changed(self: ManualAlignWidget, value: int) -> None:
        if self._suppress_spinbox_event:
            return
        rot = value / 10.0
        with self._suppress_events():
            self.spin_rot.setValue(rot)
        state = AlignmentState(tx=self.spin_tx.value(), ty=self.spin_ty.value(), rotation=rot)
        self._apply_state(state, push=True)
        self._update_status()

    # ----- Projection and Z-offset handlers -----

    def _on_mode_btn_toggled(self: ManualAlignWidget, mode: str, checked: bool) -> None:
        """Handle XY / Z mode toggle buttons (mutually exclusive)."""
        if not checked:
            return
        if mode == "xy":
            self._btn_mode_z.setChecked(False)
            self._projection_mode = "xy"
            self._mode_stack.setCurrentIndex(0)
        else:
            self._btn_mode_xy.setChecked(False)
            self._projection_mode = "xz" if self._btn_proj_xz.isChecked() else "yz"
            self._mode_stack.setCurrentIndex(1)
        if self.pairs:
            self._load_pair_preserve_camera(self.current_pair_idx)

    def _on_z_proj_changed(self: ManualAlignWidget, btn_id: int) -> None:
        """Switch between XZ and YZ within the Z Alignment page."""
        if self._projection_mode == "xy":
            return  # ignore if XY mode is active
        self._projection_mode = "xz" if btn_id == 0 else "yz"
        if self.pairs:
            self._load_pair_preserve_camera(self.current_pair_idx)

    def _on_z_offset_changed(self: ManualAlignWidget) -> None:
        """Handle Z-offset spinbox changes."""
        if self._suppress_z_offset_event or not self.pairs:
            return
        mid = self.pairs[self.current_pair_idx][1]
        self._current_offsets[mid] = (self.spin_fixed_z.value(), self.spin_moving_z.value())
        self.unsaved_changes.add(mid)
        self._update_z_relative_label()

        # Re-apply current state to update display (Z-offset affects XZ/YZ views)
        state = self._current_state()
        self._apply_state(state, push=False)
        self._update_status()

    def _update_z_relative_label(self: ManualAlignWidget) -> None:
        """Update the relative Z-shift label."""
        fixed_z = self.spin_fixed_z.value()
        moving_z = self.spin_moving_z.value()
        diff = fixed_z - moving_z
        self.z_relative_label.setText(f"Relative shift: {diff:+d} voxels")
