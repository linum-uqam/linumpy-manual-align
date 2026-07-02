"""Overlay compositing and alignment state application."""

from __future__ import annotations

import numpy as np

from linumpy_manual_align.io.image_utils import (
    OVERLAY_COLOR,
    apply_transform,
)
from linumpy_manual_align.state import AlignmentState
from linumpy_manual_align.ui import napari_layers
from linumpy_manual_align.ui.widget_typing import ManualAlignWidget


class OverlayStateMixin:
    """Mixin that builds and refreshes the composite image overlay (color, diff, checkerboard)."""

    def _make_shifted_moving(self: ManualAlignWidget, state: AlignmentState) -> np.ndarray:
        """Return the moving AIP with rotation + pixel-level shift baked in (for composite modes)."""
        moving = self._original_moving_aip
        if moving is None:
            return np.zeros((1, 1), dtype=np.float32)
        return apply_transform(moving, rotation=state.rotation, tx=state.tx, ty=state.ty)

    def _rebuild_layer_visibility(self: ManualAlignWidget) -> None:
        scale = list(self.fixed_layer.scale) if self.fixed_layer is not None else [1.0, 1.0]
        self._composite_layer = napari_layers.set_overlay_mode(
            self.viewer,
            overlay_mode=self._overlay_mode,
            fixed_layer=self.fixed_layer,
            moving_layer=self.moving_layer,
            composite_layer=self._composite_layer,
            base_image=self._original_fixed_aip,
            scale=scale,
            projection_mode=self._projection_mode,
        )

    def _refresh_composite(self: ManualAlignWidget, state: AlignmentState) -> None:
        """Recompute and push composite image data for non-color overlay modes."""
        if self._overlay_mode == OVERLAY_COLOR or self._composite_layer is None:
            return
        if self._original_fixed_aip is None or self._original_moving_aip is None:
            return
        shifted = self._make_shifted_moving(state)
        napari_layers.refresh_composite(
            self.viewer,
            overlay_mode=self._overlay_mode,
            composite_layer=self._composite_layer,
            fixed_base=self._original_fixed_aip,
            shifted_moving=shifted,
            tile_size=self.spin_tile.value(),
        )

    # ----- State application -----

    def _apply_state(self: ManualAlignWidget, state: AlignmentState, push: bool = True) -> None:
        """Apply an alignment state to the moving layer."""
        if self.moving_layer is None or self._original_moving_aip is None or not self.pairs:
            return

        mid = self.pairs[self.current_pair_idx][1]

        if push:
            self.undo_stacks[mid].push(state)
            self.unsaved_changes.add(mid)
            if hasattr(self, "uploaded_pairs"):
                self.uploaded_pairs.discard(mid)
            if hasattr(self, "_refresh_session_state"):
                self._refresh_session_state()

        sy, sx = self._moving_scale_yx[0], self._moving_scale_yx[1]

        if self._projection_mode == "xy":
            # Bake ONLY rotation into pixel data (about the image centre, tx=ty=0).
            # Translation is applied via layer.translate so napari can size the canvas
            # correctly — baking a large shift would push content into one corner and
            # leave most of the frame as empty black padding.
            # Pure-translation layer.translate is reliable (no QR decomposition issue).
            baked = apply_transform(
                self._original_moving_aip,
                rotation=state.rotation,
                tx=0,
                ty=0,
            )
            self.moving_layer.data = baked
            self.moving_layer.rotate = 0.0
            self.moving_layer.translate = [state.ty * sy, state.tx * sx]
        else:
            # XZ/YZ mode: pure translation only (no rotation).
            # The export script flips Z ([::-1]) so row 0 = deepest voxel.
            # A higher Z-index therefore maps to a LOWER row index, so the
            # correct shift to bring the overlap regions into alignment is
            # (moving_z - fixed_z) / scale, not (fixed_z - moving_z).
            offsets = self._current_offsets.get(mid, (0, 0))
            dz_display = (offsets[1] - offsets[0]) / 2**self.level
            horiz = state.tx if self._projection_mode == "xz" else state.ty
            self.moving_layer.rotate = 0.0
            self.moving_layer.translate = [dz_display * sy, horiz * sx]

        # Update composite overlay if active
        if self._overlay_mode != OVERLAY_COLOR and self._projection_mode == "xy":
            self._refresh_composite(state)

        # Sync spinboxes
        with self._suppress_events():
            self.spin_tx.setValue(state.tx)
            self.spin_ty.setValue(state.ty)
            self.spin_rot.setValue(state.rotation)
            self.rot_slider.setValue(int(state.rotation * 10))

    def _current_state(self: ManualAlignWidget) -> AlignmentState:
        return AlignmentState(
            tx=self.spin_tx.value(),
            ty=self.spin_ty.value(),
            rotation=self.spin_rot.value(),
        )
