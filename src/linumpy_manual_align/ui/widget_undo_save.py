"""Undo/redo, automated transform load, save."""

from __future__ import annotations

import logging

from qtpy.QtWidgets import QMessageBox

from linumpy_manual_align.io.transform_io import (
    adjust_for_rotation_center,
    load_transform,
    save_transform,
)
from linumpy_manual_align.state import AlignmentState
from linumpy_manual_align.ui.widget_typing import ManualAlignWidget

logger = logging.getLogger(__name__)


class UndoSaveMixin:
    def _undo(self: ManualAlignWidget) -> None:
        if not self.pairs:
            return
        mid = self.pairs[self.current_pair_idx][1]
        stack = self.undo_stacks.get(mid)
        if stack:
            state = stack.undo()
            if state:
                self._apply_state(state, push=False)
                self._update_status()

    def _redo(self: ManualAlignWidget) -> None:
        if not self.pairs:
            return
        mid = self.pairs[self.current_pair_idx][1]
        stack = self.undo_stacks.get(mid)
        if stack:
            state = stack.redo()
            if state:
                self._apply_state(state, push=False)
                self._update_status()

    # ----- Transform actions -----

    def _load_automated_transform(self: ManualAlignWidget) -> None:
        """Load the existing automated transform as starting point."""
        if not self.pairs:
            return
        mid = self.pairs[self.current_pair_idx][1]
        if mid not in self.existing_transforms:
            self.viewer.status = f"No automated transform for z{mid:02d}"
            return

        tfm_dir = self.existing_transforms[mid]
        tfm_files = list(tfm_dir.glob("*.tfm"))
        if not tfm_files:
            self.viewer.status = f"No .tfm file in {tfm_dir}"
            return

        # load_transform returns (tx, ty) already in widget content-shift
        # convention (matching AlignmentState and napari layer.translate).
        tx, ty, rot, tfm_center = load_transform(tfm_files[0])
        scale = 2**self.level
        img_center = self.pair_centers.get(mid)
        if img_center is not None:
            tx, ty = adjust_for_rotation_center(tx, ty, rot, tfm_center, (img_center[0] * scale, img_center[1] * scale))
        state = AlignmentState(tx=tx / scale, ty=ty / scale, rotation=rot)
        self._apply_state(state, push=True)
        self.viewer.status = f"Loaded automated transform for z{mid:02d}: tx={state.tx:.1f} ty={state.ty:.1f} rot={rot:.2f}°"
        self._update_status()

    def _reset_transform(self: ManualAlignWidget) -> None:
        state = AlignmentState()
        self._apply_state(state, push=True)
        self._update_status()

    # ----- Save -----

    def _save_current(self: ManualAlignWidget) -> None:
        """Save the current transform for the current pair."""
        if not self.pairs:
            return
        _fid, mid = self.pairs[self.current_pair_idx]
        state = self._current_state()

        cx, cy = self.pair_centers.get(mid, (0.0, 0.0))
        offsets = self._current_offsets.get(mid, (0, 0))

        out_dir = self.output_dir / f"slice_z{mid:02d}"
        save_transform(out_dir, state.tx, state.ty, state.rotation, center=(cx, cy), level=self.level, offsets=offsets)
        self.saved_pairs.add(mid)
        self.unsaved_changes.discard(mid)
        self.viewer.status = f"Saved transform for z{mid:02d} → {out_dir}"
        self._flash_saved(mid)

    def _save_all_and_exit(self: ManualAlignWidget, skip_confirm: bool = False) -> None:
        """Save all modified pairs and close."""
        unsaved = [mid for _fid, mid in self.pairs if mid in self.unsaved_changes and self.undo_stacks.get(mid) is not None]
        if not skip_confirm:
            total_saved = len(self.saved_pairs)
            msg = QMessageBox(self)
            msg.setWindowTitle("Save All & Exit")
            msg.setIcon(QMessageBox.Question)
            if unsaved:
                msg.setText(
                    f"Save {len(unsaved)} modified pair(s) and exit?\n\n"
                    f"{total_saved} pair(s) were already saved in this session."
                )
            else:
                msg.setText(f"No unsaved changes. Exit?\n\n{total_saved} pair(s) were saved in this session.")
            msg.setStandardButtons(QMessageBox.Ok | QMessageBox.Cancel)
            msg.setDefaultButton(QMessageBox.Ok)
            msg.button(QMessageBox.Ok).setText("Save && Exit" if unsaved else "Exit")
            if msg.exec() != QMessageBox.Ok:
                return

        count = 0
        for _fid, mid in self.pairs:
            if mid not in self.unsaved_changes:
                continue
            stack = self.undo_stacks.get(mid)
            if stack is None:
                continue
            state = stack.current

            cx, cy = self.pair_centers.get(mid, (0.0, 0.0))
            offsets = self._current_offsets.get(mid, (0, 0))

            out_dir = self.output_dir / f"slice_z{mid:02d}"
            save_transform(out_dir, state.tx, state.ty, state.rotation, center=(cx, cy), level=self.level, offsets=offsets)
            self.saved_pairs.add(mid)
            count += 1

        self.unsaved_changes.clear()
        self._close_confirmed = True
        self.viewer.status = f"Saved {count} transforms to {self.output_dir}"
        logger.info(f"Saved {count} manual transforms to {self.output_dir}")
        self.viewer.close()
