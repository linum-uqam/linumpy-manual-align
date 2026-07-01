"""Undo/redo, automated transform load, save."""

from __future__ import annotations

import logging
from pathlib import Path

from qtpy.QtWidgets import QMessageBox

from linumpy_manual_align.contracts import (
    SEVERITY_ERROR,
    SEVERITY_WARNING,
    ContractIssue,
    manual_output_dir,
    validate_manual_output,
)
from linumpy_manual_align.io.transform_io import (
    adjust_for_rotation_center,
    load_transform,
    save_transform,
)
from linumpy_manual_align.state import AlignmentState
from linumpy_manual_align.ui.widget_typing import ManualAlignWidget

logger = logging.getLogger(__name__)


class UndoSaveMixin:
    """Mixin that implements undo/redo, per-pair save, and batch save-all operations."""

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

    def _format_save_error(self: ManualAlignWidget, mid: int, errors: list[ContractIssue], out_dir: Path) -> str:
        primary = errors[0]
        file = primary.affected_path.name if primary.affected_path is not None else out_dir.name
        return f"z{mid:02d}: {file} - {primary.code}: {primary.message}"

    def _save_and_validate_pair(
        self: ManualAlignWidget, mid: int, state: AlignmentState
    ) -> list[ContractIssue]:
        cx, cy = self.pair_centers.get(mid, (0.0, 0.0))
        offsets = self._current_offsets.get(mid, (0, 0))

        out_dir = manual_output_dir(self.output_dir, mid)
        save_transform(
            out_dir,
            state.tx,
            state.ty,
            state.rotation,
            center=(cx, cy),
            level=self.level,
            offsets=offsets,
        )

        issues = validate_manual_output(out_dir, mid)
        errors = [i for i in issues if i.severity == SEVERITY_ERROR]
        warnings = [i for i in issues if i.severity == SEVERITY_WARNING]

        if errors:
            logger.warning("Save validation failed for z%02d: %s", mid, issues)
            self.saved_pairs.discard(mid)
            message = self._format_save_error(mid, errors, out_dir)
            self.viewer.status = message
            self.status_label.setText(message)
            QMessageBox.critical(self, "Save validation failed", message)
            return errors

        self.saved_pairs.add(mid)
        self.unsaved_changes.discard(mid)
        base = f"Saved transform for z{mid:02d} -> {out_dir}"
        if warnings:
            warning_text = "; ".join(f"{w.code}: {w.message}" for w in warnings)
            status = f"{base} (warning {warning_text})"
        else:
            status = base
        self.viewer.status = status
        self.status_label.setText(status)
        self._flash_saved(mid)
        return []

    def _save_current(self: ManualAlignWidget) -> None:
        """Save the current transform for the current pair."""
        if not self.pairs:
            return
        _fid, mid = self.pairs[self.current_pair_idx]
        state = self._current_state()
        self._save_and_validate_pair(mid, state)

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

        saved = 0
        for _fid, mid in self.pairs:
            if mid not in self.unsaved_changes:
                continue
            stack = self.undo_stacks.get(mid)
            if stack is None:
                continue
            state = stack.current
            errors = self._save_and_validate_pair(mid, state)
            if errors:
                detail = self.viewer.status
                self.viewer.status = f"Saved {saved} pair(s), 1 failed validation - {detail}"
                self.status_label.setText(self.viewer.status)
                return

            saved += 1

        remaining = [m for m in self.unsaved_changes if self.undo_stacks.get(m) is not None]
        if remaining:
            self.viewer.status = f"Cannot exit: {len(remaining)} pair(s) still have unsaved changes."
            self.status_label.setText(self.viewer.status)
            return

        self._close_confirmed = True
        self.viewer.status = f"Saved {saved} transforms to {self.output_dir}"
        logger.info("Saved %d manual transforms to %s", saved, self.output_dir)
        self.viewer.close()
