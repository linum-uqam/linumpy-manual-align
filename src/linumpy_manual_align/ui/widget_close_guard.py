"""Intercept main-window close when there are unsaved edits."""

from __future__ import annotations

from qtpy.QtWidgets import QMessageBox

from linumpy_manual_align.ui.widget_typing import ManualAlignWidget


class CloseGuardMixin:
    def _install_close_guard(self: ManualAlignWidget) -> None:
        """Install event filter to warn about unsaved changes on window close."""
        try:
            main_window = self.viewer.window._qt_window
            main_window.installEventFilter(self)
        except AttributeError:
            pass  # napari API changed; skip gracefully

    def eventFilter(self: ManualAlignWidget, obj: object, event: object) -> bool:
        """Intercept close events to warn about unsaved changes."""
        from qtpy.QtCore import QEvent as _QEvent

        if isinstance(event, _QEvent) and event.type() == _QEvent.Close and not self._confirm_close():
            event.ignore()
            return True
        return super().eventFilter(obj, event)

    def _confirm_close(self: ManualAlignWidget) -> bool:
        """Return True if it's OK to close (no unsaved changes or user confirmed)."""
        if self._close_confirmed:
            return True
        if not self.unsaved_changes:
            return True
        n = len(self.unsaved_changes)
        msg = QMessageBox(self)
        msg.setWindowTitle("Unsaved Changes")
        msg.setText(f"{n} modified pair(s) have unsaved changes.")
        msg.setInformativeText("Save all before closing?")
        msg.setIcon(QMessageBox.Warning)
        save_btn = msg.addButton("Save All && Close", QMessageBox.AcceptRole)
        msg.addButton("Discard && Close", QMessageBox.DestructiveRole)
        cancel_btn = msg.addButton("Cancel", QMessageBox.RejectRole)
        msg.setDefaultButton(save_btn)
        msg.exec_()
        clicked = msg.clickedButton()
        if clicked is cancel_btn:
            return False
        if clicked is save_btn:
            self._save_all_and_exit(skip_confirm=True)
            return False  # _save_all_and_exit handles close itself
        return True  # Discard
