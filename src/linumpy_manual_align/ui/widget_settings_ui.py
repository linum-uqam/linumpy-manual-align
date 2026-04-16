"""Settings dialog, hints, and cross-section slider steps."""

from __future__ import annotations

from linumpy_manual_align.settings_runtime import (
    apply_cross_section_slider_steps,
    apply_settings_changed,
    shortcut_hints_footer_html,
)
from linumpy_manual_align.ui.widget_typing import ManualAlignWidget


class SettingsUiMixin:
    def _refresh_shortcut_hints(self: ManualAlignWidget) -> None:
        """Update the dock footer hint text from current :data:`settings` values."""
        self.hints_label.setText(shortcut_hints_footer_html())

    def _apply_cs_slider_step(self: ManualAlignWidget) -> None:
        """Match moving cross-section slider arrow/Page steps to ``shortcuts/cs_nudge_px``."""
        apply_cross_section_slider_steps(self.slider_cs_y, self.slider_cs_x)

    # ----- Settings dialog -----

    def _open_settings_dialog(self: ManualAlignWidget) -> None:
        """Open (or raise) the modeless Settings dialog."""
        from linumpy_manual_align.ui.settings_dialog import SettingsDialog

        if self._settings_dialog is None:
            self._settings_dialog = SettingsDialog(self)
        self._settings_dialog.show()
        self._settings_dialog.raise_()
        self._settings_dialog.activateWindow()

    def _on_settings_changed(self: ManualAlignWidget, key: str, _value: object) -> None:
        """React live to settings changes emitted by the :data:`settings` singleton.

        - Any ``shortcuts/*`` change reinstalls keybindings.
        - ``spin/tx_ty_step`` updates the TX/TY spinbox steps.
        - ``spin/rot_step`` updates the rotation spinbox step.
        - ``spin/tile_step`` updates the checkerboard tile spinbox step.
        - ``server/default_host`` syncs the dock Host field from Settings.
        """
        apply_settings_changed(self, key)
