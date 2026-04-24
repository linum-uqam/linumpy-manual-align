"""Small UI helpers shared by :class:`~linumpy_manual_align.ui.widget.ManualAlignWidget`."""

from __future__ import annotations

import contextlib

from linumpy_manual_align.ui.widget_typing import ManualAlignWidget


class UiHelpersMixin:
    """Context managers and cross-section slider visibility."""

    @property
    def _use_precomputed_aips(self: ManualAlignWidget) -> bool:
        """True when pre-computed NPZ AIPs are available (``aips_dir`` is set)."""
        return self.aips_dir is not None

    @contextlib.contextmanager
    def _suppress_events(self: ManualAlignWidget):  # type: ignore[return]
        """Context manager that suppresses spinbox-changed signals.

        Guarantees the flag is restored even if an exception is raised inside
        the block, eliminating the fragile set/finally pattern.
        """
        self._suppress_spinbox_event = True
        try:
            yield
        finally:
            self._suppress_spinbox_event = False

    @contextlib.contextmanager
    def _suppress_z_events(self: ManualAlignWidget):  # type: ignore[return]
        """Context manager that suppresses Z-offset spinbox-changed signals."""
        self._suppress_z_offset_event = True
        try:
            yield
        finally:
            self._suppress_z_offset_event = False

    def _set_cs_sliders_visible(self: ManualAlignWidget, visible: bool) -> None:
        """Show or hide the interactive cross-section slider rows."""
        for widget in (
            self.slider_fixed_y,
            self._lbl_fixed_y,
            self._fixed_y_form_row_label,
            self.slider_cs_y,
            self._lbl_cs_y,
            self._cs_y_form_row_label,
            self.slider_fixed_x,
            self._lbl_fixed_x,
            self._fixed_x_form_row_label,
            self.slider_cs_x,
            self._lbl_cs_x,
            self._cs_x_form_row_label,
            self._cs_loading_label,
        ):
            widget.setVisible(visible)

    def _update_cs_slider_visibility(self: ManualAlignWidget) -> None:
        """Show the correct cross-section slider for the current projection mode."""
        if self._cs_mgr.slices_remote_dir is None and not self._cs_mgr.slice_remote_paths:
            return
        xz = self._projection_mode == "xz"
        yz = self._projection_mode == "yz"
        self.slider_fixed_y.setVisible(xz)
        self._lbl_fixed_y.setVisible(xz)
        self._fixed_y_form_row_label.setVisible(xz)
        self.slider_cs_y.setVisible(xz)
        self._lbl_cs_y.setVisible(xz)
        self._cs_y_form_row_label.setVisible(xz)
        self.slider_fixed_x.setVisible(yz)
        self._lbl_fixed_x.setVisible(yz)
        self._fixed_x_form_row_label.setVisible(yz)
        self.slider_cs_x.setVisible(yz)
        self._lbl_cs_x.setVisible(yz)
        self._cs_x_form_row_label.setVisible(yz)
        self._cs_loading_label.setVisible(xz or yz)
