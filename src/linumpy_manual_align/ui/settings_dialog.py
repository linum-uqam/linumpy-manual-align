"""Settings dialog for linumpy-manual-align.

Opens as a *modeless* :class:`QDialog` (non-blocking) so the user can
adjust parameters while continuing to work in the main widget.

All edits are staged in the dialog's widgets until the user clicks
**Apply** or **OK** — then they are written to the :data:`settings`
singleton in one pass so that ``changed`` signals fire per key and
live-update any connected consumers (keybindings, spinbox steps, …).

**Cancel** discards staged edits and closes the dialog.
**Reset All** reverts every key to its coded default.
"""

from __future__ import annotations

import sys

from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSizePolicy,
    QSpinBox,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from linumpy_manual_align.settings import DEFAULTS, settings

_IS_MACOS = sys.platform == "darwin"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _reset_btn(key: str, widget: QSpinBox | QDoubleSpinBox | QLineEdit) -> QPushButton:
    """Return a small Reset button that restores *widget* to the default for *key*."""
    btn = QPushButton("↺")
    btn.setFixedWidth(24)
    btn.setToolTip(f"Reset to default: {DEFAULTS[key]}")
    btn.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)

    default = DEFAULTS[key]
    if isinstance(widget, QLineEdit):
        btn.clicked.connect(lambda: widget.setText(str(default)))
    elif isinstance(widget, QDoubleSpinBox):
        btn.clicked.connect(lambda: widget.setValue(float(default)))
    else:
        btn.clicked.connect(lambda: widget.setValue(int(default)))
    return btn


def _row(
    label: str,
    key: str,
    widget: QSpinBox | QDoubleSpinBox | QLineEdit,
) -> tuple[QLabel, QWidget]:
    """Return ``(label_widget, field_with_reset_button)`` for a form row."""
    lbl = QLabel(label)
    lbl.setToolTip(f"Default: {DEFAULTS[key]}")

    container = QWidget()
    h = QHBoxLayout()
    h.setContentsMargins(0, 0, 0, 0)
    h.setSpacing(4)
    h.addWidget(widget, stretch=1)
    h.addWidget(_reset_btn(key, widget))
    container.setLayout(h)
    return lbl, container


def _int_spin(key: str, min_val: int = 0, max_val: int = 9999) -> QSpinBox:
    s = QSpinBox()
    s.setRange(min_val, max_val)
    s.setValue(int(settings.get(key)))
    return s


def _float_spin(key: str, min_val: float = 0.0, max_val: float = 9999.0, decimals: int = 2) -> QDoubleSpinBox:
    s = QDoubleSpinBox()
    s.setRange(min_val, max_val)
    s.setDecimals(decimals)
    s.setValue(float(settings.get(key)))
    return s


def _line_edit(key: str) -> QLineEdit:
    le = QLineEdit(str(settings.get(key)))
    return le


# ---------------------------------------------------------------------------
# Tab builders
# ---------------------------------------------------------------------------


def _build_shortcuts_tab(widgets: dict) -> QWidget:
    """Build the Shortcuts tab — translation and rotation step sizes."""
    tab = QWidget()
    layout = QVBoxLayout()
    layout.setContentsMargins(8, 8, 8, 8)
    layout.setSpacing(8)
    tab.setLayout(layout)

    # Translation group
    trans_group = QGroupBox("Translation step sizes (pixels)")
    form = QFormLayout()
    form.setLabelAlignment(Qt.AlignRight | Qt.AlignVCenter)
    form.setSpacing(4)
    trans_group.setLayout(form)

    for key, label in [
        ("shortcuts/translate_fine_px", "Fine (Arrow):"),
        ("shortcuts/translate_coarse_px", "Coarse (Alt+Arrow):"),
        ("shortcuts/translate_large_px", "Large (Shift+Arrow):"),
    ]:
        spin = _int_spin(key, 1, 500)
        widgets[key] = spin
        form.addRow(*_row(label, key, spin))
    layout.addWidget(trans_group)

    # Rotation group
    rot_group = QGroupBox("Rotation step sizes (degrees)")
    form2 = QFormLayout()
    form2.setLabelAlignment(Qt.AlignRight | Qt.AlignVCenter)
    form2.setSpacing(4)
    rot_group.setLayout(form2)

    for key, label in [
        ("shortcuts/rotate_fine_deg", "Fine ([/]):"),
        ("shortcuts/rotate_coarse_deg", "Coarse (Alt+[/]):"),
        ("shortcuts/rotate_large_deg", "Large (Ctrl+[/]):"),
    ]:
        spin = _float_spin(key, 0.01, 45.0, decimals=2)
        widgets[key] = spin
        form2.addRow(*_row(label, key, spin))
    layout.addWidget(rot_group)

    # Cross-section nudge group (keyboard Alt+,/. + Ctrl+,/. + moving slider single/page step)
    cs_group = QGroupBox("Cross-section slider nudge")
    cs_group.setToolTip(
        "Pixel steps for moving-slice Y (XZ) / X (YZ): coarse Alt+, / Alt-., fine Ctrl+, / Ctrl-., "
        "slider arrows, and prefetch spacing along the cross-section axis (Cross-section tab)."
    )
    form3 = QFormLayout()
    form3.setLabelAlignment(Qt.AlignRight | Qt.AlignVCenter)
    form3.setSpacing(4)
    cs_group.setLayout(form3)

    for key, label in [
        ("shortcuts/cs_nudge_px", "Coarse (Alt+,/.):"),
        ("shortcuts/cs_nudge_fine_px", "Fine (Ctrl+,/.):"),
    ]:
        spin = _int_spin(key, 1, 500)
        widgets[key] = spin
        form3.addRow(*_row(label, key, spin))
    layout.addWidget(cs_group)

    layout.addStretch()
    return tab


def _build_cross_section_tab(widgets: dict) -> QWidget:
    """Build the Cross-section prefetch / cache tab."""
    tab = QWidget()
    layout = QVBoxLayout()
    layout.setContentsMargins(8, 8, 8, 8)
    layout.setSpacing(8)
    tab.setLayout(layout)

    group = QGroupBox("Prefetch / cache")
    form = QFormLayout()
    form.setLabelAlignment(Qt.AlignRight | Qt.AlignVCenter)
    form.setSpacing(4)
    group.setLayout(form)

    for key, label in [
        ("prefetch/steps", "Prefetch steps (each side):"),
        ("prefetch/evict_radius_multiplier", "Evict radius multiplier:"),
    ]:
        spin = _int_spin(key, 1, 200)
        widgets[key] = spin
        form.addRow(*_row(label, key, spin))

    note = QLabel(
        "<i style='color:grey;'>Spacing between prefetched positions uses the same pixel step as "
        "<b>Cross-section nudge</b> on the Shortcuts tab.<br>"
        "Evict radius = multiplier x that step. "
        "Cache entries further than this from the current position are dropped.</i>"
    )
    note.setWordWrap(True)
    layout.addWidget(group)
    layout.addWidget(note)
    layout.addStretch()
    return tab


def _build_spinboxes_tab(widgets: dict) -> QWidget:
    """Build the Spinboxes tab — single-step increments for the transform controls."""
    tab = QWidget()
    layout = QVBoxLayout()
    layout.setContentsMargins(8, 8, 8, 8)
    layout.setSpacing(8)
    tab.setLayout(layout)

    group = QGroupBox("Spinbox single-step increments")
    form = QFormLayout()
    form.setLabelAlignment(Qt.AlignRight | Qt.AlignVCenter)
    form.setSpacing(4)
    group.setLayout(form)

    for key, label in [
        ("spin/tx_ty_step", "TX / TY (px):"),
        ("spin/rot_step", "Rotation (°):"),
    ]:
        spin = _float_spin(key, 0.01, 100.0, decimals=2)
        widgets[key] = spin
        form.addRow(*_row(label, key, spin))

    key = "spin/tile_step"
    spin = _int_spin(key, 1, 256)
    widgets[key] = spin
    form.addRow(*_row("Checkerboard tile step:", key, spin))

    layout.addWidget(group)
    layout.addStretch()
    return tab


def _build_server_tab(widgets: dict) -> QWidget:
    """Build the Server defaults tab."""
    tab = QWidget()
    layout = QVBoxLayout()
    layout.setContentsMargins(8, 8, 8, 8)
    layout.setSpacing(8)
    tab.setLayout(layout)

    group = QGroupBox("Server defaults")
    form = QFormLayout()
    form.setLabelAlignment(Qt.AlignRight | Qt.AlignVCenter)
    form.setSpacing(4)
    group.setLayout(form)

    for key, label in [
        ("server/default_host", "Default host:"),
        ("server/remote_python", "Remote Python path:"),
    ]:
        le = _line_edit(key)
        widgets[key] = le
        form.addRow(*_row(label, key, le))

    note = QLabel(
        "<i style='color:grey;'>These are used as fallback defaults when no server config is loaded.<br>"
        "<b>Remote Python</b> is required for SSH cross-sections: it must be the interpreter "
        "where linumpy, zarr, ome-zarr, etc. are installed (typically your linumpy repo "
        "<code>.venv/bin/python</code> on the server). System <code>python3</code> will not work. "
        "Alternatively set <code>LINUMPY_MANUAL_ALIGN_REMOTE_PYTHON</code> (overrides this field).</i>"
    )
    note.setWordWrap(True)
    layout.addWidget(group)
    layout.addWidget(note)
    layout.addStretch()
    return tab


# ---------------------------------------------------------------------------
# Dialog
# ---------------------------------------------------------------------------


class SettingsDialog(QDialog):
    """Modeless settings dialog for linumpy-manual-align.

    Changes are written to the :data:`settings` singleton only on
    **Apply** or **OK**.  Closing with **Cancel** discards staged edits.
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent, Qt.Window)
        self.setWindowTitle("Manual Align — Settings")
        self.setMinimumWidth(420)
        self.setAttribute(Qt.WA_DeleteOnClose, False)  # keep for reuse via show()

        self._widgets: dict[str, QSpinBox | QDoubleSpinBox | QLineEdit] = {}

        layout = QVBoxLayout()
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)
        self.setLayout(layout)

        # Scrollable tabs for small screens
        self._tabs = QTabWidget()
        self._tabs.addTab(_build_shortcuts_tab(self._widgets), "Shortcuts")
        self._tabs.addTab(_build_cross_section_tab(self._widgets), "Cross-section")
        self._tabs.addTab(_build_spinboxes_tab(self._widgets), "Spinboxes")
        self._tabs.addTab(_build_server_tab(self._widgets), "Server")
        layout.addWidget(self._tabs)

        # Standard OK / Apply / Cancel button box + Reset All
        btn_box = QDialogButtonBox()
        btn_reset_all = QPushButton("Reset All Defaults")
        btn_reset_all.clicked.connect(self._reset_all)
        btn_box.addButton(btn_reset_all, QDialogButtonBox.ResetRole)
        btn_box.addButton(QDialogButtonBox.Ok)
        btn_box.addButton(QDialogButtonBox.Apply)
        btn_box.addButton(QDialogButtonBox.Cancel)

        # Wire each button explicitly. Relying on QDialogButtonBox.accepted/rejected is
        # brittle with a modeless dialog and mixed roles (OK vs Apply vs Cancel).
        ok_btn = btn_box.button(QDialogButtonBox.Ok)
        apply_btn = btn_box.button(QDialogButtonBox.Apply)
        cancel_btn = btn_box.button(QDialogButtonBox.Cancel)
        ok_btn.clicked.connect(self._apply)
        ok_btn.clicked.connect(self.hide)
        apply_btn.clicked.connect(self._apply)
        cancel_btn.clicked.connect(self._revert)
        cancel_btn.clicked.connect(self.hide)

        layout.addWidget(btn_box)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _apply(self) -> None:
        """Write all staged widget values to the settings singleton."""
        for key, widget in self._widgets.items():
            if isinstance(widget, QLineEdit):
                settings.set(key, widget.text())
            elif isinstance(widget, QDoubleSpinBox):
                settings.set(key, widget.value())
            else:
                settings.set(key, widget.value())

    def _revert(self) -> None:
        """Reset all widgets back to the currently stored (or default) values."""
        for key, widget in self._widgets.items():
            val = settings.get(key)
            if isinstance(widget, QLineEdit):
                widget.setText(str(val))
            elif isinstance(widget, QDoubleSpinBox):
                widget.setValue(float(val))
            else:
                widget.setValue(int(val))

    def _reset_all(self) -> None:
        """Restore all defaults in settings and update dialog widgets."""
        settings.reset_all()
        self._revert()

    # ------------------------------------------------------------------
    # Override show() to sync widgets with current stored values each time
    # the dialog is opened (in case settings changed externally).
    # ------------------------------------------------------------------

    def showEvent(self, event):  # type: ignore[override]
        self._revert()
        super().showEvent(event)
