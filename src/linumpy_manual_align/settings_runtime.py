"""Apply :mod:`linumpy_manual_align.settings` to the live UI and napari bindings.

Centralises every ``settings.get`` / keybinding / hint string that used to be
spread across ``widget.py`` and ``ui_builder.py`` so there is a single place
to read shortcut defaults and push them into widgets.

The :class:`~linumpy_manual_align.settings.AppSettings` singleton and
:class:`~linumpy_manual_align.ui.settings_dialog.SettingsDialog` stay in their own
modules; this file is the *runtime* adaptation layer only.
"""

from __future__ import annotations

from typing import Any

from qtpy.QtWidgets import QLineEdit, QSlider

from linumpy_manual_align.remote import ServerConfig
from linumpy_manual_align.settings import settings


def keybindings_from_settings() -> list[tuple[str, str, tuple]]:
    """Build the napari ``(key, method_name, args)`` list from current settings."""
    fine = int(settings.get("shortcuts/translate_fine_px"))
    coarse = int(settings.get("shortcuts/translate_coarse_px"))
    large = int(settings.get("shortcuts/translate_large_px"))
    rot_fine = float(settings.get("shortcuts/rotate_fine_deg"))
    rot_coarse = float(settings.get("shortcuts/rotate_coarse_deg"))
    rot_large = float(settings.get("shortcuts/rotate_large_deg"))
    cs_nudge = int(settings.get("shortcuts/cs_nudge_px"))

    return [
        ("n", "_next_pair", ()),
        ("p", "_prev_pair", ()),
        ("s", "_save_current", ()),
        ("Right", "_nudge_translate", (fine, 0)),
        ("Left", "_nudge_translate", (-fine, 0)),
        ("Up", "_nudge_translate", (0, -fine)),
        ("Down", "_nudge_translate", (0, fine)),
        ("Alt-Right", "_nudge_translate", (coarse, 0)),
        ("Alt-Left", "_nudge_translate", (-coarse, 0)),
        ("Alt-Up", "_nudge_translate", (0, -coarse)),
        ("Alt-Down", "_nudge_translate", (0, coarse)),
        ("Shift-Right", "_nudge_translate", (large, 0)),
        ("Shift-Left", "_nudge_translate", (-large, 0)),
        ("Shift-Up", "_nudge_translate", (0, -large)),
        ("Shift-Down", "_nudge_translate", (0, large)),
        ("]", "_nudge_rotate", (rot_fine,)),
        ("[", "_nudge_rotate", (-rot_fine,)),
        ("Alt-]", "_nudge_rotate", (rot_coarse,)),
        ("Alt-[", "_nudge_rotate", (-rot_coarse,)),
        ("Control-]", "_nudge_rotate", (rot_large,)),
        ("Control-[", "_nudge_rotate", (-rot_large,)),
        ("v", "_toggle_z_proj", ()),
        ("m", "_toggle_alignment_mode", ()),
        ("Alt-,", "_nudge_cs_position", (-cs_nudge,)),
        ("Alt-.", "_nudge_cs_position", (cs_nudge,)),
    ]


def shortcut_hints_footer_html() -> str:
    """HTML for the dock footer shortcut reference (XY / Z / Nav)."""
    fine = int(settings.get("shortcuts/translate_fine_px"))
    coarse = int(settings.get("shortcuts/translate_coarse_px"))
    large = int(settings.get("shortcuts/translate_large_px"))
    rot_f = float(settings.get("shortcuts/rotate_fine_deg"))
    rot_c = float(settings.get("shortcuts/rotate_coarse_deg"))
    rot_l = float(settings.get("shortcuts/rotate_large_deg"))
    cs_n = int(settings.get("shortcuts/cs_nudge_px"))
    hint_xy = (
        f"←→↑↓: {fine}px &nbsp; Alt+←→↑↓: {coarse}px &nbsp; Shift+←→↑↓: {large}px"
        f"<br>[/]: {rot_f}° &nbsp; Alt+[/]: {rot_c}° &nbsp; Ctrl+[/]: {rot_l}°"
    )
    hint_z = (
        f"↑↓: {fine} vox &nbsp; Alt+↑↓: {coarse} vox (Z depth)"
        f"<br>←→: cross-section &nbsp; Alt+,/.: ±{cs_n} cross-section<br>V: toggle XZ/YZ"
    )
    hint_nav = "N/P: next/prev &nbsp; M: toggle XY/Z &nbsp; S: save<br>Ctrl+Z: undo &nbsp; Ctrl+Shift+Z: redo"
    return f"<small><i style='color: grey;'><b>XY:</b> {hint_xy}<br><b>Z:</b> {hint_z}<br><b>Nav:</b> {hint_nav}</i></small>"


def xy_page_keyboard_hint_html() -> str:
    """Small italic hint under TX/TY/rotation on the XY alignment page."""
    tf = int(settings.get("shortcuts/translate_fine_px"))
    tc = int(settings.get("shortcuts/translate_coarse_px"))
    tl = int(settings.get("shortcuts/translate_large_px"))
    rf = float(settings.get("shortcuts/rotate_fine_deg"))
    rc = float(settings.get("shortcuts/rotate_coarse_deg"))
    rl = float(settings.get("shortcuts/rotate_large_deg"))
    return (
        f"<i style='color: grey;'>Arrow: {tf}px · Alt: {tc}px · Ctrl: {tl}px · [/]: {rf}° · Alt[/]: {rc}° · Ctrl[/]: {rl}°</i>"
    )


def spin_step_tx_ty() -> float:
    return float(settings.get("spin/tx_ty_step"))


def spin_step_rot() -> float:
    return float(settings.get("spin/rot_step"))


def spin_step_tile() -> int:
    return int(settings.get("spin/tile_step"))


def cross_section_nudge_px() -> int:
    """Pixel step for moving cross-section sliders, Alt+,/., and prefetch spacing."""
    return int(settings.get("shortcuts/cs_nudge_px"))


def default_host_display() -> str:
    """Initial text for the dock Host field."""
    return str(settings.get("server/default_host"))


def apply_cross_section_slider_steps(slider_cs_y: QSlider, slider_cs_x: QSlider) -> None:
    """Set single/page step and tooltips from ``shortcuts/cs_nudge_px``."""
    step = cross_section_nudge_px()
    slider_cs_y.setSingleStep(step)
    slider_cs_y.setPageStep(step)
    slider_cs_x.setSingleStep(step)
    slider_cs_x.setPageStep(step)
    slider_cs_y.setToolTip(
        f"Moving slice Y position — slide to find matching tissue  [Alt-, / Alt-.: ±{step} px; slider arrows use same step]"
    )
    slider_cs_x.setToolTip(
        f"Moving slice X position — slide to find matching tissue  [Alt-, / Alt-.: ±{step} px; slider arrows use same step]"
    )


def sync_server_config_host_from_ui(server_config: object | None, host_edit: QLineEdit) -> None:
    """If CLI left ``ServerConfig.host`` empty, copy the dock host (from QSettings)."""
    if server_config is None or not isinstance(server_config, ServerConfig):
        return
    if str(server_config.host).strip():
        return
    h = host_edit.text().strip()
    if h:
        server_config.host = h


def persist_dock_host_if_changed(host_edit: QLineEdit) -> None:
    """Write the dock Host field to QSettings when it differs from the stored value."""
    text = host_edit.text().strip()
    if text == str(settings.get("server/default_host")):
        return
    settings.set("server/default_host", text)


def apply_settings_changed(widget: Any, key: str) -> None:
    """Dispatch ``settings.changed`` to keybindings, hints, spinboxes, and host field.

    *widget* is the :class:`~linumpy_manual_align.ui.widget.ManualAlignWidget` instance
    (typed as :class:`typing.Any` to avoid a circular import).
    """
    if key.startswith("shortcuts/"):
        widget._install_keybindings()
        widget.hints_label.setText(shortcut_hints_footer_html())
        if key == "shortcuts/cs_nudge_px":
            apply_cross_section_slider_steps(widget.slider_cs_y, widget.slider_cs_x)
    elif key == "spin/tx_ty_step":
        step = float(settings.get(key))
        widget.spin_tx.setSingleStep(step)
        widget.spin_ty.setSingleStep(step)
    elif key == "spin/rot_step":
        widget.spin_rot.setSingleStep(float(settings.get(key)))
    elif key == "spin/tile_step":
        widget.spin_tile.setSingleStep(int(settings.get(key)))
    elif key == "server/default_host":
        new_host = str(settings.get(key))
        widget.host_edit.blockSignals(True)
        widget.host_edit.setText(new_host)
        widget.host_edit.blockSignals(False)
        cfg = widget.server_config
        if cfg is not None and isinstance(cfg, ServerConfig):
            cfg.host = new_host.strip()
