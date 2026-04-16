"""Attach actions to the napari :class:`~napari.Window` using public APIs only."""

from __future__ import annotations

from collections.abc import Callable
from typing import Protocol

from qtpy.QtGui import QAction


class _PluginsMenu(Protocol):
    def addSeparator(self) -> None: ...
    def addAction(self, action: QAction) -> None: ...


class _NapariWindow(Protocol):
    """Stable ``plugins_menu`` on :class:`napari.Window` (no ``_qt_window``)."""

    plugins_menu: _PluginsMenu


class _NapariViewer(Protocol):
    window: _NapariWindow


def add_manual_align_settings_action(viewer: _NapariViewer, on_trigger: Callable[[], None]) -> None:
    """Add *Manual Align Settings…* under the window's Plugins menu.

    Uses :attr:`napari.Window.plugins_menu` and ``QAction(parent=plugins_menu)`` —
    no private ``_qt_window`` access.
    """
    win = viewer.window
    plugins_menu = win.plugins_menu
    action = QAction("Manual Align Settings\u2026", plugins_menu)
    action.triggered.connect(on_trigger)
    plugins_menu.addSeparator()
    plugins_menu.addAction(action)
