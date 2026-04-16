"""Linumpy Manual Align — CLI app that embeds a napari viewer for slice alignment correction."""

from __future__ import annotations

from linumpy_manual_align.api import create_manual_align_widget

__all__ = ["__version__", "create_manual_align_widget"]
__version__ = "0.1.0"
