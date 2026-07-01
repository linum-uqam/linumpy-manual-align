"""Linumpy Manual Align — CLI app that embeds a napari viewer for slice alignment correction."""

from __future__ import annotations

from typing import Any

__all__ = ["__version__", "create_manual_align_widget"]
__version__ = "0.1.0"


def __getattr__(name: str) -> Any:
    if name == "create_manual_align_widget":
        from linumpy_manual_align.api import create_manual_align_widget

        return create_manual_align_widget
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
