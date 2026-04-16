"""Stable entry points for embedding the manual-align UI in a napari viewer."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from linumpy_manual_align.ui.widget import ManualAlignWidget

if TYPE_CHECKING:
    import napari


def create_manual_align_widget(
    viewer: napari.Viewer,
    *,
    input_dir: Path | None = None,
    transforms_dir: Path | None = None,
    output_dir: Path,
    level: int = 1,
    filter_slices: list[int] | None = None,
    aips_dir: Path | None = None,
    aips_xz_dir: Path | None = None,
    aips_yz_dir: Path | None = None,
    server_config: object = None,
) -> ManualAlignWidget:
    """Build a :class:`~linumpy_manual_align.ui.widget.ManualAlignWidget` docked in *viewer*.

    Parameters match :class:`~linumpy_manual_align.ui.widget.ManualAlignWidget` — see that class for semantics.
    """
    w = ManualAlignWidget(
        viewer=viewer,
        input_dir=input_dir,
        transforms_dir=transforms_dir,
        output_dir=output_dir,
        level=level,
        filter_slices=filter_slices,
        aips_dir=aips_dir,
        aips_xz_dir=aips_xz_dir,
        aips_yz_dir=aips_yz_dir,
        server_config=server_config,
    )
    viewer.window.add_dock_widget(w, name="Manual Align", area="right")
    return w
