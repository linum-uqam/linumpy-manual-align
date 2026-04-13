#!/usr/bin/env python
"""Interactive manual slice alignment tool using napari.

Displays consecutive common-space slices as red/green AIP (average intensity
projection) overlays.  The user adjusts translation and rotation of the moving
slice until it aligns with the fixed slice (yellow = aligned).

Saves corrected transforms as SimpleITK .tfm files that are drop-in compatible
with the linumpy stacking pipeline (linum_stack_slices_motor.py).

Usage
-----
    .venv/bin/python tools/manual-align/manual_align.py \\
        --input_dir /path/to/bring_to_common_space/ \\
        --transforms_dir /path/to/register_pairwise/ \\
        --output_dir /path/to/manual_transforms/ \\
        --level 1

After saving, copy the contents of ``output_dir`` into the server's
``register_pairwise/`` directory and re-run the pipeline from the ``stack``
step with ``-resume``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure the tool directory is on sys.path so local modules are importable
_tool_dir = Path(__file__).resolve().parent
if str(_tool_dir) not in sys.path:
    sys.path.insert(0, str(_tool_dir))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Interactive manual slice alignment (napari).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--input_dir",
        type=Path,
        default=None,
        help="Directory with common-space slices (slice_z##.ome.zarr).\n"
        "Not needed when --data_package is used.",
    )
    p.add_argument(
        "--transforms_dir",
        type=Path,
        default=None,
        help="Directory with automated pairwise transforms (register_pairwise/).",
    )
    p.add_argument(
        "--output_dir",
        type=Path,
        default=None,
        help="Directory to save manual transforms. Default: input_dir/../manual_transforms/",
    )
    p.add_argument(
        "--level",
        type=int,
        default=1,
        help="Pyramid level to use (0=full, 1=2x downsample, ...). Default: 1",
    )
    p.add_argument(
        "--slices",
        type=int,
        nargs="*",
        default=None,
        help="Only show pairs involving these moving slice IDs. Default: all.",
    )
    p.add_argument(
        "--data_package",
        type=Path,
        default=None,
        help="Path to a data package exported by linum_export_manual_align.py.\n"
        "When used, --input_dir and --transforms_dir are read from the package.",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    # Resolve data package paths
    aips_dir = None
    if args.data_package is not None:
        pkg = Path(args.data_package)
        aips_dir = pkg / "aips"
        if not aips_dir.exists():
            raise FileNotFoundError(f"AIPs directory not found in data package: {aips_dir}")
        # Use package transforms unless explicitly overridden
        if args.transforms_dir is None:
            pkg_tfm = pkg / "transforms"
            if pkg_tfm.exists():
                args.transforms_dir = pkg_tfm
        # Read level from package metadata if not explicitly set
        metadata_path = pkg / "manual_align_metadata.json"
        if metadata_path.exists():
            import json

            metadata = json.loads(metadata_path.read_text())
            if args.level == 1 and "pyramid_level" in metadata:
                args.level = metadata["pyramid_level"]
    elif args.input_dir is None:
        raise ValueError("Either --input_dir or --data_package is required.")

    if args.output_dir is None:
        if args.data_package is not None:
            args.output_dir = Path(args.data_package) / "manual_transforms"
        else:
            args.output_dir = args.input_dir.parent / "manual_transforms"

    # Import napari late — startup takes a moment
    import napari
    from widget import ManualAlignWidget

    viewer = napari.Viewer(title="Manual Slice Alignment")

    widget = ManualAlignWidget(
        viewer=viewer,
        input_dir=args.input_dir,
        transforms_dir=args.transforms_dir,
        output_dir=args.output_dir,
        level=args.level,
        filter_slices=args.slices,
        aips_dir=aips_dir,
    )
    viewer.window.add_dock_widget(widget, name="Manual Align", area="right")

    napari.run()


if __name__ == "__main__":
    main()
