#!/usr/bin/env python
r"""Interactive manual slice alignment tool using napari.

Displays consecutive common-space slices as red/green AIP (average intensity
projection) overlays.  The user adjusts translation and rotation of the moving
slice until it aligns with the fixed slice (yellow = aligned).

Saves corrected transforms as SimpleITK .tfm files that are drop-in compatible
with the linumpy stacking pipeline (linum_stack_slices_motor.py).

Usage
-----
    linumpy-manual-align \
        --data_package /path/to/manual_align_package/ \
        --server_config ~/Downloads/sub-22/nextflow.config

    # Or directly from OME-Zarr volumes (requires the ome-zarr extra):
    linumpy-manual-align \
        --input_dir /path/to/bring_to_common_space/ \
        --transforms_dir /path/to/register_pairwise/ \
        --level 1

After saving, use the Upload button (if --server_config provided) or manually
copy the transforms to the server and re-run the pipeline from the ``stack``
step with ``-resume``.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from linumpy_manual_align.contracts import MANUAL_TRANSFORMS_DIRNAME, load_manual_align_metadata
from linumpy_manual_align.contracts.models import SEVERITY_WARNING


logger = logging.getLogger(__name__)


def resolve_output_dir(
    data_package: Path | None,
    input_dir: Path | None,
    server_config: Path | None,
) -> Path:
    """Resolve the manual-transforms output directory using contract layout constants."""
    if data_package is not None:
        pkg = Path(data_package)
        if pkg.parent.name == "server_package":
            return pkg.parent.parent / MANUAL_TRANSFORMS_DIRNAME
        return pkg / MANUAL_TRANSFORMS_DIRNAME
    if input_dir is not None:
        return Path(input_dir).parent / MANUAL_TRANSFORMS_DIRNAME
    return Path(server_config).parent / MANUAL_TRANSFORMS_DIRNAME


def resolve_package_level(data_package: Path | None) -> int | None:
    """Return the normalized package level when metadata specifies one, else None."""
    if data_package is None:
        return None
    normalized, issues = load_manual_align_metadata(Path(data_package))
    for issue in issues:
        if issue.severity == SEVERITY_WARNING:
            logger.warning("%s: %s", issue.code, issue.message)
    if normalized.source_path is None:
        return None
    if not normalized.pyramid_level_explicit:
        return None
    return normalized.pyramid_level


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments and return the parsed namespace."""
    p = argparse.ArgumentParser(
        description="Interactive manual slice alignment (napari).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--input_dir",
        type=Path,
        default=None,
        help="Directory with common-space slices (slice_z##.ome.zarr).\nNot needed when --data_package is used.",
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
        default=None,
        help=(
            "Pyramid level to use (0=full, 1=2x downsample, ...). "
            "When omitted, the level recorded in the data package metadata is "
            "used, otherwise it defaults to 1."
        ),
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
    p.add_argument(
        "--server_config",
        type=Path,
        default=None,
        help="Path to a local nextflow.config (e.g. ~/Downloads/sub-22/nextflow.config).\n"
        "Enables download/upload buttons in the UI for server interaction.",
    )
    p.add_argument(
        "--debug",
        action="store_true",
        default=False,
        help="Enable DEBUG logging for linumpy_manual_align (useful for diagnosing slider issues).",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """Entry point: parse arguments, configure logging, launch the napari viewer."""
    args = parse_args(argv)

    if args.debug:
        logging.basicConfig(
            level=logging.DEBUG,
            format="%(asctime)s %(name)s %(levelname)s %(message)s",
        )
        logging.getLogger("linumpy_manual_align").setLevel(logging.DEBUG)

    # Resolve data package paths
    aips_dir = None
    aips_xz_dir = None
    aips_yz_dir = None
    if args.data_package is not None:
        pkg = Path(args.data_package)
        aips_dir = pkg / "aips"
        if not aips_dir.exists():
            raise FileNotFoundError(f"AIPs directory not found in data package: {aips_dir}")
        # Discover axis-specific AIP directories
        if (pkg / "aips_xz").exists():
            aips_xz_dir = pkg / "aips_xz"
        if (pkg / "aips_yz").exists():
            aips_yz_dir = pkg / "aips_yz"
        # Use package transforms unless explicitly overridden
        if args.transforms_dir is None:
            pkg_tfm = pkg / "transforms"
            if pkg_tfm.exists():
                args.transforms_dir = pkg_tfm
        # Read level from package metadata if not explicitly set
        if args.level is None:
            pkg_level = resolve_package_level(pkg)
            if pkg_level is not None:
                args.level = pkg_level
    elif args.input_dir is None and args.server_config is None:
        from qtpy.QtWidgets import QApplication, QFileDialog, QMessageBox

        _app = QApplication.instance() or QApplication([])
        choice, _ = QFileDialog.getOpenFileName(
            None,
            "Open data package or server config",
            str(Path.home()),
            "Supported files (*.json *.config);;All files (*)",
        )
        if not choice:
            QMessageBox.critical(None, "No input", "Either --input_dir, --data_package, or --server_config is required.")
            return
        chosen = Path(choice)
        if chosen.suffix == ".config":
            args.server_config = chosen
        else:
            # Assume it's inside a data package — go up to the package root
            args.data_package = chosen.parent if chosen.name != chosen.parent.name else chosen
            pkg = Path(args.data_package)
            aips_dir = pkg / "aips"
            if not aips_dir.exists():
                raise FileNotFoundError(f"AIPs directory not found in data package: {aips_dir}")
            if (pkg / "aips_xz").exists():
                aips_xz_dir = pkg / "aips_xz"
            if (pkg / "aips_yz").exists():
                aips_yz_dir = pkg / "aips_yz"
            if args.transforms_dir is None:
                pkg_tfm = pkg / "transforms"
                if pkg_tfm.exists():
                    args.transforms_dir = pkg_tfm
            if args.level is None:
                pkg_level = resolve_package_level(pkg)
                if pkg_level is not None:
                    args.level = pkg_level

    if args.output_dir is None:
        args.output_dir = resolve_output_dir(args.data_package, args.input_dir, args.server_config)

    # Fix Qt settings scope before any code imports :data:`linumpy_manual_align.settings.settings`
    # (the widget pulls it in via :mod:`linumpy_manual_align.api`). Without this, macOS
    # can store QSettings in an inconsistent location relative to the Settings dialog.
    from qtpy.QtCore import QCoreApplication

    QCoreApplication.setOrganizationName("linum-uqam")
    QCoreApplication.setApplicationName("linumpy-manual-align")

    # Import napari late — startup takes a moment
    import napari
    from qtpy.QtWidgets import QApplication

    from linumpy_manual_align.api import create_manual_align_widget
    from linumpy_manual_align.remote import parse_server_config
    from linumpy_manual_align.ui.napari_menus import add_manual_align_settings_action

    # QSettings reads are unreliable on macOS before a QApplication exists.
    _app = QApplication.instance() or QApplication([])

    # Parse server config if provided
    server_config = None
    if args.server_config is not None:
        from linumpy_manual_align.settings import settings

        # Host is not in nextflow.config; use the same value as the dock (QSettings).
        server_config = parse_server_config(
            args.server_config,
            host=str(settings.get("server/default_host")).strip(),
            remote_base=str(settings.get("server/remote_workspace_base")),
        )

    # Fall back to the historical default when neither the user nor the package
    # metadata specified a pyramid level.
    if args.level is None:
        args.level = 1

    viewer = napari.Viewer(title="Manual Slice Alignment")

    widget = create_manual_align_widget(
        viewer,
        input_dir=args.input_dir,
        transforms_dir=args.transforms_dir,
        output_dir=args.output_dir,
        level=args.level,
        filter_slices=args.slices,
        aips_dir=aips_dir,
        aips_xz_dir=aips_xz_dir,
        aips_yz_dir=aips_yz_dir,
        server_config=server_config,
    )

    add_manual_align_settings_action(viewer, widget._open_settings_dialog)

    napari.run()


if __name__ == "__main__":
    main()
