"""Headless workflow contract layer for manual-align package and output layout."""

from __future__ import annotations

from linumpy_manual_align.contracts.layout import (
    AIPS_DIRNAME,
    AIPS_XZ_DIRNAME,
    AIPS_YZ_DIRNAME,
    MANUAL_TRANSFORMS_DIRNAME,
    METADATA_FILENAME,
    METRICS_FILENAME,
    OFFSETS_FILENAME,
    REQUIRED_OUTPUT_FILES,
    TRANSFORM_FILENAME,
    TRANSFORMS_DIRNAME,
    discover_manual_slice_dirs,
    format_manual_slice_dir,
    manual_output_dir,
    parse_manual_slice_dir,
    resolve_package_layout,
    validate_manual_output,
    validate_metrics_file,
    validate_offsets_file,
)
from linumpy_manual_align.contracts.metadata import (
    NormalizedMetadata,
    load_manual_align_metadata,
    metadata_candidates,
    reconcile_slice_annotations,
)
from linumpy_manual_align.contracts.models import (
    SEVERITY_ERROR,
    SEVERITY_INFO,
    SEVERITY_WARNING,
    ContractIssue,
    PackageLayout,
)

__all__ = [
    "AIPS_DIRNAME",
    "AIPS_XZ_DIRNAME",
    "AIPS_YZ_DIRNAME",
    "ContractIssue",
    "MANUAL_TRANSFORMS_DIRNAME",
    "METADATA_FILENAME",
    "METRICS_FILENAME",
    "NormalizedMetadata",
    "OFFSETS_FILENAME",
    "PackageLayout",
    "REQUIRED_OUTPUT_FILES",
    "SEVERITY_ERROR",
    "SEVERITY_INFO",
    "SEVERITY_WARNING",
    "TRANSFORM_FILENAME",
    "TRANSFORMS_DIRNAME",
    "discover_manual_slice_dirs",
    "format_manual_slice_dir",
    "load_manual_align_metadata",
    "manual_output_dir",
    "metadata_candidates",
    "parse_manual_slice_dir",
    "reconcile_slice_annotations",
    "resolve_package_layout",
    "validate_manual_output",
    "validate_metrics_file",
    "validate_offsets_file",
]
