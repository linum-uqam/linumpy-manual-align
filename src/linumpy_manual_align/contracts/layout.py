"""Package layout constants, slice naming, and manual output validation."""

from __future__ import annotations

import json
import re
from pathlib import Path

from linumpy_manual_align.contracts.models import (
    SEVERITY_ERROR,
    SEVERITY_WARNING,
    ContractIssue,
    PackageLayout,
)

METADATA_FILENAME = "manual_align_metadata.json"
MANUAL_TRANSFORMS_DIRNAME = "manual_transforms"
AIPS_DIRNAME = "aips"
AIPS_XZ_DIRNAME = "aips_xz"
AIPS_YZ_DIRNAME = "aips_yz"
TRANSFORMS_DIRNAME = "transforms"
TRANSFORM_FILENAME = "transform.tfm"
OFFSETS_FILENAME = "offsets.txt"
METRICS_FILENAME = "pairwise_registration_metrics.json"

REQUIRED_OUTPUT_FILES = (
    TRANSFORM_FILENAME,
    OFFSETS_FILENAME,
    METRICS_FILENAME,
)

_SLICE_DIR_RE = re.compile(r"^slice_z(\d+)$")


def format_manual_slice_dir(moving_id: int) -> str:
    """Return the strict manual output directory name for *moving_id*."""
    return f"slice_z{moving_id:02d}"


def parse_manual_slice_dir(name: str) -> int | None:
    """Parse *name* as a strict ``slice_z<digits>`` directory name, or return None."""
    match = _SLICE_DIR_RE.fullmatch(name)
    if match is None:
        return None
    return int(match.group(1))


def manual_output_dir(transforms_root: Path, moving_id: int) -> Path:
    """Return the manual output directory for *moving_id* under *transforms_root*."""
    return transforms_root / format_manual_slice_dir(moving_id)


def resolve_package_layout(pkg_root: Path) -> PackageLayout:
    """Resolve canonical package paths anchored under *pkg_root*."""
    aips_xz = pkg_root / AIPS_XZ_DIRNAME
    aips_yz = pkg_root / AIPS_YZ_DIRNAME
    return PackageLayout(
        root=pkg_root,
        aips_dir=pkg_root / AIPS_DIRNAME,
        aips_xz_dir=aips_xz if aips_xz.exists() else None,
        aips_yz_dir=aips_yz if aips_yz.exists() else None,
        transforms_dir=pkg_root / TRANSFORMS_DIRNAME,
        metadata_path=pkg_root / METADATA_FILENAME,
        manual_transforms_dir=pkg_root / MANUAL_TRANSFORMS_DIRNAME,
    )


def discover_manual_slice_dirs(transforms_root: Path) -> tuple[dict[int, Path], list[ContractIssue]]:
    """Discover manual output directories using strict ``slice_z`` parsing."""
    discovered: dict[int, Path] = {}
    issues: list[ContractIssue] = []
    if not transforms_root.is_dir():
        return discovered, issues
    for entry in sorted(transforms_root.iterdir()):
        if not entry.is_dir():
            continue
        moving_id = parse_manual_slice_dir(entry.name)
        if moving_id is None:
            continue
        if moving_id in discovered:
            issues.append(
                ContractIssue(
                    severity=SEVERITY_WARNING,
                    code="naming.duplicate_slice_dir",
                    message=(
                        f"Duplicate moving ID {moving_id} for {entry.name!r} "
                        f"and {discovered[moving_id].name!r}."
                    ),
                    affected_path=entry,
                )
            )
            continue
        discovered[moving_id] = entry
    return dict(sorted(discovered.items())), issues


def validate_manual_output(output_dir: Path, moving_id: int) -> list[ContractIssue]:
    """Validate *output_dir* for required manual output files and strict naming."""
    issues: list[ContractIssue] = []
    expected_name = format_manual_slice_dir(moving_id)

    if output_dir.name != expected_name or parse_manual_slice_dir(output_dir.name) != moving_id:
        issues.append(
            ContractIssue(
                severity=SEVERITY_ERROR,
                code="naming.invalid_slice_dir",
                message=(
                    f"Output directory must be named {expected_name!r}; "
                    f"found {output_dir.name!r}."
                ),
                affected_path=output_dir,
            )
        )

    missing_checks = (
        (TRANSFORM_FILENAME, "output.missing_transform"),
        (OFFSETS_FILENAME, "output.missing_offsets"),
        (METRICS_FILENAME, "output.missing_metrics"),
    )
    for filename, code in missing_checks:
        file_path = output_dir / filename
        if not file_path.is_file():
            issues.append(
                ContractIssue(
                    severity=SEVERITY_ERROR,
                    code=code,
                    message=f"Required output file {filename!r} is missing.",
                    affected_path=file_path,
                )
            )

    offsets_path = output_dir / OFFSETS_FILENAME
    if offsets_path.is_file():
        issues.extend(validate_offsets_file(offsets_path))

    metrics_path = output_dir / METRICS_FILENAME
    if metrics_path.is_file():
        issues.extend(validate_metrics_file(metrics_path))

    return issues


def validate_offsets_file(offsets_path: Path) -> list[ContractIssue]:
    """Validate offsets text contains exactly two integer tokens (fixed, then moving)."""
    text = offsets_path.read_text(encoding="utf-8")
    tokens = text.split()
    if len(tokens) != 2:
        return [
            ContractIssue(
                severity=SEVERITY_ERROR,
                code="offsets.invalid_format",
                message=(
                    "offsets.txt must contain exactly two integer values "
                    "(fixed offset, then moving offset)."
                ),
                affected_path=offsets_path,
            )
        ]
    for token in tokens:
        try:
            int(token)
        except ValueError:
            return [
                ContractIssue(
                    severity=SEVERITY_ERROR,
                    code="offsets.invalid_format",
                    message=(
                        "offsets.txt must contain exactly two integer values "
                        "(fixed offset, then moving offset)."
                    ),
                    affected_path=offsets_path,
                )
            ]
    return []


def _is_numeric_value(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _metric_numeric_value(metrics_block: object, field: str) -> object | None:
    if not isinstance(metrics_block, dict):
        return None
    entry = metrics_block.get(field)
    if not isinstance(entry, dict):
        return None
    return entry.get("value")


def validate_metrics_file(metrics_path: Path) -> list[ContractIssue]:
    """Validate pairwise registration metrics JSON for manual output contract."""
    text = metrics_path.read_text(encoding="utf-8")
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return [
            ContractIssue(
                severity=SEVERITY_ERROR,
                code="metrics.invalid_json",
                message="pairwise_registration_metrics.json is not valid JSON.",
                affected_path=metrics_path,
            )
        ]

    if not isinstance(payload, dict):
        return [
            ContractIssue(
                severity=SEVERITY_ERROR,
                code="metrics.invalid_json",
                message="pairwise_registration_metrics.json must be a JSON object.",
                affected_path=metrics_path,
            )
        ]

    issues: list[ContractIssue] = []

    if "source" not in payload:
        issues.append(
            ContractIssue(
                severity=SEVERITY_ERROR,
                code="metrics.missing_source",
                message='pairwise_registration_metrics.json must include "source".',
                affected_path=metrics_path,
            )
        )
    elif payload.get("source") != "manual":
        issues.append(
            ContractIssue(
                severity=SEVERITY_ERROR,
                code="metrics.invalid_source",
                message='pairwise_registration_metrics.json "source" must be "manual".',
                affected_path=metrics_path,
            )
        )

    if payload.get("overall_status") != "ok":
        issues.append(
            ContractIssue(
                severity=SEVERITY_ERROR,
                code="metrics.invalid_status",
                message='pairwise_registration_metrics.json "overall_status" must be "ok".',
                affected_path=metrics_path,
            )
        )

    metrics_block = payload.get("metrics")
    required_metric_fields = (
        "translation_x",
        "translation_y",
        "translation_magnitude",
        "rotation",
    )
    for field in required_metric_fields:
        value = _metric_numeric_value(metrics_block, field)
        if not _is_numeric_value(value):
            issues.append(
                ContractIssue(
                    severity=SEVERITY_ERROR,
                    code="metrics.missing_field",
                    message=(
                        f'pairwise_registration_metrics.json metrics.{field} '
                        "must include a numeric value."
                    ),
                    affected_path=metrics_path,
                    field=field,
                )
            )

    manual_alignment = payload.get("manual_alignment")
    required_manual_fields = (
        "pyramid_level",
        "working_tx",
        "working_ty",
        "center_working",
    )
    for field in required_manual_fields:
        if not isinstance(manual_alignment, dict) or field not in manual_alignment:
            issues.append(
                ContractIssue(
                    severity=SEVERITY_ERROR,
                    code="metrics.missing_field",
                    message=(
                        f"pairwise_registration_metrics.json manual_alignment.{field} "
                        "is required."
                    ),
                    affected_path=metrics_path,
                    field=field,
                )
            )
            continue
        value = manual_alignment[field]
        if field == "center_working":
            if (
                not isinstance(value, (list, tuple))
                or len(value) != 2
                or not all(_is_numeric_value(v) for v in value)
            ):
                issues.append(
                    ContractIssue(
                        severity=SEVERITY_ERROR,
                        code="metrics.missing_field",
                        message=(
                            "pairwise_registration_metrics.json manual_alignment.center_working "
                            "must be a pair of numeric values."
                        ),
                        affected_path=metrics_path,
                        field=field,
                    )
                )
        elif not _is_numeric_value(value):
            issues.append(
                ContractIssue(
                    severity=SEVERITY_ERROR,
                    code="metrics.missing_field",
                    message=(
                        f"pairwise_registration_metrics.json manual_alignment.{field} "
                        "must be numeric."
                    ),
                    affected_path=metrics_path,
                    field=field,
                )
            )

    return issues
