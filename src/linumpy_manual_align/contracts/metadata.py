"""Tolerant normalization for manual_align_metadata.json workflow metadata."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from linumpy_manual_align.contracts.layout import METADATA_FILENAME
from linumpy_manual_align.contracts.models import (
    SEVERITY_WARNING,
    ContractIssue,
)


@dataclass(frozen=True, slots=True)
class NormalizedMetadata:
    """Typed, normalized view of manual_align_metadata.json known workflow fields."""

    pyramid_level: int = 0
    pyramid_level_explicit: bool = False
    slices_remote_dir: str | None = None
    slice_filenames: dict[int, str] = field(default_factory=dict)
    slice_remote_paths: dict[int, str] = field(default_factory=dict)
    interpolated_slice_ids: frozenset[int] = frozenset()
    package_root: Path | None = None
    source_path: Path | None = None


def metadata_candidates(pkg_root: Path) -> tuple[Path, Path]:
    """Return root-then-parent metadata paths with no recursive search."""
    return (
        pkg_root / METADATA_FILENAME,
        pkg_root.parent / METADATA_FILENAME,
    )


def _default_metadata(pkg_root: Path, source_path: Path | None = None) -> NormalizedMetadata:
    return NormalizedMetadata(package_root=pkg_root, source_path=source_path)


def _parse_level(raw: object, field_name: str, issues: list[ContractIssue]) -> int:
    if raw is None:
        return 0
    try:
        return int(raw)
    except (TypeError, ValueError):
        issues.append(
            ContractIssue(
                severity=SEVERITY_WARNING,
                code="metadata.invalid_field",
                message=f"Metadata field {field_name!r} must be an integer; using default 0.",
                field=field_name,
            )
        )
        return 0


def _parse_optional_str(raw: object, field_name: str, issues: list[ContractIssue]) -> str | None:
    if raw is None:
        return None
    if isinstance(raw, str):
        return raw
    issues.append(
        ContractIssue(
            severity=SEVERITY_WARNING,
            code="metadata.invalid_field",
            message=f"Metadata field {field_name!r} must be a string; ignoring value.",
            field=field_name,
        )
    )
    return None


def _parse_int_str_map(raw: object, field_name: str, issues: list[ContractIssue]) -> dict[int, str]:
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        issues.append(
            ContractIssue(
                severity=SEVERITY_WARNING,
                code="metadata.invalid_field",
                message=f"Metadata field {field_name!r} must be an object; using empty mapping.",
                field=field_name,
            )
        )
        return {}
    result: dict[int, str] = {}
    for key, value in raw.items():
        try:
            int_key = int(key)
        except (TypeError, ValueError):
            issues.append(
                ContractIssue(
                    severity=SEVERITY_WARNING,
                    code="metadata.invalid_field",
                    message=f"Metadata field {field_name!r} has non-integer key {key!r}; skipping entry.",
                    field=field_name,
                )
            )
            continue
        if not isinstance(value, str):
            issues.append(
                ContractIssue(
                    severity=SEVERITY_WARNING,
                    code="metadata.invalid_field",
                    message=f"Metadata field {field_name!r}[{int_key!r}] must be a string; skipping entry.",
                    field=field_name,
                )
            )
            continue
        result[int_key] = value
    return result


def _parse_int_set(raw: object, field_name: str, issues: list[ContractIssue]) -> frozenset[int]:
    if raw is None:
        return frozenset()
    if not isinstance(raw, list):
        issues.append(
            ContractIssue(
                severity=SEVERITY_WARNING,
                code="metadata.invalid_field",
                message=f"Metadata field {field_name!r} must be a list; using empty set.",
                field=field_name,
            )
        )
        return frozenset()
    result: set[int] = set()
    for item in raw:
        try:
            result.add(int(item))
        except (TypeError, ValueError):
            issues.append(
                ContractIssue(
                    severity=SEVERITY_WARNING,
                    code="metadata.invalid_field",
                    message=f"Metadata field {field_name!r} contains non-integer value {item!r}; skipping entry.",
                    field=field_name,
                )
            )
    return frozenset(result)


def load_manual_align_metadata(pkg_root: Path) -> tuple[NormalizedMetadata, list[ContractIssue]]:
    """Load and normalize metadata from root-then-parent candidates."""
    issues: list[ContractIssue] = []
    root_candidate, parent_candidate = metadata_candidates(pkg_root)

    source_path: Path | None = None
    for candidate in (root_candidate, parent_candidate):
        if candidate.exists():
            source_path = candidate
            break

    if source_path is None:
        issues.append(
            ContractIssue(
                severity=SEVERITY_WARNING,
                code="metadata.missing",
                message="No manual_align_metadata.json found at package root or parent directory.",
                affected_path=root_candidate,
            )
        )
        return _default_metadata(pkg_root), issues

    try:
        raw = json.loads(source_path.read_text())
    except json.JSONDecodeError:
        issues.append(
            ContractIssue(
                severity=SEVERITY_WARNING,
                code="metadata.invalid_json",
                message="manual_align_metadata.json is not valid JSON; using defaults.",
                affected_path=source_path,
            )
        )
        return _default_metadata(pkg_root), issues

    if not isinstance(raw, dict):
        issues.append(
            ContractIssue(
                severity=SEVERITY_WARNING,
                code="metadata.invalid_json",
                message="manual_align_metadata.json must be a JSON object; using defaults.",
                affected_path=source_path,
            )
        )
        return _default_metadata(pkg_root), issues

    has_cs = "cross_section_level" in raw
    has_pl = "pyramid_level" in raw
    if has_cs and has_pl:
        cs = raw["cross_section_level"]
        pl = raw["pyramid_level"]
        try:
            if cs is not None and pl is not None and int(cs) != int(pl):
                issues.append(
                    ContractIssue(
                        severity=SEVERITY_WARNING,
                        code="metadata.conflicting_level",
                        message=(
                            f"cross_section_level ({cs!r}) and pyramid_level ({pl!r}) differ; "
                            "using cross_section_level."
                        ),
                        field="cross_section_level",
                    )
                )
        except (TypeError, ValueError):
            pass
    level_raw = raw.get("cross_section_level", raw.get("pyramid_level", 0))
    level_field = "cross_section_level" if has_cs else "pyramid_level"
    has_level = has_cs or has_pl

    normalized = NormalizedMetadata(
        pyramid_level=_parse_level(level_raw, level_field, issues),
        pyramid_level_explicit=has_level,
        slices_remote_dir=_parse_optional_str(raw.get("slices_remote_dir"), "slices_remote_dir", issues),
        slice_filenames=_parse_int_str_map(raw.get("slice_filenames"), "slice_filenames", issues),
        slice_remote_paths=_parse_int_str_map(raw.get("slice_remote_paths"), "slice_remote_paths", issues),
        interpolated_slice_ids=_parse_int_set(raw.get("interpolated_slice_ids"), "interpolated_slice_ids", issues),
        package_root=pkg_root,
        source_path=source_path,
    )
    return normalized, issues


def reconcile_slice_annotations(
    metadata: NormalizedMetadata,
    discovered: dict[int, Path],
) -> list[ContractIssue]:
    """Compare metadata slice annotations against discovered package slice files."""
    issues: list[ContractIssue] = []
    for moving_id in sorted(metadata.slice_filenames):
        annotated_name = metadata.slice_filenames[moving_id]
        discovered_path = discovered.get(moving_id)
        if discovered_path is None:
            issues.append(
                ContractIssue(
                    severity=SEVERITY_WARNING,
                    code="metadata.slice_disagreement",
                    message=(
                        f"Metadata annotates slice {moving_id} as {annotated_name!r}, "
                        f"but no matching slice file was discovered in the package."
                    ),
                    field="slice_filenames",
                )
            )
            continue
        if annotated_name != discovered_path.name:
            issues.append(
                ContractIssue(
                    severity=SEVERITY_WARNING,
                    code="metadata.slice_disagreement",
                    message=(
                        f"Metadata slice_filenames[{moving_id}] is {annotated_name!r}, "
                        f"but discovered file is {discovered_path.name!r}."
                    ),
                    field="slice_filenames",
                    affected_path=discovered_path,
                )
            )
    return issues
