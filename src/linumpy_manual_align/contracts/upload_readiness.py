"""Headless upload readiness classification for manual transform outputs."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from linumpy_manual_align.contracts.layout import (
    format_manual_slice_dir,
    manual_output_dir,
    parse_manual_slice_dir,
    validate_manual_output,
)
from linumpy_manual_align.contracts.models import (
    SEVERITY_ERROR,
    SEVERITY_WARNING,
    ContractIssue,
)

UPLOAD_NOT_SAVED = "upload.not_saved"
UPLOAD_NO_OUTPUT_DIR = "upload.no_output_dir"

_SLICE_PREFIX = "slice_z"


class PairUploadStatus(StrEnum):
    """Upload readiness classification for a single session pair."""

    READY = "ready"
    MISSING = "missing"
    INVALID = "invalid"


@dataclass(frozen=True, slots=True)
class PairUploadReadiness:
    """Readiness verdict for one ``(fixed_id, moving_id)`` session pair."""

    moving_id: int
    fixed_id: int
    status: PairUploadStatus
    output_dir: Path | None
    issues: tuple[ContractIssue, ...]


@dataclass(frozen=True, slots=True)
class UploadReadinessReport:
    """Aggregated upload readiness across all assessed session pairs."""

    pairs: tuple[PairUploadReadiness, ...]
    ready_count: int
    missing_count: int
    invalid_count: int
    warning_count: int
    ready_dirs: tuple[Path, ...]

    @property
    def has_blocking_errors(self) -> bool:
        """Return True when any pair is MISSING or INVALID."""
        return self.missing_count + self.invalid_count > 0

    def summary_line(self) -> str:
        """Render the one-line operator summary of aggregate counts."""
        return (
            f"{self.ready_count} ready, {self.missing_count} missing, "
            f"{self.invalid_count} invalid, {self.warning_count} warning"
        )

    def error_lines(self) -> list[str]:
        """Return formatted ERROR lines for MISSING and INVALID pairs."""
        lines: list[str] = []
        for pair in sorted(self.pairs, key=lambda item: item.moving_id):
            if pair.status not in (PairUploadStatus.MISSING, PairUploadStatus.INVALID):
                continue
            lines.extend(
                format_upload_issue_line(pair.moving_id, issue, pair.output_dir)
                for issue in pair.issues
                if issue.severity == SEVERITY_ERROR
            )
        return lines

    def warning_lines(self) -> list[str]:
        """Return formatted WARNING lines across all assessed pairs."""
        lines: list[str] = []
        for pair in sorted(self.pairs, key=lambda item: item.moving_id):
            lines.extend(
                format_upload_issue_line(pair.moving_id, issue, pair.output_dir)
                for issue in pair.issues
                if issue.severity == SEVERITY_WARNING
            )
        return lines


def _resolve_session_output_dir(transforms_root: Path, moving_id: int) -> Path | None:
    """Resolve the on-disk output directory for *moving_id* under *transforms_root*."""
    canonical = manual_output_dir(transforms_root, moving_id)
    if canonical.is_dir():
        return canonical

    if not transforms_root.is_dir():
        return None

    expected_name = format_manual_slice_dir(moving_id)
    loose_match: Path | None = None
    for entry in transforms_root.iterdir():
        if not entry.is_dir():
            continue
        name = entry.name
        parsed = parse_manual_slice_dir(name)
        if parsed == moving_id and name != expected_name:
            loose_match = entry
            continue
        if parsed is not None:
            continue
        if not name.startswith(_SLICE_PREFIX):
            continue
        suffix = name[len(_SLICE_PREFIX) :]
        if suffix.isdigit() and int(suffix) == moving_id:
            loose_match = entry

    return loose_match


def assess_upload_readiness(
    pairs: Sequence[tuple[int, int]],
    transforms_root: Path,
    saved_pairs: set[int] | frozenset[int],
) -> UploadReadinessReport:
    """Classify upload readiness for *pairs* under *transforms_root*."""
    pair_results: list[PairUploadReadiness] = []
    ready_count = 0
    missing_count = 0
    invalid_count = 0
    warning_count = 0

    for fixed_id, moving_id in pairs:
        if moving_id not in saved_pairs:
            issue = ContractIssue(
                severity=SEVERITY_ERROR,
                code=UPLOAD_NOT_SAVED,
                message="Pair not saved in this session",
                affected_path=None,
            )
            pair_results.append(
                PairUploadReadiness(
                    moving_id=moving_id,
                    fixed_id=fixed_id,
                    status=PairUploadStatus.MISSING,
                    output_dir=None,
                    issues=(issue,),
                )
            )
            missing_count += 1
            continue

        resolved = _resolve_session_output_dir(transforms_root, moving_id)
        if resolved is None:
            issue = ContractIssue(
                severity=SEVERITY_ERROR,
                code=UPLOAD_NO_OUTPUT_DIR,
                message="No output directory on disk",
                affected_path=None,
            )
            pair_results.append(
                PairUploadReadiness(
                    moving_id=moving_id,
                    fixed_id=fixed_id,
                    status=PairUploadStatus.MISSING,
                    output_dir=None,
                    issues=(issue,),
                )
            )
            missing_count += 1
            continue

        issues = validate_manual_output(resolved, moving_id)
        errors = [issue for issue in issues if issue.severity == SEVERITY_ERROR]
        warnings = [issue for issue in issues if issue.severity == SEVERITY_WARNING]
        warning_count += len(warnings)

        if errors:
            pair_results.append(
                PairUploadReadiness(
                    moving_id=moving_id,
                    fixed_id=fixed_id,
                    status=PairUploadStatus.INVALID,
                    output_dir=resolved,
                    issues=tuple(issues),
                )
            )
            invalid_count += 1
        else:
            pair_results.append(
                PairUploadReadiness(
                    moving_id=moving_id,
                    fixed_id=fixed_id,
                    status=PairUploadStatus.READY,
                    output_dir=resolved,
                    issues=tuple(warnings),
                )
            )
            ready_count += 1

    ready_dirs = tuple(
        pair.output_dir
        for pair in sorted(
            (item for item in pair_results if item.status == PairUploadStatus.READY),
            key=lambda item: item.moving_id,
        )
        if pair.output_dir is not None
    )

    return UploadReadinessReport(
        pairs=tuple(pair_results),
        ready_count=ready_count,
        missing_count=missing_count,
        invalid_count=invalid_count,
        warning_count=warning_count,
        ready_dirs=ready_dirs,
    )


def format_upload_issue_line(
    moving_id: int,
    issue: ContractIssue,
    output_dir: Path | None,
) -> str:
    """Format one operator-facing upload issue line for *moving_id*."""
    if issue.affected_path is not None:
        name = issue.affected_path.name
    elif output_dir is not None:
        name = output_dir.name
    else:
        name = "(missing)"
    return f"z{moving_id:02d}: {name} - {issue.code}: {issue.message}"
