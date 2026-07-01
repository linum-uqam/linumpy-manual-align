"""Frozen value models for workflow contract validation issues and layout."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

SEVERITY_ERROR = "error"
SEVERITY_WARNING = "warning"
SEVERITY_INFO = "info"


@dataclass(frozen=True, slots=True)
class ContractIssue:
    """Structured validation issue with severity, code, message, and optional context."""

    severity: str
    code: str
    message: str
    affected_path: Path | None = None
    field: str | None = None


@dataclass(frozen=True, slots=True)
class PackageLayout:
    """Resolved filesystem layout for a manual-align data package."""

    root: Path
    aips_dir: Path
    aips_xz_dir: Path | None
    aips_yz_dir: Path | None
    transforms_dir: Path
    metadata_path: Path
    manual_transforms_dir: Path
