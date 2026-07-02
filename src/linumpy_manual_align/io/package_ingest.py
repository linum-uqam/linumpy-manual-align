"""Shared package ingest for server download, CLI, and cached-package paths."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from linumpy_manual_align.contracts import load_manual_align_metadata
from linumpy_manual_align.contracts.metadata import NormalizedMetadata
from linumpy_manual_align.contracts.models import SEVERITY_ERROR, SEVERITY_WARNING, ContractIssue
from linumpy_manual_align.io.transform_io import discover_aips, discover_pair_aips, discover_transforms


@dataclass(frozen=True, slots=True)
class PackageIngestResult:
    """Immutable result of ingesting a manual-align data package."""

    pkg_root: Path
    aips_dir: Path | None
    aips_xz_dir: Path | None
    aips_yz_dir: Path | None
    transforms_dir: Path | None
    slice_paths: dict[int, Path]
    pair_paths_xy: dict[tuple[int, int], dict[str, Path]]
    slice_paths_xz: dict[int, Path]
    slice_paths_yz: dict[int, Path]
    pair_paths_xz: dict[tuple[int, int], dict[str, Path]]
    pair_paths_yz: dict[tuple[int, int], dict[str, Path]]
    existing_transforms: dict[int, Path]
    metadata: NormalizedMetadata
    issues: list[ContractIssue] = field(default_factory=list)


def resolve_package_root(entry_path: Path) -> Path:
    """Normalize flexible entry paths to the metadata/discovery anchor (pkg_root)."""
    path = entry_path.resolve()
    if path.name == "aips":
        return path.parent
    if (path / "aips").is_dir():
        return path
    nested = path / "manual_align_package"
    if (nested / "aips").is_dir():
        return nested
    return path


def find_downloaded_package(output_dir: Path) -> Path | None:
    """Return the aips/ dir of an already-downloaded server package, or None."""
    candidates = [
        output_dir.parent / "server_package" / "manual_align_package" / "aips",
        output_dir.parent / "server_package" / "aips",
    ]
    for path in candidates:
        if path.exists() and any(path.glob("*.npz")):
            return path
    return None


def ingest_manual_align_package(entry_path: Path) -> PackageIngestResult:
    """Single shared ingest for server, CLI, and cached-package paths."""
    issues: list[ContractIssue] = []
    pkg_root = resolve_package_root(entry_path)

    aips_dir = pkg_root / "aips"
    if not aips_dir.is_dir() or not any(aips_dir.glob("*.npz")):
        issues.append(
            ContractIssue(
                severity=SEVERITY_ERROR,
                code="package.missing_aips",
                message=f"No resolvable aips/ directory with .npz files under {pkg_root}",
                affected_path=pkg_root,
            )
        )
        metadata, meta_issues = load_manual_align_metadata(pkg_root)
        issues.extend(meta_issues)
        return PackageIngestResult(
            pkg_root=pkg_root,
            aips_dir=None,
            aips_xz_dir=None,
            aips_yz_dir=None,
            transforms_dir=None,
            slice_paths={},
            pair_paths_xy={},
            slice_paths_xz={},
            slice_paths_yz={},
            pair_paths_xz={},
            pair_paths_yz={},
            existing_transforms={},
            metadata=metadata,
            issues=issues,
        )

    metadata, meta_issues = load_manual_align_metadata(pkg_root)
    issues.extend(meta_issues)

    transforms_dir = None
    for candidate in (pkg_root / "transforms", pkg_root.parent / "transforms"):
        if candidate.is_dir():
            transforms_dir = candidate
            break
    if transforms_dir is None:
        issues.append(
            ContractIssue(
                severity=SEVERITY_WARNING,
                code="package.missing_transforms",
                message=f"No transforms/ directory beside {pkg_root} or its parent",
                affected_path=pkg_root,
            )
        )

    def _axis(name: str) -> tuple[Path | None, dict[int, Path], dict[tuple[int, int], dict[str, Path]]]:
        axis_dir = pkg_root / name
        if not axis_dir.is_dir():
            return None, {}, {}
        return axis_dir, discover_aips(axis_dir), discover_pair_aips(axis_dir)

    aips_xz_dir, slice_paths_xz, pair_paths_xz = _axis("aips_xz")
    aips_yz_dir, slice_paths_yz, pair_paths_yz = _axis("aips_yz")

    return PackageIngestResult(
        pkg_root=pkg_root,
        aips_dir=aips_dir,
        aips_xz_dir=aips_xz_dir,
        aips_yz_dir=aips_yz_dir,
        transforms_dir=transforms_dir,
        slice_paths=discover_aips(aips_dir),
        pair_paths_xy=discover_pair_aips(aips_dir),
        slice_paths_xz=slice_paths_xz,
        slice_paths_yz=slice_paths_yz,
        pair_paths_xz=pair_paths_xz,
        pair_paths_yz=pair_paths_yz,
        existing_transforms=discover_transforms(transforms_dir) if transforms_dir else {},
        metadata=metadata,
        issues=issues,
    )
