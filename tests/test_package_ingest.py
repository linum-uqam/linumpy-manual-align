"""EXTR-01: unit and parity tests for io/package_ingest."""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from linumpy_manual_align.contracts.models import SEVERITY_ERROR
from linumpy_manual_align.io.package_ingest import (
    find_downloaded_package,
    ingest_manual_align_package,
    resolve_package_root,
)


def _issue_tuples(issues) -> list[tuple]:
    return [(i.severity, i.code, i.message) for i in issues]


def _ingest_signature(result, *, root: Path) -> dict:
    """Normalized ingest signature for cross-layout parity (relative paths only)."""

    def _rel(path: Path | None) -> str | None:
        if path is None:
            return None
        return str(path.relative_to(root))

    return {
        "aips": _rel(result.aips_dir),
        "transforms": _rel(result.transforms_dir),
        "aips_xz": _rel(result.aips_xz_dir),
        "aips_yz": _rel(result.aips_yz_dir),
        "slice_ids": sorted(result.slice_paths),
        "transform_ids": sorted(result.existing_transforms),
        "metadata_level": result.metadata.pyramid_level,
        "metadata_level_explicit": result.metadata.pyramid_level_explicit,
        "issues": _issue_tuples(result.issues),
    }


def test_ingest_canonical_golden(copy_fixture_tree) -> None:
    """Golden manual_align_package fixture ingests with expected discovery maps."""
    pkg_root = copy_fixture_tree("manual_align_package")
    result = ingest_manual_align_package(pkg_root)

    assert result.aips_dir is not None
    assert sorted(result.slice_paths) == [0, 1, 2]
    assert sorted(result.existing_transforms) == [1, 2]
    assert result.metadata.pyramid_level == 1
    assert result.metadata.pyramid_level_explicit is True
    assert not any(issue.severity == SEVERITY_ERROR for issue in result.issues)


def test_ingest_missing_aips(tmp_path: Path) -> None:
    """Missing aips/ returns error issue and empty maps instead of raising."""
    empty = tmp_path / "empty_pkg"
    empty.mkdir()
    result = ingest_manual_align_package(empty)

    assert result.aips_dir is None
    assert result.slice_paths == {}
    assert result.pair_paths_xy == {}
    assert result.existing_transforms == {}
    error_issues = [i for i in result.issues if i.severity == SEVERITY_ERROR]
    assert len(error_issues) == 1
    assert "aips" in error_issues[0].message.lower()


def test_resolve_package_root(copy_fixture_tree) -> None:
    """Flexible entry paths resolve to the directory containing aips/."""
    pkg_root = copy_fixture_tree("manual_align_package")
    aips_dir = pkg_root / "aips"
    nested = pkg_root.parent / "wrapper" / "manual_align_package"
    shutil.copytree(pkg_root, nested)

    assert resolve_package_root(pkg_root) == pkg_root
    assert resolve_package_root(aips_dir) == pkg_root
    assert resolve_package_root(nested) == nested


def test_find_downloaded_package(copy_fixture_tree, tmp_path: Path) -> None:
    """find_downloaded_package locates server_package layouts with .npz files."""
    manual_transforms = tmp_path / "manual_transforms"
    manual_transforms.mkdir()
    server_pkg = tmp_path / "server_package" / "manual_align_package"
    copy_fixture_tree("manual_align_package", server_pkg)

    found = find_downloaded_package(manual_transforms)
    assert found is not None
    assert found.name == "aips"
    assert any(found.glob("*.npz"))

    isolated = tmp_path / "isolated_workspace" / "manual_transforms"
    isolated.parent.mkdir()
    isolated.mkdir()
    assert find_downloaded_package(isolated) is None


def test_three_path_ingest_parity(copy_fixture_tree, tmp_path: Path) -> None:
    """Canonical, server_package wrapper, and flat-aips layouts share ingest signatures."""
    canonical = copy_fixture_tree("manual_align_package", tmp_path / "canonical")

    server_wrapped = tmp_path / "server_wrapped" / "server_package" / "manual_align_package"
    shutil.copytree(canonical, server_wrapped)

    flat_root = tmp_path / "flat" / "server_package"
    shutil.copytree(canonical / "aips", flat_root / "aips")
    shutil.copytree(canonical / "transforms", flat_root / "transforms")
    shutil.copy2(canonical / "manual_align_metadata.json", flat_root / "manual_align_metadata.json")

    server_package_root = server_wrapped.parent

    sig_canonical = _ingest_signature(
        ingest_manual_align_package(canonical), root=ingest_manual_align_package(canonical).pkg_root
    )
    sig_wrapped = _ingest_signature(
        ingest_manual_align_package(server_package_root),
        root=ingest_manual_align_package(server_package_root).pkg_root,
    )
    sig_flat = _ingest_signature(
        ingest_manual_align_package(flat_root),
        root=ingest_manual_align_package(flat_root).pkg_root,
    )

    assert sig_canonical == sig_wrapped == sig_flat
