"""Tests for the headless workflow contract layer."""

from __future__ import annotations

import json
import shutil
from dataclasses import FrozenInstanceError
from pathlib import Path

import numpy as np
import pytest

from linumpy_manual_align.contracts import (
    MANUAL_TRANSFORMS_DIRNAME,
    METADATA_FILENAME,
    REQUIRED_OUTPUT_FILES,
    SEVERITY_ERROR,
    SEVERITY_INFO,
    SEVERITY_WARNING,
    ContractIssue,
    discover_manual_slice_dirs,
    format_manual_slice_dir,
    manual_output_dir,
    parse_manual_slice_dir,
    resolve_package_layout,
    validate_manual_output,
)
from linumpy_manual_align.io.transform_io import discover_aips


def test_contract_layout_is_inspectable(tmp_path: Path) -> None:
    """Contract package exposes layout constants and resolves package paths under root."""
    assert "transform.tfm" in REQUIRED_OUTPUT_FILES
    assert "offsets.txt" in REQUIRED_OUTPUT_FILES
    assert "pairwise_registration_metrics.json" in REQUIRED_OUTPUT_FILES
    assert MANUAL_TRANSFORMS_DIRNAME == "manual_transforms"
    assert METADATA_FILENAME == "manual_align_metadata.json"

    layout = resolve_package_layout(tmp_path)
    assert layout.aips_dir == tmp_path / "aips"
    assert layout.transforms_dir == tmp_path / "transforms"
    assert layout.metadata_path == tmp_path / METADATA_FILENAME
    assert layout.manual_transforms_dir == tmp_path / MANUAL_TRANSFORMS_DIRNAME


def test_manual_output_path_uses_moving_id(tmp_path: Path) -> None:
    """Manual output directories are keyed solely by moving slice ID."""
    root = tmp_path / "manual_transforms"

    for moving_id in (1, 9, 99, 100):
        out_dir = manual_output_dir(root, moving_id)
        expected_name = format_manual_slice_dir(moving_id)
        assert out_dir == root / expected_name
        assert out_dir.name == expected_name

    for moving_id in (0, 3, 17):
        out_dir = manual_output_dir(root, moving_id)
        expected_name = format_manual_slice_dir(moving_id)
        assert out_dir.name == expected_name
        assert out_dir == root / expected_name


def test_strict_slice_dir_parsing() -> None:
    """Strict parsing accepts exact slice_z names and rejects decorated variants."""
    assert parse_manual_slice_dir("slice_z01") == 1
    assert parse_manual_slice_dir("slice_z09") == 9
    assert parse_manual_slice_dir("slice_z100") == 100

    base = "slice_z01"
    backup_suffix = "_backup"
    prefix = "bad_"
    non_digit_variant = "slice_z01a"

    assert parse_manual_slice_dir(base + backup_suffix) is None
    assert parse_manual_slice_dir(prefix + base) is None
    assert parse_manual_slice_dir(non_digit_variant) is None

    for moving_id in range(151):
        name = format_manual_slice_dir(moving_id)
        assert parse_manual_slice_dir(name) == moving_id


def test_contract_issues_are_structured(tmp_path: Path) -> None:
    """Validation issues carry severity, stable codes, messages, and optional paths."""
    moving_id = 4
    output_dir = tmp_path / format_manual_slice_dir(moving_id)
    output_dir.mkdir()

    issues = validate_manual_output(output_dir, moving_id)
    assert issues
    transform_issues = [i for i in issues if i.code == "output.missing_transform"]
    assert transform_issues
    issue = transform_issues[0]
    assert issue.severity == SEVERITY_ERROR
    assert "." in issue.code
    assert issue.message
    assert issue.affected_path is not None

    frozen = ContractIssue(
        severity=SEVERITY_WARNING,
        code="test.example",
        message="Example issue",
    )
    with pytest.raises(FrozenInstanceError):
        frozen.severity = SEVERITY_INFO  # type: ignore[misc]


class TestMetadataNormalization:
    def test_metadata_normalization_root_parent_and_malformed(self, tmp_path: Path) -> None:
        """CONT-02: root/parent lookup, missing, malformed-field, and invalid-JSON tolerance."""
        from linumpy_manual_align.contracts import (
            NormalizedMetadata,
            load_manual_align_metadata,
        )
        # Root present: known fields normalized; provenance is the root metadata file.
        pkg_root = tmp_path / "package_root"
        pkg_root.mkdir()
        root_meta = {
            "slices_remote_dir": "/remote/slices",
            "cross_section_level": 2,
            "slice_filenames": {"1": "slice_z01.ome.zarr", "2": "slice_z02.ome.zarr"},
            "slice_remote_paths": {"1": "/abs/slice_z01.ome.zarr"},
            "interpolated_slice_ids": [5, 6],
        }
        root_file = pkg_root / METADATA_FILENAME
        root_file.write_text(json.dumps(root_meta))
        normalized, issues = load_manual_align_metadata(pkg_root)
        assert isinstance(normalized, NormalizedMetadata)
        assert normalized.source_path == root_file
        assert normalized.package_root == pkg_root
        assert normalized.slices_remote_dir == "/remote/slices"
        assert normalized.pyramid_level == 2
        assert normalized.slice_filenames == {1: "slice_z01.ome.zarr", 2: "slice_z02.ome.zarr"}
        assert normalized.slice_remote_paths == {1: "/abs/slice_z01.ome.zarr"}
        assert normalized.interpolated_slice_ids == frozenset({5, 6})
        assert not any(i.severity == SEVERITY_WARNING for i in issues)

        # Parent fallback: metadata only at parent; no recursive search beyond parent.
        parent_only = tmp_path / "parent_only"
        sub_pkg = parent_only / "aips"
        sub_pkg.mkdir(parents=True)
        parent_file = parent_only / METADATA_FILENAME
        parent_file.write_text(json.dumps({"slices_remote_dir": "/parent/path", "pyramid_level": 3}))
        normalized, issues = load_manual_align_metadata(sub_pkg)
        assert normalized.source_path == parent_file
        assert normalized.slices_remote_dir == "/parent/path"
        assert normalized.pyramid_level == 3
        assert not any(i.code == "metadata.missing" for i in issues)

        # Missing: defaults plus a single metadata.missing warning.
        missing_root = tmp_path / "missing"
        missing_root.mkdir()
        normalized, issues = load_manual_align_metadata(missing_root)
        assert normalized.pyramid_level == 0
        assert normalized.slices_remote_dir is None
        assert normalized.slice_filenames == {}
        assert normalized.slice_remote_paths == {}
        assert normalized.interpolated_slice_ids == frozenset()
        assert normalized.source_path is None
        missing_issues = [i for i in issues if i.code == "metadata.missing"]
        assert len(missing_issues) == 1
        assert missing_issues[0].severity == SEVERITY_WARNING
        assert missing_issues[0].affected_path == missing_root / METADATA_FILENAME

        # Malformed field: valid fields preserved; bad field defaulted with field-level issue.
        malformed_root = tmp_path / "malformed"
        malformed_root.mkdir()
        malformed_meta = {
            "slices_remote_dir": "/still/valid",
            "cross_section_level": "not-a-number",
            "slice_filenames": {"1": "good.zarr"},
        }
        (malformed_root / METADATA_FILENAME).write_text(json.dumps(malformed_meta))
        normalized, issues = load_manual_align_metadata(malformed_root)
        assert normalized.slices_remote_dir == "/still/valid"
        assert normalized.pyramid_level == 0
        assert normalized.slice_filenames == {1: "good.zarr"}
        field_issues = [i for i in issues if i.code == "metadata.invalid_field"]
        assert len(field_issues) == 1
        assert field_issues[0].field == "cross_section_level"

        # Invalid JSON: defaults plus metadata.invalid_json; no exception escapes.
        invalid_root = tmp_path / "invalid_json"
        invalid_root.mkdir()
        invalid_file = invalid_root / METADATA_FILENAME
        invalid_file.write_text("not valid json {{{")
        normalized, issues = load_manual_align_metadata(invalid_root)
        assert normalized.pyramid_level == 0
        assert normalized.slices_remote_dir is None
        json_issues = [i for i in issues if i.code == "metadata.invalid_json"]
        assert len(json_issues) == 1
        assert json_issues[0].affected_path == invalid_file


class TestSliceAnnotationReconciliation:
    """D-11: metadata slice annotations vs discovered package files."""

    def test_no_issues_when_annotations_match_discovery(self, tmp_path: Path) -> None:
        from linumpy_manual_align.contracts import NormalizedMetadata, reconcile_slice_annotations

        base = "slice_z01"
        filename = f"{base}.ome.zarr"
        discovered = {1: tmp_path / filename}
        metadata = NormalizedMetadata(slice_filenames={1: filename})
        assert reconcile_slice_annotations(metadata, discovered) == []

    def test_warning_when_annotated_id_absent_from_discovery(self, tmp_path: Path) -> None:
        from linumpy_manual_align.contracts import NormalizedMetadata, reconcile_slice_annotations

        metadata = NormalizedMetadata(slice_filenames={99: "slice_z99.ome.zarr"})
        issues = reconcile_slice_annotations(metadata, {})
        assert len(issues) == 1
        issue = issues[0]
        assert issue.severity == SEVERITY_WARNING
        assert issue.code == "metadata.slice_disagreement"
        assert issue.field == "slice_filenames"

    def test_warning_when_filename_disagrees(self, tmp_path: Path) -> None:
        from linumpy_manual_align.contracts import NormalizedMetadata, reconcile_slice_annotations

        base = "slice_z01"
        discovered_name = f"{base}.ome.zarr"
        variant_suffix = "_variant"
        annotated_name = discovered_name.replace(".ome.zarr", f"{variant_suffix}.ome.zarr")
        discovered = {1: tmp_path / discovered_name}
        metadata = NormalizedMetadata(slice_filenames={1: annotated_name})
        issues = reconcile_slice_annotations(metadata, discovered)
        assert len(issues) == 1
        issue = issues[0]
        assert issue.severity == SEVERITY_WARNING
        assert issue.code == "metadata.slice_disagreement"
        assert issue.field == "slice_filenames"
        assert issue.affected_path == discovered[1]


# ---------------------------------------------------------------------------
# Strict slice naming, moving-ID parity, package identity (TEST-05)
# ---------------------------------------------------------------------------


def test_format_manual_slice_dir_padding() -> None:
    """Exact two-digit-minimum padding for moving IDs 1, 99, and 100."""
    assert format_manual_slice_dir(1) == "slice_z01"
    assert format_manual_slice_dir(99) == "slice_z99"
    assert format_manual_slice_dir(100) == "slice_z100"


def _rename_golden_slice_to(
    copy_fixture_tree,
    tmp_path: Path,
    invalid_name: str,
    *,
    dest_key: str,
) -> Path:
    """Copy golden manual_transforms and rename slice_z01 to *invalid_name*."""
    root = copy_fixture_tree("manual_transforms", tmp_path / dest_key)
    canonical = root / "slice_z01"
    invalid_dir = root / invalid_name
    canonical.rename(invalid_dir)
    return invalid_dir


def _assert_single_naming_error(issues: list[ContractIssue]) -> None:
    naming = [i for i in issues if i.code == "naming.invalid_slice_dir"]
    assert len(naming) == 1
    assert naming[0].severity == SEVERITY_ERROR
    assert not any(i.code.startswith(("offsets.", "metrics.")) for i in issues)


@pytest.mark.parametrize(
    ("invalid_name", "moving_id", "dest_key"),
    [
        ("z01", 1, "invalid_z01"),
        ("slice_01", 1, "invalid_slice_01"),
        ("slice_z1", 1, "invalid_slice_z1"),
        ("slice_z01_backup", 1, "invalid_slice_z01_backup"),
    ],
)
def test_invalid_slice_dir_names_rejected(
    copy_fixture_tree,
    tmp_path: Path,
    invalid_name: str,
    moving_id: int,
    dest_key: str,
) -> None:
    """Decorated or non-canonical names fail naming validation with valid golden files."""
    invalid_dir = _rename_golden_slice_to(copy_fixture_tree, tmp_path, invalid_name, dest_key=dest_key)
    issues = validate_manual_output(invalid_dir, moving_id)
    _assert_single_naming_error(issues)


def test_wrong_moving_id_dir_rejected(copy_fixture_tree, tmp_path: Path) -> None:
    """Canonical slice_z01 dir with mismatched moving_id emits naming.invalid_slice_dir."""
    root = copy_fixture_tree("manual_transforms", tmp_path / "wrong_moving_id")
    slice_dir = root / "slice_z01"
    issues = validate_manual_output(slice_dir, moving_id=2)
    _assert_single_naming_error(issues)


def test_moving_slice_naming_parity_for_filtered_ids(copy_fixture_tree, tmp_path: Path) -> None:
    """Output and package slice identity keyed by moving ID, not pair order."""
    slice_ids = [0, 4, 9]
    pairs = [(slice_ids[i], slice_ids[i + 1]) for i in range(len(slice_ids) - 1)]
    moving_ids = [mid for _fixed, mid in pairs]
    assert moving_ids == [4, 9]
    filtered_moving_ids = {9}
    assert filtered_moving_ids == {9}

    manual_root = tmp_path / "manual_transforms"
    for mid in moving_ids:
        canonical = format_manual_slice_dir(mid)
        assert manual_output_dir(manual_root, mid).name == canonical

    golden = copy_fixture_tree("manual_transforms", tmp_path / "golden_copy")
    transforms_root = tmp_path / "filtered_manual"
    transforms_root.mkdir()
    shutil.copytree(golden / "slice_z01", transforms_root / "slice_z09")
    for name in ("slice_z01", "slice_z02"):
        shutil.rmtree(golden / name, ignore_errors=True)

    discovered, _issues = discover_manual_slice_dirs(transforms_root)
    assert list(discovered.keys()) == [9]

    aips_dir = tmp_path / "aips"
    aips_dir.mkdir()
    for mid in moving_ids:
        aip = np.zeros((8, 8), dtype=np.float32)
        np.savez(str(aips_dir / f"{format_manual_slice_dir(mid)}.npz"), aip=aip, scale=np.array([1.0, 1.0]))

    slice_filenames = {"4": "slice_z04.ome.zarr", "9": "slice_z09.ome.zarr"}
    aips = discover_aips(aips_dir)
    assert set(aips.keys()) == set(moving_ids)

    for mid in moving_ids:
        canonical = format_manual_slice_dir(mid)
        meta_stem = slice_filenames[str(mid)].replace(".ome.zarr", "")
        assert aips[mid].stem == canonical
        assert meta_stem == canonical
        assert manual_output_dir(manual_root, mid).name == canonical


def test_duplicate_moving_id_dir_warns_and_errors(copy_fixture_tree, tmp_path: Path) -> None:
    """Duplicate parsed moving IDs warn on discovery and error on non-canonical dir."""
    golden = copy_fixture_tree("manual_transforms", tmp_path / "dup_golden")
    transforms_root = tmp_path / "dup_manual"
    transforms_root.mkdir()
    shutil.copytree(golden / "slice_z01", transforms_root / "slice_z09")
    shutil.copytree(golden / "slice_z02", transforms_root / "slice_z9")

    discovered, issues = discover_manual_slice_dirs(transforms_root)
    assert list(discovered.keys()) == [9]
    assert discovered[9].name == "slice_z09"

    dup_warnings = [i for i in issues if i.code == "naming.duplicate_slice_dir"]
    assert len(dup_warnings) == 1
    assert dup_warnings[0].severity == SEVERITY_WARNING
    assert dup_warnings[0].affected_path == transforms_root / "slice_z9"

    canonical_issues = validate_manual_output(transforms_root / "slice_z09", 9)
    assert not any(i.code == "naming.invalid_slice_dir" for i in canonical_issues)

    noncanonical_issues = validate_manual_output(transforms_root / "slice_z9", 9)
    naming_errors = [i for i in noncanonical_issues if i.code == "naming.invalid_slice_dir"]
    assert len(naming_errors) == 1
    assert naming_errors[0].severity == SEVERITY_ERROR
