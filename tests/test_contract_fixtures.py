"""Headless contract tests against committed canonical fixtures."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from linumpy_manual_align.contracts import (
    SEVERITY_ERROR,
    SEVERITY_WARNING,
    load_manual_align_metadata,
    resolve_package_layout,
    validate_manual_output,
    validate_metrics_file,
    validate_offsets_file,
)
from linumpy_manual_align.io.transform_io import discover_aips, discover_transforms


def _errors(issues):
    return [i for i in issues if i.severity == SEVERITY_ERROR]


def test_canonical_package_fixture_shape(fixtures_root: Path) -> None:
    """Committed manual_align_package mirrors export shape (3 AIPs, 2 transform dirs)."""
    pkg_root = fixtures_root / "manual_align_package"
    layout = resolve_package_layout(pkg_root)

    aips = discover_aips(layout.aips_dir)
    assert set(aips) == {0, 1, 2}

    transforms = discover_transforms(layout.transforms_dir)
    assert set(transforms) == {1, 2}

    normalized, issues = load_manual_align_metadata(pkg_root)
    assert normalized.pyramid_level == 1
    assert _errors(issues) == []


def test_golden_manual_output_fixture_validates(copy_fixture_tree) -> None:
    """Committed golden manual_transforms validate with zero contract errors."""
    root = copy_fixture_tree("manual_transforms")

    for moving_id in (1, 2):
        slice_dir = root / f"slice_z{moving_id:02d}"
        issues = validate_manual_output(slice_dir, moving_id)
        assert _errors(issues) == []


def _valid_manual_metrics() -> dict:
    return {
        "source": "manual",
        "overall_status": "ok",
        "metrics": {
            "translation_x": {"value": 1.0, "unit": "pixels"},
            "translation_y": {"value": 2.0, "unit": "pixels"},
            "translation_magnitude": {"value": 2.236, "unit": "pixels"},
            "rotation": {"value": 0.5, "unit": "degrees"},
        },
        "manual_alignment": {
            "pyramid_level": 1,
            "working_tx": 8.0,
            "working_ty": -5.0,
            "center_working": [120.0, 90.0],
        },
    }


def _assert_single_error(issues, code: str, path: Path) -> None:
    assert len(issues) == 1
    issue = issues[0]
    assert issue.code == code
    assert issue.severity == SEVERITY_ERROR
    assert issue.affected_path == path


class TestValidateOffsetsFile:
    def test_valid_offsets_produce_no_issues(self, tmp_path: Path) -> None:
        offsets_path = tmp_path / "offsets.txt"
        offsets_path.write_text("5\n12\n")
        assert validate_offsets_file(offsets_path) == []

    def test_one_token_is_invalid_format(self, tmp_path: Path) -> None:
        offsets_path = tmp_path / "offsets.txt"
        offsets_path.write_text("5\n")
        _assert_single_error(
            validate_offsets_file(offsets_path),
            "offsets.invalid_format",
            offsets_path,
        )

    def test_three_tokens_is_invalid_format(self, tmp_path: Path) -> None:
        offsets_path = tmp_path / "offsets.txt"
        offsets_path.write_text("5 12 3\n")
        _assert_single_error(
            validate_offsets_file(offsets_path),
            "offsets.invalid_format",
            offsets_path,
        )

    def test_non_integer_token_is_invalid_format(self, tmp_path: Path) -> None:
        offsets_path = tmp_path / "offsets.txt"
        offsets_path.write_text("5 abc\n")
        _assert_single_error(
            validate_offsets_file(offsets_path),
            "offsets.invalid_format",
            offsets_path,
        )

    def test_float_token_is_invalid_format(self, tmp_path: Path) -> None:
        offsets_path = tmp_path / "offsets.txt"
        offsets_path.write_text("5.0 12\n")
        _assert_single_error(
            validate_offsets_file(offsets_path),
            "offsets.invalid_format",
            offsets_path,
        )

    def test_empty_file_is_invalid_format(self, tmp_path: Path) -> None:
        offsets_path = tmp_path / "offsets.txt"
        offsets_path.write_text("")
        _assert_single_error(
            validate_offsets_file(offsets_path),
            "offsets.invalid_format",
            offsets_path,
        )


class TestValidateMetricsFile:
    def test_valid_manual_metrics_produce_no_issues(self, tmp_path: Path) -> None:
        metrics_path = tmp_path / "pairwise_registration_metrics.json"
        metrics_path.write_text(json.dumps(_valid_manual_metrics()))
        assert validate_metrics_file(metrics_path) == []

    def test_invalid_json_produces_invalid_json(self, tmp_path: Path) -> None:
        metrics_path = tmp_path / "pairwise_registration_metrics.json"
        metrics_path.write_text("not json {{{")
        _assert_single_error(
            validate_metrics_file(metrics_path),
            "metrics.invalid_json",
            metrics_path,
        )

    def test_non_object_json_produces_invalid_json(self, tmp_path: Path) -> None:
        metrics_path = tmp_path / "pairwise_registration_metrics.json"
        metrics_path.write_text(json.dumps([1, 2, 3]))
        _assert_single_error(
            validate_metrics_file(metrics_path),
            "metrics.invalid_json",
            metrics_path,
        )

    def test_missing_source_produces_missing_source(self, tmp_path: Path) -> None:
        metrics_path = tmp_path / "pairwise_registration_metrics.json"
        payload = _valid_manual_metrics()
        del payload["source"]
        metrics_path.write_text(json.dumps(payload))
        _assert_single_error(
            validate_metrics_file(metrics_path),
            "metrics.missing_source",
            metrics_path,
        )

    def test_non_manual_source_produces_invalid_source(self, tmp_path: Path) -> None:
        metrics_path = tmp_path / "pairwise_registration_metrics.json"
        payload = _valid_manual_metrics()
        payload["source"] = "automated"
        metrics_path.write_text(json.dumps(payload))
        _assert_single_error(
            validate_metrics_file(metrics_path),
            "metrics.invalid_source",
            metrics_path,
        )

    def test_bad_status_produces_invalid_status(self, tmp_path: Path) -> None:
        metrics_path = tmp_path / "pairwise_registration_metrics.json"
        payload = _valid_manual_metrics()
        payload["overall_status"] = "failed"
        metrics_path.write_text(json.dumps(payload))
        _assert_single_error(
            validate_metrics_file(metrics_path),
            "metrics.invalid_status",
            metrics_path,
        )

    def test_missing_translation_x_produces_missing_field(self, tmp_path: Path) -> None:
        metrics_path = tmp_path / "pairwise_registration_metrics.json"
        payload = _valid_manual_metrics()
        del payload["metrics"]["translation_x"]
        metrics_path.write_text(json.dumps(payload))
        issues = validate_metrics_file(metrics_path)
        assert len(issues) == 1
        assert issues[0].code == "metrics.missing_field"
        assert issues[0].severity == SEVERITY_ERROR
        assert issues[0].affected_path == metrics_path

    def test_non_numeric_translation_y_produces_missing_field(self, tmp_path: Path) -> None:
        metrics_path = tmp_path / "pairwise_registration_metrics.json"
        payload = _valid_manual_metrics()
        payload["metrics"]["translation_y"] = {"value": "bad", "unit": "pixels"}
        metrics_path.write_text(json.dumps(payload))
        issues = validate_metrics_file(metrics_path)
        assert len(issues) == 1
        assert issues[0].code == "metrics.missing_field"
        assert issues[0].severity == SEVERITY_ERROR

    def test_bool_metric_value_produces_missing_field(self, tmp_path: Path) -> None:
        metrics_path = tmp_path / "pairwise_registration_metrics.json"
        payload = _valid_manual_metrics()
        payload["metrics"]["rotation"] = {"value": True, "unit": "degrees"}
        metrics_path.write_text(json.dumps(payload))
        issues = validate_metrics_file(metrics_path)
        assert len(issues) == 1
        assert issues[0].code == "metrics.missing_field"
        assert issues[0].severity == SEVERITY_ERROR

    def test_missing_manual_alignment_field_produces_missing_field(self, tmp_path: Path) -> None:
        metrics_path = tmp_path / "pairwise_registration_metrics.json"
        payload = _valid_manual_metrics()
        del payload["manual_alignment"]["pyramid_level"]
        metrics_path.write_text(json.dumps(payload))
        issues = validate_metrics_file(metrics_path)
        assert len(issues) == 1
        assert issues[0].code == "metrics.missing_field"
        assert issues[0].severity == SEVERITY_ERROR


def _issue_codes(issues) -> set[str]:
    return {issue.code for issue in issues}


def test_negative_contract_cases_return_issues(copy_fixture_tree, tmp_path: Path) -> None:
    """Generated negative cases cover D-12 mandatory contract failures (TEST-03)."""
    moving_id = 1

    missing_transform_root = copy_fixture_tree("manual_transforms", tmp_path / "missing_transform")
    missing_transform_slice = missing_transform_root / "slice_z01"
    (missing_transform_slice / "transform.tfm").unlink()
    issues = validate_manual_output(missing_transform_slice, moving_id)
    assert "output.missing_transform" in _issue_codes(issues)

    missing_offsets_root = copy_fixture_tree("manual_transforms", tmp_path / "missing_offsets")
    missing_offsets_slice = missing_offsets_root / "slice_z01"
    (missing_offsets_slice / "offsets.txt").unlink()
    issues = validate_manual_output(missing_offsets_slice, moving_id)
    assert "output.missing_offsets" in _issue_codes(issues)

    missing_metrics_root = copy_fixture_tree("manual_transforms", tmp_path / "missing_metrics")
    missing_metrics_slice = missing_metrics_root / "slice_z01"
    (missing_metrics_slice / "pairwise_registration_metrics.json").unlink()
    issues = validate_manual_output(missing_metrics_slice, moving_id)
    assert "output.missing_metrics" in _issue_codes(issues)

    invalid_name_root = copy_fixture_tree("manual_transforms", tmp_path / "invalid_name")
    invalid_name_slice = invalid_name_root / "slice_z01"
    decorated = invalid_name_root / "slice_z01_backup"
    invalid_name_slice.rename(decorated)
    issues = validate_manual_output(decorated, moving_id)
    assert "naming.invalid_slice_dir" in _issue_codes(issues)

    malformed_offsets_root = copy_fixture_tree("manual_transforms", tmp_path / "malformed_offsets")
    malformed_offsets_slice = malformed_offsets_root / "slice_z01"
    (malformed_offsets_slice / "offsets.txt").write_text("5\n")
    issues = validate_manual_output(malformed_offsets_slice, moving_id)
    assert "offsets.invalid_format" in _issue_codes(issues)

    invalid_source_root = copy_fixture_tree("manual_transforms", tmp_path / "invalid_source")
    invalid_source_slice = invalid_source_root / "slice_z01"
    invalid_source_metrics = invalid_source_slice / "pairwise_registration_metrics.json"
    payload = json.loads(invalid_source_metrics.read_text())
    payload["source"] = "automated"
    invalid_source_metrics.write_text(json.dumps(payload))
    issues = validate_manual_output(invalid_source_slice, moving_id)
    assert "metrics.invalid_source" in _issue_codes(issues)

    missing_source_root = copy_fixture_tree("manual_transforms", tmp_path / "missing_source")
    missing_source_slice = missing_source_root / "slice_z01"
    missing_source_metrics = missing_source_slice / "pairwise_registration_metrics.json"
    payload = json.loads(missing_source_metrics.read_text())
    del payload["source"]
    missing_source_metrics.write_text(json.dumps(payload))
    issues = validate_manual_output(missing_source_slice, moving_id)
    assert "metrics.missing_source" in _issue_codes(issues)

    pkg_root = copy_fixture_tree("manual_align_package", tmp_path / "bad_metadata")
    metadata_path = pkg_root / "manual_align_metadata.json"
    metadata_path.write_text("not valid json {{{")
    normalized, issues = load_manual_align_metadata(pkg_root)
    assert normalized.pyramid_level == 0
    json_issues = [i for i in issues if i.code == "metadata.invalid_json"]
    assert len(json_issues) == 1
    assert json_issues[0].severity == SEVERITY_WARNING

    corrupt_root = copy_fixture_tree("manual_transforms", tmp_path / "corrupt_transform")
    corrupt_slice = corrupt_root / "slice_z01"
    (corrupt_slice / "transform.tfm").write_bytes(b"not a transform file")
    from linumpy_manual_align.io.transform_io import load_transform

    with pytest.raises(Exception):
        load_transform(corrupt_slice / "transform.tfm")
