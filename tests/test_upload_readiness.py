"""Headless tests for upload readiness classification."""

from __future__ import annotations

import json
import shutil
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from linumpy_manual_align.contracts import (
    SEVERITY_ERROR,
    SEVERITY_WARNING,
    UPLOAD_NOT_SAVED,
    UPLOAD_NO_OUTPUT_DIR,
    ContractIssue,
    PairUploadReadiness,
    PairUploadStatus,
    UploadReadinessReport,
    assess_upload_readiness,
    format_upload_issue_line,
)
from linumpy_manual_align.contracts import upload_readiness as upload_readiness_module

DEFAULT_PAIRS = [(0, 1), (1, 2)]


def _assess(
    copy_fixture_tree,
    tmp_path: Path,
    *,
    pairs: list[tuple[int, int]] | None = None,
    saved: set[int] | frozenset[int],
    fixture_name: str = "manual_transforms",
    transforms_root: Path | None = None,
):
    if transforms_root is None:
        transforms_root = copy_fixture_tree(fixture_name)
    return assess_upload_readiness(
        pairs or DEFAULT_PAIRS,
        transforms_root,
        saved,
    )


def test_t1_all_ready(copy_fixture_tree) -> None:
    """T1: golden z01+z02 with saved={1,2} -> both READY."""
    report = _assess(copy_fixture_tree, Path(), saved={1, 2})

    assert report.ready_count == 2
    assert report.missing_count == 0
    assert report.invalid_count == 0
    assert report.warning_count == 0
    assert not report.has_blocking_errors
    assert len(report.ready_dirs) == 2
    assert all(p.status == PairUploadStatus.READY for p in report.pairs)


def test_t2_ready_with_warning(
    copy_fixture_tree, monkeypatch: pytest.MonkeyPatch
) -> None:
    """T2: saved pair with WARNING-only validation -> READY, warning_count incremented."""
    warning = ContractIssue(
        severity=SEVERITY_WARNING,
        code="test.warning_only",
        message="Non-blocking warning for upload readiness.",
        affected_path=None,
    )
    monkeypatch.setattr(
        upload_readiness_module,
        "validate_manual_output",
        lambda out_dir, moving_id: [warning],
        raising=False,
    )

    root = copy_fixture_tree("manual_transforms")
    report = assess_upload_readiness([(0, 1)], root, {1})

    assert report.ready_count == 1
    assert report.missing_count == 0
    assert report.invalid_count == 0
    assert report.warning_count == 1
    assert not report.has_blocking_errors
    assert report.pairs[0].status == PairUploadStatus.READY
    assert len(report.pairs[0].issues) == 1
    assert report.pairs[0].issues[0].severity == SEVERITY_WARNING
    assert len(report.ready_dirs) == 1
    assert len(report.warning_lines()) == 1
    assert "test.warning_only" in report.warning_lines()[0]


def test_t3_not_saved(copy_fixture_tree) -> None:
    """T3: golden present but saved=set() -> both MISSING with upload.not_saved."""
    report = _assess(copy_fixture_tree, Path(), saved=set())

    assert report.ready_count == 0
    assert report.missing_count == 2
    assert report.invalid_count == 0
    assert report.warning_count == 0
    assert report.has_blocking_errors
    for pair in report.pairs:
        assert pair.status == PairUploadStatus.MISSING
        assert len(pair.issues) == 1
        assert pair.issues[0].code == UPLOAD_NOT_SAVED
        assert pair.issues[0].severity == SEVERITY_ERROR


def test_t4_saved_no_dir(copy_fixture_tree, tmp_path: Path) -> None:
    """T4: saved={1}, slice_z01 removed -> MISSING upload.no_output_dir."""
    root = copy_fixture_tree("manual_transforms", tmp_path / "t4")
    shutil.rmtree(root / "slice_z01")

    report = assess_upload_readiness([(0, 1)], root, {1})

    assert report.ready_count == 0
    assert report.missing_count == 1
    assert report.invalid_count == 0
    assert report.warning_count == 0
    assert report.pairs[0].status == PairUploadStatus.MISSING
    assert report.pairs[0].issues[0].code == UPLOAD_NO_OUTPUT_DIR


def test_t5_invalid_contract(copy_fixture_tree, tmp_path: Path) -> None:
    """T5: saved={1}, offsets.txt deleted -> INVALID."""
    root = copy_fixture_tree("manual_transforms", tmp_path / "t5")
    (root / "slice_z01" / "offsets.txt").unlink()

    report = assess_upload_readiness([(0, 1)], root, {1})

    assert report.ready_count == 0
    assert report.missing_count == 0
    assert report.invalid_count == 1
    assert report.warning_count == 0
    assert report.has_blocking_errors
    assert report.pairs[0].status == PairUploadStatus.INVALID


def test_t8_mixed(copy_fixture_tree, tmp_path: Path) -> None:
    """T8: saved={1}, slice_z02 removed -> ready z01 + missing z02."""
    root = copy_fixture_tree("manual_transforms", tmp_path / "t8")
    shutil.rmtree(root / "slice_z02")

    report = assess_upload_readiness(DEFAULT_PAIRS, root, {1})

    assert report.ready_count == 1
    assert report.missing_count == 1
    assert report.invalid_count == 0
    assert report.warning_count == 0
    assert report.has_blocking_errors
    by_id = {p.moving_id: p for p in report.pairs}
    assert by_id[1].status == PairUploadStatus.READY
    assert by_id[2].status == PairUploadStatus.MISSING
    assert by_id[2].issues[0].code == UPLOAD_NOT_SAVED


def test_empty_session_pairs_not_blocking(tmp_path: Path) -> None:
    """Empty pairs: all counts zero; has_blocking_errors False (T-4-10 / D-11 known gap)."""
    empty_root = tmp_path / "transforms"
    empty_root.mkdir()

    report = assess_upload_readiness([], empty_root, set())

    assert report.ready_count == 0
    assert report.missing_count == 0
    assert report.invalid_count == 0
    assert report.warning_count == 0
    assert len(report.pairs) == 0
    assert not report.has_blocking_errors
    assert report.summary_line() == "0 ready, 0 missing, 0 invalid, 0 warning"


def test_t11_zero_ready_empty(tmp_path: Path) -> None:
    """T11: empty transforms_root, saved=set() -> all MISSING."""
    empty_root = tmp_path / "empty_transforms"
    empty_root.mkdir()

    report = assess_upload_readiness(DEFAULT_PAIRS, empty_root, set())

    assert report.ready_count == 0
    assert report.missing_count == 2
    assert report.has_blocking_errors


def test_t8_summary_line(copy_fixture_tree, tmp_path: Path) -> None:
    """T8 summary_line renders exact aggregate counts."""
    root = copy_fixture_tree("manual_transforms", tmp_path / "t8_summary")
    shutil.rmtree(root / "slice_z02")

    report = assess_upload_readiness(DEFAULT_PAIRS, root, {1})

    assert report.summary_line() == "1 ready, 1 missing, 0 invalid, 0 warning"


def test_t8_error_lines_missing_pair(copy_fixture_tree, tmp_path: Path) -> None:
    """error_lines() includes MISSING pair with (missing) prefix token."""
    root = copy_fixture_tree("manual_transforms", tmp_path / "t8_errors")
    shutil.rmtree(root / "slice_z02")

    report = assess_upload_readiness(DEFAULT_PAIRS, root, {1})
    missing_lines = [
        ln for ln in report.error_lines() if ln.startswith("z02:")
    ]

    assert len(missing_lines) == 1
    assert missing_lines[0].startswith("z02: (missing) - upload.not_saved:")


class TestFormatUploadIssueLine:
    def test_affected_path_present(self, tmp_path: Path) -> None:
        affected = tmp_path / "offsets.txt"
        issue = ContractIssue(
            severity=SEVERITY_ERROR,
            code="output.missing_offsets",
            message="Required output file 'offsets.txt' is missing.",
            affected_path=affected,
        )
        line = format_upload_issue_line(1, issue, tmp_path / "slice_z01")
        assert line == (
            "z01: offsets.txt - output.missing_offsets: "
            "Required output file 'offsets.txt' is missing."
        )

    def test_output_dir_fallback(self, tmp_path: Path) -> None:
        out_dir = tmp_path / "slice_z01"
        issue = ContractIssue(
            severity=SEVERITY_ERROR,
            code=UPLOAD_NO_OUTPUT_DIR,
            message="No output directory on disk",
            affected_path=None,
        )
        line = format_upload_issue_line(1, issue, out_dir)
        assert line == "z01: slice_z01 - upload.no_output_dir: No output directory on disk"

    def test_missing_token(self) -> None:
        issue = ContractIssue(
            severity=SEVERITY_ERROR,
            code=UPLOAD_NOT_SAVED,
            message="Pair not saved in this session",
            affected_path=None,
        )
        line = format_upload_issue_line(2, issue, None)
        assert line == "z02: (missing) - upload.not_saved: Pair not saved in this session"


def test_t6_loose_dir_name(copy_fixture_tree, tmp_path: Path) -> None:
    """T6: slice_z1 loose name -> INVALID with naming.invalid_slice_dir."""
    root = copy_fixture_tree("manual_transforms", tmp_path / "t6")
    (root / "slice_z01").rename(root / "slice_z1")

    report = assess_upload_readiness([(0, 1)], root, {1})

    assert report.invalid_count == 1
    assert report.has_blocking_errors
    assert report.pairs[0].status == PairUploadStatus.INVALID
    assert any(
        issue.code == "naming.invalid_slice_dir" for issue in report.pairs[0].issues
    )
    assert any(
        "naming.invalid_slice_dir" in line for line in report.error_lines()
    )


def test_t7_stale_failed_save(copy_fixture_tree, tmp_path: Path) -> None:
    """T7: non-manual metrics source -> INVALID metrics.invalid_source."""
    root = copy_fixture_tree("manual_transforms", tmp_path / "t7")
    metrics_path = root / "slice_z01" / "pairwise_registration_metrics.json"
    payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    payload["source"] = "automated"
    metrics_path.write_text(json.dumps(payload), encoding="utf-8")

    report = assess_upload_readiness([(0, 1)], root, {1})

    assert report.invalid_count == 1
    assert report.has_blocking_errors
    assert any(
        issue.code == "metrics.invalid_source" for issue in report.pairs[0].issues
    )


def test_t9_filter_slices_scope(copy_fixture_tree, tmp_path: Path) -> None:
    """T9: only the requested session pair is assessed (D-14)."""
    root = copy_fixture_tree("manual_transforms", tmp_path / "t9")
    shutil.copytree(root / "slice_z01", root / "slice_z09")
    assert (root / "slice_z01").is_dir()

    report = assess_upload_readiness([(4, 9)], root, {9})

    assert report.ready_count == 1
    assert len(report.pairs) == 1
    assert report.pairs[0].moving_id == 9
    assert len(report.ready_dirs) == 1
    assert report.ready_dirs[0].name == "slice_z09"


def test_t10_extra_non_session_dir(copy_fixture_tree, tmp_path: Path) -> None:
    """T10: unrelated on-disk dirs never enter the readiness report."""
    root = copy_fixture_tree("manual_transforms", tmp_path / "t10")
    shutil.copytree(root / "slice_z01", root / "slice_z99")

    report = assess_upload_readiness([(0, 1)], root, {1})

    assert report.ready_count == 1
    assert len(report.pairs) == 1
    assert report.pairs[0].moving_id == 1
    assert all("slice_z99" not in str(path) for path in report.ready_dirs)
    assert all(pair.moving_id != 99 for pair in report.pairs)


def test_t12_multi_error_pair(copy_fixture_tree, tmp_path: Path) -> None:
    """T12: INVALID pairs emit every ERROR line, not just the primary issue."""
    root = copy_fixture_tree("manual_transforms", tmp_path / "t12")
    slice_dir = root / "slice_z01"
    (slice_dir / "offsets.txt").unlink()
    metrics_path = slice_dir / "pairwise_registration_metrics.json"
    payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    payload["source"] = "automated"
    metrics_path.write_text(json.dumps(payload), encoding="utf-8")

    report = assess_upload_readiness([(0, 1)], root, {1})

    assert report.invalid_count == 1
    z01_lines = [line for line in report.error_lines() if line.startswith("z01:")]
    assert len(z01_lines) >= 2


def test_upload_readiness_records_are_immutable(copy_fixture_tree) -> None:
    """Frozen dataclasses reject attribute assignment."""
    report = _assess(copy_fixture_tree, Path(), saved={1, 2})
    pair = report.pairs[0]

    with pytest.raises(FrozenInstanceError):
        report.ready_count = 0  # type: ignore[misc]

    with pytest.raises(FrozenInstanceError):
        pair.status = PairUploadStatus.INVALID  # type: ignore[misc]
