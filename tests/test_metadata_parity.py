"""CONT-08: normalization parity across CLI, CrossSectionManager, and widget ingest apply."""

from __future__ import annotations

from pathlib import Path

import pytest

from linumpy_manual_align.__main__ import resolve_package_level
from linumpy_manual_align.contracts.layout import METADATA_FILENAME
from linumpy_manual_align.contracts.metadata import load_manual_align_metadata
from linumpy_manual_align.contracts.models import SEVERITY_WARNING
from linumpy_manual_align.io.package_ingest import ingest_manual_align_package
from linumpy_manual_align.remote.cross_section import CrossSectionManager
from linumpy_manual_align.ui.widget_pair_loading import PairLoadingMixin
from linumpy_manual_align.ui.widget_server import ServerMixin

FIXTURE_CASES = ("golden", "missing", "invalid_json")


class _StatusLabel:
    def __init__(self) -> None:
        self.text = ""

    def setText(self, text: str) -> None:
        self.text = text


class _BtnModeZ:
    def __init__(self) -> None:
        self.enabled = False
        self.tooltip = ""

    def setEnabled(self, enabled: bool) -> None:
        self.enabled = enabled

    def setToolTip(self, tooltip: str) -> None:
        self.tooltip = tooltip


def _issue_tuples(issues) -> list[tuple]:
    return [(i.severity, i.code, i.message) for i in issues]


def _expected_cli_level(canonical_norm) -> int | None:
    if not canonical_norm.pyramid_level_explicit:
        return None
    return canonical_norm.pyramid_level


def _materialize_fixture(
    case_name: str, tmp_path: Path, copy_fixture_tree
) -> Path:
    if case_name == "golden":
        return copy_fixture_tree("manual_align_package", tmp_path / "golden")
    if case_name == "missing":
        pkg_root = tmp_path / "missing"
        pkg_root.mkdir()
        return pkg_root
    if case_name == "invalid_json":
        pkg_root = tmp_path / "invalid_json"
        pkg_root.mkdir()
        (pkg_root / METADATA_FILENAME).write_text("not valid json {{{")
        return pkg_root
    raise ValueError(f"unknown fixture case: {case_name}")


class _WidgetStub(ServerMixin, PairLoadingMixin):
    """Minimal stack for _apply_package_ingest parity tests."""


def _make_widget(*, level: int = 42) -> _WidgetStub:
    widget = object.__new__(_WidgetStub)
    widget.level = level
    widget.server_status_label = _StatusLabel()
    widget._cs_mgr = CrossSectionManager()
    widget._btn_mode_z = _BtnModeZ()
    widget.aips_dir = None
    widget.slice_paths = {}
    widget.slice_ids = []
    widget.pair_paths_xy = {}
    widget.aips_xz_dir = None
    widget.aips_yz_dir = None
    widget.slice_paths_xz = {}
    widget.slice_paths_yz = {}
    widget.pair_paths_xz = {}
    widget.pair_paths_yz = {}
    widget.transforms_dir = None
    widget.existing_transforms = {}
    return widget


@pytest.mark.parametrize("case_name", FIXTURE_CASES)
def test_cli_and_manager_parity(
    case_name: str, tmp_path: Path, copy_fixture_tree, qapp
) -> None:
    """CLI resolve_package_level and CrossSectionManager.apply_metadata match canonical load."""
    pkg_root = _materialize_fixture(case_name, tmp_path, copy_fixture_tree)
    canonical_norm, canonical_issues = load_manual_align_metadata(pkg_root)

    cli_level = resolve_package_level(pkg_root)
    assert cli_level == _expected_cli_level(canonical_norm)

    mgr = CrossSectionManager()
    mgr.apply_metadata(canonical_norm, list(canonical_issues))
    assert mgr.cs_level == canonical_norm.pyramid_level
    assert mgr.slices_remote_dir == canonical_norm.slices_remote_dir
    assert mgr.slice_filenames == dict(canonical_norm.slice_filenames)
    assert mgr.slice_remote_paths == dict(canonical_norm.slice_remote_paths)
    assert mgr.interpolated_slice_ids == set(canonical_norm.interpolated_slice_ids)

    _, issues_again = load_manual_align_metadata(pkg_root)
    assert _issue_tuples(canonical_issues) == _issue_tuples(issues_again)


@pytest.mark.parametrize("case_name", FIXTURE_CASES)
def test_widget_hook_parity(
    case_name: str, tmp_path: Path, copy_fixture_tree, qapp
) -> None:
    """PairLoadingMixin._apply_package_ingest matches CLI gating and warning surfacing."""
    pkg_root = _materialize_fixture(case_name, tmp_path, copy_fixture_tree)
    canonical_norm, canonical_issues = load_manual_align_metadata(pkg_root)
    starting_level = 42
    base_status = "base"
    widget = _make_widget(level=starting_level)
    ingest_result = ingest_manual_align_package(pkg_root)

    widget._apply_package_ingest(ingest_result, base_status=base_status)

    cli_level = resolve_package_level(pkg_root)
    if canonical_norm.pyramid_level_explicit:
        assert widget.level == canonical_norm.pyramid_level
        assert cli_level == canonical_norm.pyramid_level
    else:
        assert widget.level == starting_level
        assert cli_level is None

    expected_warnings = [
        issue.message for issue in canonical_issues if issue.severity == SEVERITY_WARNING
    ]
    if expected_warnings:
        assert widget.server_status_label.text == base_status + "\n" + "\n".join(
            expected_warnings
        )
    else:
        assert widget.server_status_label.text == base_status
