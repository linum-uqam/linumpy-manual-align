"""Tests for widget data structures and CLI argument parsing."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from unittest.mock import MagicMock

import pytest

import linumpy_manual_align.ui.widget_close_guard as widget_close_guard_module
import linumpy_manual_align.ui.widget_server as widget_server_module
import linumpy_manual_align.ui.widget_undo_save as widget_undo_save_module
from linumpy_manual_align.__main__ import parse_args
from linumpy_manual_align.contracts import (
    SEVERITY_ERROR,
    SEVERITY_WARNING,
    ContractIssue,
    PairUploadReadiness,
    PairUploadStatus,
    UploadReadinessReport,
    validate_manual_output,
)
from linumpy_manual_align.contracts.layout import (
    METADATA_FILENAME,
    TRANSFORM_FILENAME,
    manual_output_dir,
)
from linumpy_manual_align.io.transform_io import save_transform
from linumpy_manual_align.state import _MAX_UNDO_HISTORY, AlignmentState, UndoStack
from linumpy_manual_align.ui.widget_close_guard import CloseGuardMixin
from linumpy_manual_align.ui.widget_mixins import PairNavigationMixin
from linumpy_manual_align.ui.widget_server import ServerMixin
from linumpy_manual_align.ui.widget_status import StatusMixin
from linumpy_manual_align.ui.widget_undo_save import UndoSaveMixin


class _SaveWidget(UndoSaveMixin):
    pass


class _SaveCloseWidget(UndoSaveMixin, CloseGuardMixin):
    pass


class _StatusLabel:
    def __init__(self) -> None:
        self.text = ""

    def setText(self, text: str) -> None:
        self.text = text


class _Viewer:
    def __init__(self) -> None:
        self.status = ""
        self.closed = False

    def close(self) -> None:
        self.closed = True


def _make_save_widget(output_dir: Path, mid: int, level: int = 1) -> _SaveWidget:
    widget = object.__new__(_SaveWidget)
    widget.pairs = [(mid - 1, mid)]
    widget.current_pair_idx = 0
    widget.output_dir = output_dir
    widget.level = level
    widget.pair_centers = {mid: (50.0, 50.0)}
    widget._current_offsets = {mid: (0, 0)}
    widget.saved_pairs = set()
    widget.unsaved_changes = {mid}
    widget.undo_stacks = {mid: UndoStack(AlignmentState(tx=1.0, ty=2.0, rotation=0.0))}
    widget.viewer = _Viewer()
    widget.status_label = _StatusLabel()
    widget._flash_saved_calls: list[int] = []
    widget._current_state = lambda: AlignmentState(tx=1.0, ty=2.0, rotation=0.0)
    widget._flash_saved = lambda m: widget._flash_saved_calls.append(m)
    return widget


def _make_batch_widget(output_dir: Path, mids: list[int], level: int = 1) -> _SaveWidget:
    widget = object.__new__(_SaveWidget)
    widget.pairs = [(m - 1, m) for m in mids]
    widget.current_pair_idx = 0
    widget.output_dir = output_dir
    widget.level = level
    widget.pair_centers = dict.fromkeys(mids, (50.0, 50.0))
    widget._current_offsets = dict.fromkeys(mids, (0, 0))
    widget.saved_pairs = set()
    widget.unsaved_changes = set(mids)
    widget.undo_stacks = {
        m: UndoStack(AlignmentState(tx=1.0, ty=2.0, rotation=0.0)) for m in mids
    }
    widget.viewer = _Viewer()
    widget.status_label = _StatusLabel()
    widget._close_confirmed = False
    widget._flash_saved_calls: list[int] = []
    widget._flash_saved = lambda m: widget._flash_saved_calls.append(m)
    widget._current_state = lambda: AlignmentState(tx=1.0, ty=2.0, rotation=0.0)
    return widget


def _batch_validate_side_effect(fail_mid: int, issue: ContractIssue):
    """Return a validate callable: [] for pairs before fail_mid, [issue] at fail_mid."""

    def fake_validate(out_dir: Path, mid: int) -> list[ContractIssue]:
        if mid < fail_mid:
            return []
        if mid == fail_mid:
            return [issue]
        return []

    return fake_validate


class TestAlignmentState:
    def test_defaults(self) -> None:
        s = AlignmentState()
        assert s.tx == 0.0
        assert s.ty == 0.0
        assert s.rotation == 0.0

    def test_custom_values(self) -> None:
        s = AlignmentState(tx=1.5, ty=-2.3, rotation=0.7)
        assert s.tx == 1.5
        assert s.ty == -2.3
        assert s.rotation == 0.7


class TestUndoStack:
    def test_initial_state_default(self) -> None:
        stack = UndoStack()
        assert stack.current.tx == 0.0
        assert stack.current.ty == 0.0

    def test_initial_state_custom(self) -> None:
        stack = UndoStack(AlignmentState(tx=5.0, ty=3.0, rotation=1.0))
        assert stack.current.tx == 5.0
        assert stack.current.rotation == 1.0

    def test_push_and_current(self) -> None:
        stack = UndoStack()
        stack.push(AlignmentState(tx=10.0))
        assert stack.current.tx == 10.0

    def test_undo(self) -> None:
        stack = UndoStack()
        stack.push(AlignmentState(tx=10.0))
        result = stack.undo()
        assert result is not None
        assert result.tx == 0.0
        assert stack.current.tx == 0.0

    def test_undo_at_beginning(self) -> None:
        stack = UndoStack()
        assert stack.undo() is None

    def test_redo(self) -> None:
        stack = UndoStack()
        stack.push(AlignmentState(tx=10.0))
        stack.undo()
        result = stack.redo()
        assert result is not None
        assert result.tx == 10.0

    def test_redo_at_end(self) -> None:
        stack = UndoStack()
        assert stack.redo() is None

    def test_push_discards_redo_history(self) -> None:
        stack = UndoStack()
        stack.push(AlignmentState(tx=1.0))
        stack.push(AlignmentState(tx=2.0))
        stack.undo()  # back to tx=1
        stack.push(AlignmentState(tx=3.0))  # discards tx=2
        assert stack.redo() is None
        assert stack.current.tx == 3.0

    def test_multiple_undo_redo(self) -> None:
        stack = UndoStack()
        stack.push(AlignmentState(tx=1.0))
        stack.push(AlignmentState(tx=2.0))
        stack.push(AlignmentState(tx=3.0))

        stack.undo()
        assert stack.current.tx == 2.0
        stack.undo()
        assert stack.current.tx == 1.0
        stack.undo()
        assert stack.current.tx == 0.0
        assert stack.undo() is None

        stack.redo()
        assert stack.current.tx == 1.0
        stack.redo()
        assert stack.current.tx == 2.0

    def test_max_history_enforced(self) -> None:
        stack = UndoStack()
        for i in range(1, _MAX_UNDO_HISTORY + 100):
            stack.push(AlignmentState(tx=float(i)))
        # History should be capped
        assert stack.current.tx == float(_MAX_UNDO_HISTORY + 99)
        # We should be able to undo _MAX_UNDO_HISTORY - 1 times (current is one entry)
        count = 0
        while stack.undo() is not None:
            count += 1
        assert count == _MAX_UNDO_HISTORY - 1


class TestParseArgs:
    def test_data_package(self) -> None:
        args = parse_args(["--data_package", "/tmp/pkg"])
        assert str(args.data_package) == "/tmp/pkg"
        assert args.input_dir is None

    def test_input_dir(self) -> None:
        args = parse_args(["--input_dir", "/tmp/slices", "--transforms_dir", "/tmp/tfm"])
        assert str(args.input_dir) == "/tmp/slices"
        assert str(args.transforms_dir) == "/tmp/tfm"

    def test_defaults(self) -> None:
        args = parse_args([])
        # --level defaults to None as a sentinel so main() can tell "not set"
        # apart from an explicit "1" and adopt the data-package level instead.
        assert args.level is None
        assert args.slices is None
        assert args.server_config is None
        assert args.output_dir is None

    def test_level(self) -> None:
        args = parse_args(["--level", "3"])
        assert args.level == 3

    def test_slices_filter(self) -> None:
        args = parse_args(["--slices", "4", "5", "6"])
        assert args.slices == [4, 5, 6]

    def test_server_config(self) -> None:
        args = parse_args(["--server_config", "/tmp/nextflow.config"])
        assert str(args.server_config) == "/tmp/nextflow.config"


class TestOutputDirResolution:
    """Verify the server_package output_dir fix from main() inline."""

    def test_server_package_output_dir_is_subject_level(self, tmp_path: Path) -> None:
        """Package inside server_package/ → output_dir sits at the subject level."""
        pkg = tmp_path / "sub-22" / "server_package" / "manual_align_package"
        # Simulate the resolution logic from main()
        if pkg.parent.name == "server_package":
            output_dir = pkg.parent.parent / "manual_transforms"
        else:
            output_dir = pkg / "manual_transforms"
        assert output_dir == tmp_path / "sub-22" / "manual_transforms"

    def test_standalone_data_package_output_dir_is_nested(self, tmp_path: Path) -> None:
        """Package NOT inside server_package/ → output_dir inside the package dir."""
        pkg = tmp_path / "my_package"
        if pkg.parent.name == "server_package":
            output_dir = pkg.parent.parent / "manual_transforms"
        else:
            output_dir = pkg / "manual_transforms"
        assert output_dir == tmp_path / "my_package" / "manual_transforms"


class TestContractCliResolution:
    """Contract-backed CLI output-dir and package-level resolution (Plan 01-03)."""

    def test_resolve_output_dir_input_dir_only(self, tmp_path: Path) -> None:
        from linumpy_manual_align.__main__ import resolve_output_dir
        from linumpy_manual_align.contracts import MANUAL_TRANSFORMS_DIRNAME

        input_dir = tmp_path / "bring_to_common_space"
        input_dir.mkdir()
        result = resolve_output_dir(None, input_dir, None)
        assert result == input_dir.parent / MANUAL_TRANSFORMS_DIRNAME
        assert result.name == MANUAL_TRANSFORMS_DIRNAME

    def test_resolve_output_dir_standalone_package(self, tmp_path: Path) -> None:
        from linumpy_manual_align.__main__ import resolve_output_dir
        from linumpy_manual_align.contracts import MANUAL_TRANSFORMS_DIRNAME

        pkg = tmp_path / "manual_align_package"
        pkg.mkdir()
        result = resolve_output_dir(pkg, None, None)
        assert result == pkg / MANUAL_TRANSFORMS_DIRNAME
        assert result.name == MANUAL_TRANSFORMS_DIRNAME

    def test_resolve_output_dir_server_package_nested(self, tmp_path: Path) -> None:
        from linumpy_manual_align.__main__ import resolve_output_dir
        from linumpy_manual_align.contracts import MANUAL_TRANSFORMS_DIRNAME

        pkg = tmp_path / "sub-22" / "server_package" / "manual_align_package"
        pkg.mkdir(parents=True)
        result = resolve_output_dir(pkg, None, None)
        assert result == tmp_path / "sub-22" / MANUAL_TRANSFORMS_DIRNAME
        assert result.name == MANUAL_TRANSFORMS_DIRNAME

    def test_resolve_output_dir_server_config_only(self, tmp_path: Path) -> None:
        from linumpy_manual_align.__main__ import resolve_output_dir
        from linumpy_manual_align.contracts import MANUAL_TRANSFORMS_DIRNAME

        server_config = tmp_path / "sub-22" / "nextflow.config"
        server_config.parent.mkdir(parents=True)
        server_config.touch()
        result = resolve_output_dir(None, None, server_config)
        assert result == server_config.parent / MANUAL_TRANSFORMS_DIRNAME
        assert result.name == MANUAL_TRANSFORMS_DIRNAME

    def test_resolve_package_level_from_metadata(self, tmp_path: Path) -> None:
        import json

        from linumpy_manual_align.__main__ import resolve_package_level
        from linumpy_manual_align.contracts import METADATA_FILENAME

        pkg = tmp_path / "pkg"
        pkg.mkdir()
        (pkg / METADATA_FILENAME).write_text(json.dumps({"pyramid_level": 3}))
        assert resolve_package_level(pkg) == 3

    def test_resolve_package_level_none_without_metadata(self, tmp_path: Path) -> None:
        from linumpy_manual_align.__main__ import resolve_package_level

        pkg = tmp_path / "pkg"
        pkg.mkdir()
        assert resolve_package_level(pkg) is None


class TestSaveCurrentValidation:
    """Widget-level save validation wiring (Phase 03 plan 01)."""

    def test_save_current_marks_saved_on_valid_output(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        mid = 5
        widget = _make_save_widget(tmp_path, mid)
        save_calls: list[tuple] = []

        def fake_save(out_dir, tx, ty, rotation, *, center, level, offsets):
            save_calls.append((out_dir, tx, ty, rotation, center, level, offsets))

        monkeypatch.setattr("linumpy_manual_align.ui.widget_undo_save.save_transform", fake_save)
        monkeypatch.setattr(widget_undo_save_module, "validate_manual_output", lambda out_dir, m: [], raising=False)

        widget._save_current()

        assert mid in widget.saved_pairs
        assert mid not in widget.unsaved_changes
        assert widget._flash_saved_calls == [mid]
        assert len(save_calls) == 1
        assert save_calls[0][0].name == f"slice_z{mid:02d}"

    def test_save_current_failure_keeps_unsaved(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        mid = 5
        widget = _make_save_widget(tmp_path, mid)
        issue = ContractIssue(
            severity=SEVERITY_ERROR,
            code="output.missing_offsets",
            message="offsets.txt is missing",
        )

        monkeypatch.setattr("linumpy_manual_align.ui.widget_undo_save.save_transform", lambda *a, **k: None)
        monkeypatch.setattr(widget_undo_save_module, "validate_manual_output", lambda out_dir, m: [issue], raising=False)
        monkeypatch.setattr("linumpy_manual_align.ui.widget_undo_save.QMessageBox.critical", MagicMock())

        widget._save_current()

        assert mid not in widget.saved_pairs
        assert mid in widget.unsaved_changes
        assert widget._flash_saved_calls == []

    def test_save_current_failure_clears_stale_saved_pair(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        mid = 5
        widget = _make_save_widget(tmp_path, mid)
        widget.saved_pairs.add(mid)
        issue = ContractIssue(
            severity=SEVERITY_ERROR,
            code="output.missing_offsets",
            message="offsets.txt is missing",
        )

        monkeypatch.setattr("linumpy_manual_align.ui.widget_undo_save.save_transform", lambda *a, **k: None)
        monkeypatch.setattr(widget_undo_save_module, "validate_manual_output", lambda out_dir, m: [issue], raising=False)
        monkeypatch.setattr("linumpy_manual_align.ui.widget_undo_save.QMessageBox.critical", MagicMock())

        widget._save_current()

        assert mid not in widget.saved_pairs
        assert mid in widget.unsaved_changes
        assert widget._flash_saved_calls == []

    def test_save_current_failure_status_and_modal(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        mid = 5
        widget = _make_save_widget(tmp_path, mid)
        out_dir = tmp_path / f"slice_z{mid:02d}"
        offsets_path = out_dir / "offsets.txt"
        issue = ContractIssue(
            severity=SEVERITY_ERROR,
            code="output.missing_offsets",
            message="offsets.txt is missing",
            affected_path=offsets_path,
        )
        critical_calls: list[tuple] = []

        def fake_critical(parent, title, message):
            critical_calls.append((parent, title, message))

        monkeypatch.setattr("linumpy_manual_align.ui.widget_undo_save.save_transform", lambda *a, **k: None)
        monkeypatch.setattr(widget_undo_save_module, "validate_manual_output", lambda out_dir, m: [issue], raising=False)
        monkeypatch.setattr("linumpy_manual_align.ui.widget_undo_save.QMessageBox.critical", fake_critical)

        widget._save_current()

        assert len(critical_calls) == 1
        message = critical_calls[0][2]
        assert f"z{mid:02d}" in message
        assert "offsets.txt" in message
        assert "output.missing_offsets" in message
        assert "offsets.txt is missing" in message
        assert f"z{mid:02d}" in widget.viewer.status
        assert "offsets.txt" in widget.status_label.text
        assert "output.missing_offsets" in widget.status_label.text

    def test_save_current_metrics_failure_blocks_saved(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        mid = 5
        widget = _make_save_widget(tmp_path, mid)
        issue = ContractIssue(
            severity=SEVERITY_ERROR,
            code="metrics.invalid_source",
            message="source must be manual",
        )

        monkeypatch.setattr("linumpy_manual_align.ui.widget_undo_save.save_transform", lambda *a, **k: None)
        monkeypatch.setattr(widget_undo_save_module, "validate_manual_output", lambda out_dir, m: [issue], raising=False)
        monkeypatch.setattr("linumpy_manual_align.ui.widget_undo_save.QMessageBox.critical", MagicMock())

        widget._save_current()

        assert mid not in widget.saved_pairs
        assert mid in widget.unsaved_changes
        assert widget._flash_saved_calls == []

    def test_save_current_full_issue_list_surfaces_primary(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        mid = 5
        widget = _make_save_widget(tmp_path, mid)
        errors = [
            ContractIssue(severity=SEVERITY_ERROR, code="output.missing_offsets", message="first error"),
            ContractIssue(severity=SEVERITY_ERROR, code="metrics.invalid_source", message="second error"),
        ]
        critical_calls: list[tuple] = []

        def fake_critical(parent, title, message):
            critical_calls.append((parent, title, message))

        monkeypatch.setattr("linumpy_manual_align.ui.widget_undo_save.save_transform", lambda *a, **k: None)
        monkeypatch.setattr(widget_undo_save_module, "validate_manual_output", lambda out_dir, m: errors, raising=False)
        monkeypatch.setattr("linumpy_manual_align.ui.widget_undo_save.QMessageBox.critical", fake_critical)

        with caplog.at_level("WARNING"):
            widget._save_current()

        assert len(critical_calls) == 1
        message = critical_calls[0][2]
        assert "output.missing_offsets" in message
        assert "first error" in message
        assert "metrics.invalid_source" not in message
        assert "second error" not in message
        assert "Save validation failed for z05" in caplog.text

    def test_save_current_warning_only_does_not_block(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        mid = 5
        widget = _make_save_widget(tmp_path, mid)
        warning = ContractIssue(
            severity=SEVERITY_WARNING,
            code="metrics.low_overlap",
            message="overlap below threshold",
        )
        critical_mock = MagicMock()

        monkeypatch.setattr("linumpy_manual_align.ui.widget_undo_save.save_transform", lambda *a, **k: None)
        monkeypatch.setattr(widget_undo_save_module, "validate_manual_output", lambda out_dir, m: [warning], raising=False)
        monkeypatch.setattr("linumpy_manual_align.ui.widget_undo_save.QMessageBox.critical", critical_mock)

        widget._save_current()

        assert mid in widget.saved_pairs
        assert mid not in widget.unsaved_changes
        assert widget._flash_saved_calls == [mid]
        critical_mock.assert_not_called()
        assert f"z{mid:02d}" in widget.viewer.status
        assert "metrics.low_overlap" in widget.viewer.status
        assert "overlap below threshold" in widget.viewer.status
        assert f"z{mid:02d}" in widget.status_label.text
        assert "metrics.low_overlap" in widget.status_label.text
        assert "overlap below threshold" in widget.status_label.text

    def test_saved_metrics_remain_manual(self, tmp_path: Path) -> None:
        import json

        mid = 7
        out_dir = tmp_path / f"slice_z{mid:02d}"
        save_transform(out_dir, tx=1.0, ty=2.0, rotation_deg=0.0, center=(50.0, 50.0), level=1)

        issues = validate_manual_output(out_dir, mid)
        errors = [i for i in issues if i.severity == SEVERITY_ERROR]
        assert errors == []

        metrics = json.loads((out_dir / "pairwise_registration_metrics.json").read_text())
        assert metrics["source"] == "manual"
        assert metrics["overall_status"] == "ok"


class TestSaveAllValidation:
    """Batch Save All & Exit validation wiring (Phase 03 plan 02)."""

    def test_save_all_stops_on_first_failure(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        mids = [1, 2, 3]
        widget = _make_batch_widget(tmp_path, mids)
        issue = ContractIssue(
            severity=SEVERITY_ERROR,
            code="output.missing_offsets",
            message="offsets.txt is missing",
        )
        save_mids: list[int] = []

        def fake_save(out_dir, tx, ty, rotation, *, center, level, offsets):
            mid = int(out_dir.name.replace("slice_z", ""))
            save_mids.append(mid)

        monkeypatch.setattr("linumpy_manual_align.ui.widget_undo_save.save_transform", fake_save)
        monkeypatch.setattr(
            widget_undo_save_module,
            "validate_manual_output",
            _batch_validate_side_effect(2, issue),
            raising=False,
        )
        monkeypatch.setattr("linumpy_manual_align.ui.widget_undo_save.QMessageBox.critical", MagicMock())

        widget._save_all_and_exit(skip_confirm=True)

        assert 1 in widget.saved_pairs
        assert 2 not in widget.saved_pairs
        assert 3 not in widget.saved_pairs
        assert save_mids == [1, 2]
        assert 3 not in save_mids

    def test_save_all_reports_counts(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        fail_mid = 2
        widget = _make_batch_widget(tmp_path, [1, 2, 3])
        out_dir = tmp_path / f"slice_z{fail_mid:02d}"
        offsets_path = out_dir / "offsets.txt"
        issue = ContractIssue(
            severity=SEVERITY_ERROR,
            code="output.missing_offsets",
            message="offsets.txt is missing",
            affected_path=offsets_path,
        )

        monkeypatch.setattr("linumpy_manual_align.ui.widget_undo_save.save_transform", lambda *a, **k: None)
        monkeypatch.setattr(
            widget_undo_save_module,
            "validate_manual_output",
            _batch_validate_side_effect(fail_mid, issue),
            raising=False,
        )
        monkeypatch.setattr("linumpy_manual_align.ui.widget_undo_save.QMessageBox.critical", MagicMock())

        widget._save_all_and_exit(skip_confirm=True)

        status = widget.viewer.status
        assert "1" in status
        assert "fail" in status.lower()
        assert "z02" in status
        assert "output.missing_offsets" in status
        assert "offsets.txt" in status
        assert widget.status_label.text == status

    def test_save_all_failure_aborts_exit(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        fail_mid = 2
        widget = _make_batch_widget(tmp_path, [1, 2, 3])
        issue = ContractIssue(
            severity=SEVERITY_ERROR,
            code="output.missing_offsets",
            message="offsets.txt is missing",
        )

        monkeypatch.setattr("linumpy_manual_align.ui.widget_undo_save.save_transform", lambda *a, **k: None)
        monkeypatch.setattr(
            widget_undo_save_module,
            "validate_manual_output",
            _batch_validate_side_effect(fail_mid, issue),
            raising=False,
        )
        monkeypatch.setattr("linumpy_manual_align.ui.widget_undo_save.QMessageBox.critical", MagicMock())

        widget._save_all_and_exit(skip_confirm=True)

        assert not widget.viewer.closed
        assert widget._close_confirmed is False
        assert 2 in widget.unsaved_changes
        assert 3 in widget.unsaved_changes

    def test_failed_save_still_blocks_close_guard(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        fail_mid = 2
        widget = object.__new__(_SaveCloseWidget)
        mids = [1, 2, 3]
        widget.pairs = [(m - 1, m) for m in mids]
        widget.current_pair_idx = 0
        widget.output_dir = tmp_path
        widget.level = 1
        widget.pair_centers = dict.fromkeys(mids, (50.0, 50.0))
        widget._current_offsets = dict.fromkeys(mids, (0, 0))
        widget.saved_pairs = set()
        widget.unsaved_changes = set(mids)
        widget.undo_stacks = {
            m: UndoStack(AlignmentState(tx=1.0, ty=2.0, rotation=0.0)) for m in mids
        }
        widget.viewer = _Viewer()
        widget.status_label = _StatusLabel()
        widget._close_confirmed = False
        widget._flash_saved_calls = []
        widget._flash_saved = lambda m: widget._flash_saved_calls.append(m)
        widget._current_state = lambda: AlignmentState(tx=1.0, ty=2.0, rotation=0.0)
        issue = ContractIssue(
            severity=SEVERITY_ERROR,
            code="output.missing_offsets",
            message="offsets.txt is missing",
        )

        monkeypatch.setattr("linumpy_manual_align.ui.widget_undo_save.save_transform", lambda *a, **k: None)
        monkeypatch.setattr(
            widget_undo_save_module,
            "validate_manual_output",
            _batch_validate_side_effect(fail_mid, issue),
            raising=False,
        )
        monkeypatch.setattr("linumpy_manual_align.ui.widget_undo_save.QMessageBox.critical", MagicMock())

        widget._save_all_and_exit(skip_confirm=True)

        cancel_btn = object()

        class FakeMsgBox:
            Warning = object()
            AcceptRole = object()
            DestructiveRole = object()
            RejectRole = object()

            def __init__(self, parent) -> None:
                self._parent = parent

            def setWindowTitle(self, title: str) -> None:
                pass

            def setText(self, text: str) -> None:
                pass

            def setInformativeText(self, text: str) -> None:
                pass

            def setIcon(self, icon) -> None:
                pass

            def addButton(self, text: str, role) -> object:
                return cancel_btn

            def setDefaultButton(self, btn) -> None:
                pass

            def exec_(self) -> None:
                pass

            def clickedButton(self) -> object:
                return cancel_btn

        monkeypatch.setattr(widget_close_guard_module, "QMessageBox", FakeMsgBox)

        assert widget._confirm_close() is False

    def test_save_current_and_save_all_use_shared_helper(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        widget = _make_batch_widget(tmp_path, [1, 2, 3])
        widget.current_pair_idx = 0
        calls: list[tuple[int, AlignmentState]] = []

        def record(mid: int, state: AlignmentState) -> list[ContractIssue]:
            calls.append((mid, state))
            return []

        widget._save_and_validate_pair = record  # type: ignore[method-assign]

        widget._save_current()
        widget._save_all_and_exit(skip_confirm=True)

        called_mids = [mid for mid, _ in calls]
        assert 1 in called_mids
        assert 2 in called_mids
        assert 3 in called_mids
        assert len(calls) >= 3


class _UploadServerConfig:
    def __init__(self, host: str, subject_id: str, remote_output: str = "") -> None:
        self.host = host
        self.subject_id = subject_id
        self.remote_output = remote_output


class _ButtonStub:
    def __init__(self) -> None:
        self.enabled: bool | None = None

    def setEnabled(self, enabled: bool) -> None:
        self.enabled = enabled


class _BtnModeZ:
    def __init__(self) -> None:
        self.enabled = False
        self.tooltip = ""

    def setEnabled(self, enabled: bool) -> None:
        self.enabled = enabled

    def setToolTip(self, tooltip: str) -> None:
        self.tooltip = tooltip


class _ProgressStub:
    def __init__(self) -> None:
        self.shown = False
        self.hidden = False

    def show(self) -> None:
        self.shown = True

    def hide(self) -> None:
        self.hidden = True


class _SignalStub:
    def connect(self, _callback) -> None:
        pass


class _ScpWorkerRecorder:
    instances: list[_ScpWorkerRecorder] = []

    def __init__(self, func, args) -> None:
        self.func = func
        self.args = args
        self.transfer_done = _SignalStub()
        self.finished = _SignalStub()
        type(self).instances.append(self)

    def start(self) -> None:
        pass

    def deleteLater(self) -> None:
        pass


def _make_upload_widget(
    tmp_path: Path,
    pairs: list[tuple[int, int]],
    saved_pairs: set[int],
    *,
    server_config: _UploadServerConfig | None = _UploadServerConfig(
        host="myserver.example.com",
        subject_id="sub-22",
        remote_output="/scratch/workspace/sub-22/output",
    ),
) -> ServerMixin:
    widget = object.__new__(ServerMixin)
    widget.server_config = server_config
    widget.output_dir = tmp_path
    widget.pairs = pairs
    widget.saved_pairs = saved_pairs
    widget.viewer = _Viewer()
    widget.server_status_label = _StatusLabel()
    widget.btn_download = _ButtonStub()
    widget.btn_upload = _ButtonStub()
    widget.server_progress = _ProgressStub()
    widget._worker = None
    return widget


def _make_readiness_report(
    *,
    ready_count: int = 0,
    missing_count: int = 0,
    invalid_count: int = 0,
    warning_count: int = 0,
    pairs: tuple[PairUploadReadiness, ...] = (),
    ready_dirs: tuple[Path, ...] = (),
) -> UploadReadinessReport:
    return UploadReadinessReport(
        pairs=pairs,
        ready_count=ready_count,
        missing_count=missing_count,
        invalid_count=invalid_count,
        warning_count=warning_count,
        ready_dirs=ready_dirs,
    )


def _patch_upload_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        widget_server_module.settings,
        "get",
        lambda key: "/scratch" if key == "server/remote_workspace_base" else "",
    )


class TestUploadGate:
    def test_upload_blocked_shows_critical(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        issue1 = ContractIssue(
            severity=SEVERITY_ERROR,
            code="output.missing_offsets",
            message="offsets.txt not found",
        )
        issue2 = ContractIssue(
            severity=SEVERITY_ERROR,
            code="output.missing_metrics",
            message="metrics not found",
        )
        out_dir = tmp_path / "slice_z01"
        pair = PairUploadReadiness(
            moving_id=1,
            fixed_id=0,
            status=PairUploadStatus.INVALID,
            output_dir=out_dir,
            issues=(issue1, issue2),
        )
        report = _make_readiness_report(
            ready_count=0,
            invalid_count=1,
            pairs=(pair,),
        )
        widget = _make_upload_widget(tmp_path, [(0, 1)], {1})
        _patch_upload_settings(monkeypatch)
        monkeypatch.setattr(
            widget_server_module,
            "assess_upload_readiness",
            lambda *a, **k: report,
            raising=False,
        )
        _ScpWorkerRecorder.instances = []
        monkeypatch.setattr(widget_server_module, "ScpWorker", _ScpWorkerRecorder)

        critical_calls: list[tuple] = []

        def record_critical(parent, title, body) -> None:
            critical_calls.append((parent, title, body))

        monkeypatch.setattr(
            widget_server_module,
            "QMessageBox",
            type("QMessageBox", (), {"critical": staticmethod(record_critical)}),
            raising=False,
        )

        widget._upload_to_server()

        assert len(critical_calls) == 1
        assert critical_calls[0][1] == "Upload blocked"
        body = critical_calls[0][2]
        assert report.summary_line() in body
        assert issue1.code in body or "offsets.txt not found" in body
        for line in report.error_lines():
            assert line in body
        assert widget.server_status_label.text.startswith("Upload blocked:")
        assert len(_ScpWorkerRecorder.instances) == 0

    def test_upload_blocked_zero_ready_message(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        issue = ContractIssue(
            severity=SEVERITY_ERROR,
            code="upload.not_saved",
            message="Pair not saved in this session",
        )
        pair = PairUploadReadiness(
            moving_id=1,
            fixed_id=0,
            status=PairUploadStatus.MISSING,
            output_dir=None,
            issues=(issue,),
        )
        report = _make_readiness_report(
            ready_count=0,
            missing_count=1,
            pairs=(pair,),
        )
        widget = _make_upload_widget(tmp_path, [(0, 1)], set())
        _patch_upload_settings(monkeypatch)
        monkeypatch.setattr(
            widget_server_module,
            "assess_upload_readiness",
            lambda *a, **k: report,
            raising=False,
        )

        critical_calls: list[str] = []
        monkeypatch.setattr(
            widget_server_module,
            "QMessageBox",
            type("QMessageBox", (), {"critical": staticmethod(lambda _p, _t, body: critical_calls.append(body))}),
            raising=False,
        )
        monkeypatch.setattr(widget_server_module, "ScpWorker", _ScpWorkerRecorder)

        widget._upload_to_server()

        assert len(critical_calls) == 1
        assert "No pairs ready to upload" in critical_calls[0]

    def test_confirm_shows_destination(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        ready_dir1 = tmp_path / "slice_z01"
        ready_dir2 = tmp_path / "slice_z02"
        ready_dir1.mkdir()
        ready_dir2.mkdir()
        pair1 = PairUploadReadiness(
            moving_id=1,
            fixed_id=0,
            status=PairUploadStatus.READY,
            output_dir=ready_dir1,
            issues=(),
        )
        pair2 = PairUploadReadiness(
            moving_id=2,
            fixed_id=1,
            status=PairUploadStatus.READY,
            output_dir=ready_dir2,
            issues=(),
        )
        report = _make_readiness_report(
            ready_count=2,
            pairs=(pair1, pair2),
            ready_dirs=(ready_dir1, ready_dir2),
        )
        widget = _make_upload_widget(tmp_path, [(0, 1), (1, 2)], {1, 2})
        _patch_upload_settings(monkeypatch)
        monkeypatch.setattr(
            widget_server_module,
            "assess_upload_readiness",
            lambda *a, **k: report,
            raising=False,
        )
        _ScpWorkerRecorder.instances = []
        monkeypatch.setattr(widget_server_module, "ScpWorker", _ScpWorkerRecorder)

        ok_sentinel = 1
        captured_dialogs: list[str] = []

        class FakeMsgBox:
            Question = 0
            Ok = 1
            Cancel = 2

            def __init__(self, parent) -> None:
                self._parent = parent
                self.text = ""

            def setWindowTitle(self, title: str) -> None:
                self.title = title

            def setText(self, text: str) -> None:
                self.text = text
                captured_dialogs.append(text)

            def setIcon(self, icon) -> None:
                pass

            def setStandardButtons(self, buttons) -> None:
                pass

            def setDefaultButton(self, button) -> None:
                pass

            def button(self, role):
                btn = MagicMock()
                btn.setText = MagicMock()
                return btn

            def exec(self):
                return ok_sentinel

        monkeypatch.setattr(widget_server_module, "QMessageBox", FakeMsgBox, raising=False)

        widget._upload_to_server()

        expected_dest = (
            "myserver.example.com:/scratch/workspace/sub-22/output/manual_transforms/"
        )
        assert len(captured_dialogs) == 1
        assert expected_dest in captured_dialogs[0]
        assert "Upload 2 pair(s)" in captured_dialogs[0]
        assert len(_ScpWorkerRecorder.instances) == 1
        worker_args = _ScpWorkerRecorder.instances[0].args
        assert worker_args[2] == list(report.ready_dirs)
        assert widget.btn_download.enabled is False
        assert widget.btn_upload.enabled is False
        assert widget.server_progress.shown is True
        assert widget.server_status_label.text == "<i>Uploading...</i>"

    def test_confirm_cancel_is_silent(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        ready_dir = tmp_path / "slice_z01"
        ready_dir.mkdir()
        pair = PairUploadReadiness(
            moving_id=1,
            fixed_id=0,
            status=PairUploadStatus.READY,
            output_dir=ready_dir,
            issues=(),
        )
        report = _make_readiness_report(
            ready_count=1,
            pairs=(pair,),
            ready_dirs=(ready_dir,),
        )
        widget = _make_upload_widget(tmp_path, [(0, 1)], {1})
        _patch_upload_settings(monkeypatch)
        monkeypatch.setattr(
            widget_server_module,
            "assess_upload_readiness",
            lambda *a, **k: report,
            raising=False,
        )
        _ScpWorkerRecorder.instances = []
        monkeypatch.setattr(widget_server_module, "ScpWorker", _ScpWorkerRecorder)

        cancel_sentinel = 2

        class FakeMsgBox:
            Question = 0
            Ok = 1
            Cancel = 2

            def __init__(self, parent) -> None:
                self.text = ""

            def setWindowTitle(self, title: str) -> None:
                pass

            def setText(self, text: str) -> None:
                self.text = text

            def setIcon(self, icon) -> None:
                pass

            def setStandardButtons(self, buttons) -> None:
                pass

            def setDefaultButton(self, button) -> None:
                pass

            def button(self, role):
                btn = MagicMock()
                btn.setText = MagicMock()
                return btn

            def exec(self):
                return cancel_sentinel

        monkeypatch.setattr(widget_server_module, "QMessageBox", FakeMsgBox, raising=False)

        widget._upload_to_server()

        assert len(_ScpWorkerRecorder.instances) == 0
        assert widget.server_status_label.text == report.summary_line()
        assert "cancel" not in widget.server_status_label.text.lower()

    def test_confirm_lists_warnings(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        ready_dir = tmp_path / "slice_z01"
        ready_dir.mkdir()
        warning = ContractIssue(
            severity=SEVERITY_WARNING,
            code="metadata.level_mismatch",
            message="Pyramid level differs from package",
        )
        pair = PairUploadReadiness(
            moving_id=1,
            fixed_id=0,
            status=PairUploadStatus.READY,
            output_dir=ready_dir,
            issues=(warning,),
        )
        report = _make_readiness_report(
            ready_count=1,
            warning_count=1,
            pairs=(pair,),
            ready_dirs=(ready_dir,),
        )
        widget = _make_upload_widget(tmp_path, [(0, 1)], {1})
        _patch_upload_settings(monkeypatch)
        monkeypatch.setattr(
            widget_server_module,
            "assess_upload_readiness",
            lambda *a, **k: report,
            raising=False,
        )

        captured: list[str] = []

        class FakeMsgBox:
            Question = 0
            Ok = 1
            Cancel = 2

            def __init__(self, parent) -> None:
                self.text = ""

            def setWindowTitle(self, title: str) -> None:
                pass

            def setText(self, text: str) -> None:
                self.text = text
                captured.append(text)

            def setIcon(self, icon) -> None:
                pass

            def setStandardButtons(self, buttons) -> None:
                pass

            def setDefaultButton(self, button) -> None:
                pass

            def button(self, role):
                btn = MagicMock()
                btn.setText = MagicMock()
                return btn

            def exec(self):
                return FakeMsgBox.Cancel

        monkeypatch.setattr(widget_server_module, "QMessageBox", FakeMsgBox, raising=False)
        monkeypatch.setattr(widget_server_module, "ScpWorker", _ScpWorkerRecorder)

        widget._upload_to_server()

        assert len(captured) == 1
        assert "Warnings:" in captured[0]
        assert report.warning_lines()[0] in captured[0]

    def test_local_only_noop(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        widget = _make_upload_widget(tmp_path, [(0, 1)], {1}, server_config=None)
        initial_status = widget.server_status_label.text
        _ScpWorkerRecorder.instances = []
        monkeypatch.setattr(widget_server_module, "ScpWorker", _ScpWorkerRecorder)

        widget._upload_to_server()

        assert len(_ScpWorkerRecorder.instances) == 0
        assert widget.server_status_label.text == initial_status

    def test_revalidates_on_each_click(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        ready_dir = tmp_path / "slice_z01"
        ready_dir.mkdir()
        pair = PairUploadReadiness(
            moving_id=1,
            fixed_id=0,
            status=PairUploadStatus.READY,
            output_dir=ready_dir,
            issues=(),
        )
        report = _make_readiness_report(
            ready_count=1,
            pairs=(pair,),
            ready_dirs=(ready_dir,),
        )
        widget = _make_upload_widget(tmp_path, [(0, 1)], {1})
        _patch_upload_settings(monkeypatch)
        call_count = 0

        def count_assess(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            return report

        monkeypatch.setattr(
            widget_server_module,
            "assess_upload_readiness",
            count_assess,
            raising=False,
        )
        monkeypatch.setattr(widget_server_module, "ScpWorker", _ScpWorkerRecorder)

        class FakeMsgBox:
            Question = 0
            Ok = 1
            Cancel = 2

            def __init__(self, parent) -> None:
                pass

            def setWindowTitle(self, title: str) -> None:
                pass

            def setText(self, text: str) -> None:
                pass

            def setIcon(self, icon) -> None:
                pass

            def setStandardButtons(self, buttons) -> None:
                pass

            def setDefaultButton(self, button) -> None:
                pass

            def button(self, role):
                btn = MagicMock()
                btn.setText = MagicMock()
                return btn

            def exec(self):
                return FakeMsgBox.Cancel

        monkeypatch.setattr(widget_server_module, "QMessageBox", FakeMsgBox, raising=False)

        widget._upload_to_server()
        widget._upload_to_server()

        assert call_count == 2

    def test_blocked_status_persists_after_dialog(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        issue = ContractIssue(
            severity=SEVERITY_ERROR,
            code="upload.not_saved",
            message="Pair not saved in this session",
        )
        pair = PairUploadReadiness(
            moving_id=1,
            fixed_id=0,
            status=PairUploadStatus.MISSING,
            output_dir=None,
            issues=(issue,),
        )
        report = _make_readiness_report(
            ready_count=0,
            missing_count=1,
            pairs=(pair,),
        )
        widget = _make_upload_widget(tmp_path, [(0, 1)], set())
        _patch_upload_settings(monkeypatch)
        monkeypatch.setattr(
            widget_server_module,
            "assess_upload_readiness",
            lambda *a, **k: report,
            raising=False,
        )
        _ScpWorkerRecorder.instances = []
        monkeypatch.setattr(widget_server_module, "ScpWorker", _ScpWorkerRecorder)
        monkeypatch.setattr(
            widget_server_module,
            "QMessageBox",
            type("QMessageBox", (), {"critical": staticmethod(lambda *_a: None)}),
            raising=False,
        )

        widget._upload_to_server()

        assert widget.server_status_label.text.startswith("Upload blocked:")
        assert len(_ScpWorkerRecorder.instances) == 0

    def test_on_upload_finished_restores_buttons(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        widget = _make_upload_widget(tmp_path, [(0, 1)], {1})
        widget.btn_download.enabled = False
        widget.btn_upload.enabled = False
        widget.server_progress.shown = True

        widget._on_upload_finished(True, "Uploaded 1 transforms to server")

        assert widget.btn_download.enabled is True
        assert widget.btn_upload.enabled is True
        assert widget.server_progress.hidden is True
        assert widget.server_status_label.text == "Uploaded 1 transforms to server"


class _ComboStub:
    def __init__(self, items: list[str] | None = None) -> None:
        self.items: list[str] = list(items or [])
        self._index = 0
        self._signals_blocked = False

    def count(self) -> int:
        return len(self.items)

    def itemText(self, i: int) -> str:
        return self.items[i]

    def setItemText(self, i: int, text: str) -> None:
        self.items[i] = text

    def currentIndex(self) -> int:
        return self._index

    def setCurrentIndex(self, i: int) -> None:
        self._index = i

    def blockSignals(self, flag: bool) -> bool:
        prev = self._signals_blocked
        self._signals_blocked = flag
        return prev


class _ResumeBlockStub:
    def __init__(self) -> None:
        self._visible = False

    def hide(self) -> None:
        self._visible = False

    def show(self) -> None:
        self._visible = True

    def isVisible(self) -> bool:
        return self._visible


def _make_session_widget(
    tmp_path: Path,
    pairs: list[tuple[int, int]],
    *,
    saved_pairs: set[int] | None = None,
    unsaved_changes: set[int] | None = None,
    uploaded_pairs: set[int] | None = None,
    server_config: _UploadServerConfig | None = None,
):
    from linumpy_manual_align.ui.widget_session import SessionMixin

    class _SessionWidget(SessionMixin, StatusMixin, UndoSaveMixin, PairNavigationMixin):
        pass

    widget = object.__new__(_SessionWidget)
    widget.pairs = pairs
    widget.current_pair_idx = 0
    widget.output_dir = tmp_path
    widget.saved_pairs = set(saved_pairs or set())
    widget.unsaved_changes = set(unsaved_changes or set())
    widget.uploaded_pairs = set(uploaded_pairs or set())
    widget.server_config = server_config
    widget._session_states = {}
    widget._resume_config_line = ""
    widget.session_summary_label = _StatusLabel()
    widget.pair_combo = _ComboStub(
        [f"z{fid:02d} → z{mid:02d}" for fid, mid in pairs]
    )
    widget.status_label = _StatusLabel()
    widget.viewer = _Viewer()
    widget.existing_transforms = {}
    widget.level = 1
    widget._current_offsets = {mid: (0, 0) for _fid, mid in pairs}
    widget._projection_mode = "xy"
    widget._saved_flash_mid = None
    widget._saved_flash_timer = MagicMock()
    widget._cs_mgr = MagicMock(interpolated_slice_ids=set())
    widget.resume_block = _ResumeBlockStub()
    widget._current_state = lambda: AlignmentState(tx=0.0, ty=0.0, rotation=0.0)
    return widget


def _copy_valid_slice_output(copy_fixture_tree, tmp_path: Path, mid: int) -> Path:
    golden_root = tmp_path / "_golden_manual_transforms"
    if not golden_root.exists():
        copy_fixture_tree("manual_transforms", golden_root)
    dest = tmp_path / f"slice_z{mid:02d}"
    if dest.exists():
        shutil.rmtree(dest)
    shutil.copytree(golden_root / f"slice_z{mid:02d}", dest)
    return dest


class TestSessionGroupBuild:
    def test_build_session_group_returns_widgets(self, qapp) -> None:
        from linumpy_manual_align.ui.ui_builder import build_session_group

        group, ns = build_session_group(
            on_copy_config=lambda: None,
            on_dismiss_resume=lambda: None,
        )

        assert group.title() == "Session"
        assert ns.session_summary_label is not None
        assert ns.resume_block is not None
        assert ns.resume_config_label is not None
        assert ns.resume_guidance_label is not None
        assert ns.btn_copy_config_line is not None
        assert ns.btn_dismiss_resume is not None
        assert ns.resume_block.isHidden()

    def test_local_only_saved_valid_summary_and_prefix(
        self, tmp_path: Path, copy_fixture_tree
    ) -> None:
        _copy_valid_slice_output(copy_fixture_tree, tmp_path, mid=1)
        widget = _make_session_widget(
            tmp_path,
            [(0, 1)],
            saved_pairs={1},
            server_config=None,
        )

        widget._refresh_session_state()

        summary = widget.session_summary_label.text
        assert "saved-local" in summary
        assert "ready" not in summary
        assert "uploaded" not in summary
        assert widget.pair_combo.itemText(0).startswith("✓ ")

    def test_on_disk_invalid_summary_and_prefix(self, tmp_path: Path) -> None:
        (tmp_path / "slice_z01").mkdir()
        widget = _make_session_widget(tmp_path, [(0, 1)], saved_pairs={1})

        widget._refresh_session_state()

        summary = widget.session_summary_label.text
        assert "invalid" in summary
        assert int(summary.split()[0]) >= 1
        assert widget.pair_combo.itemText(0).startswith("✗ ")

    def test_invalid_over_unsaved_priority(self, tmp_path: Path) -> None:
        (tmp_path / "slice_z01").mkdir()
        widget = _make_session_widget(
            tmp_path,
            [(0, 1)],
            saved_pairs={1},
            unsaved_changes={1},
        )

        widget._refresh_session_state()

        assert widget.pair_combo.itemText(0).startswith("✗ ")
        assert not widget.pair_combo.itemText(0).startswith("● ")

    def test_server_mode_ready_prefix_and_summary(
        self, tmp_path: Path, copy_fixture_tree, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        ready_dir = _copy_valid_slice_output(copy_fixture_tree, tmp_path, mid=1)
        pair = PairUploadReadiness(
            moving_id=1,
            fixed_id=0,
            status=PairUploadStatus.READY,
            output_dir=ready_dir,
            issues=(),
        )
        report = _make_readiness_report(
            ready_count=1,
            pairs=(pair,),
            ready_dirs=(ready_dir,),
        )
        import linumpy_manual_align.ui.widget_session as widget_session_module

        monkeypatch.setattr(
            widget_session_module,
            "assess_upload_readiness",
            lambda *a, **k: report,
            raising=False,
        )
        widget = _make_session_widget(
            tmp_path,
            [(0, 1)],
            saved_pairs={1},
            server_config=_UploadServerConfig(
                host="myserver.example.com",
                subject_id="sub-22",
            ),
        )

        widget._refresh_session_state()

        assert widget.pair_combo.itemText(0).startswith("◎ ")
        assert "1 ready" in widget.session_summary_label.text

    def test_local_only_contrast_no_ready_prefix(
        self, tmp_path: Path, copy_fixture_tree
    ) -> None:
        _copy_valid_slice_output(copy_fixture_tree, tmp_path, mid=1)
        widget = _make_session_widget(
            tmp_path,
            [(0, 1)],
            saved_pairs={1},
            server_config=None,
        )

        widget._refresh_session_state()

        assert widget.pair_combo.itemText(0).startswith("✓ ")
        summary = widget.session_summary_label.text
        assert "ready" not in summary
        assert "uploaded" not in summary


class _ClipboardStub:
    def __init__(self) -> None:
        self.text = ""

    def setText(self, text: str) -> None:
        self.text = text


def _make_session_confidence_widget(
    tmp_path: Path,
    pairs: list[tuple[int, int]],
    *,
    saved_pairs: set[int] | None = None,
    unsaved_changes: set[int] | None = None,
    uploaded_pairs: set[int] | None = None,
    server_config: _UploadServerConfig | None = _UploadServerConfig(
        host="myserver.example.com",
        subject_id="sub-22",
        remote_output="/scratch/workspace/sub-22/output",
    ),
):
    from linumpy_manual_align.ui.widget_pair_loading import PairLoadingMixin
    from linumpy_manual_align.ui.widget_session import SessionMixin

    class _SessionConfidenceWidget(
        SessionMixin,
        ServerMixin,
        PairLoadingMixin,
        StatusMixin,
        UndoSaveMixin,
        PairNavigationMixin,
    ):
        pass

    widget = object.__new__(_SessionConfidenceWidget)
    widget.pairs = pairs
    widget.current_pair_idx = 0
    widget.output_dir = tmp_path
    widget.saved_pairs = set(saved_pairs or set())
    widget.unsaved_changes = set(unsaved_changes or set())
    widget.uploaded_pairs = set(uploaded_pairs or set())
    widget.server_config = server_config
    widget._session_states = {}
    widget._resume_config_line = ""
    widget._pending_upload_mids = None
    widget.session_summary_label = _StatusLabel()
    widget.resume_config_label = _StatusLabel()
    widget.resume_guidance_label = _StatusLabel()
    widget.pair_combo = _ComboStub(
        [f"z{fid:02d} → z{mid:02d}" for fid, mid in pairs]
    )
    widget.status_label = _StatusLabel()
    widget.server_status_label = _StatusLabel()
    widget.viewer = _Viewer()
    widget.btn_download = _ButtonStub()
    widget.btn_upload = _ButtonStub()
    widget.server_progress = _ProgressStub()
    widget._worker = None
    widget.existing_transforms = {}
    widget.level = 1
    widget._current_offsets = {mid: (0, 0) for _fid, mid in pairs}
    widget._projection_mode = "xy"
    widget._saved_flash_mid = None
    widget._saved_flash_timer = MagicMock()
    widget._cs_mgr = MagicMock(interpolated_slice_ids=set())
    widget._btn_mode_z = _BtnModeZ()
    widget.resume_block = _ResumeBlockStub()
    widget._current_state = lambda: AlignmentState(tx=0.0, ty=0.0, rotation=0.0)
    return widget


def _finish_upload(widget, *, mids: set[int], msg: str = "Uploaded 2 transforms to host:/path") -> None:
    widget._pending_upload_mids = set(mids)
    widget._on_upload_finished(True, msg)


class TestSessionConfidence:
    def test_uploaded_lifecycle_sets_pairs_and_prefixes(
        self, tmp_path: Path, copy_fixture_tree, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _copy_valid_slice_output(copy_fixture_tree, tmp_path, mid=1)
        _copy_valid_slice_output(copy_fixture_tree, tmp_path, mid=2)
        ready_dir1 = tmp_path / "slice_z01"
        ready_dir2 = tmp_path / "slice_z02"
        pair1 = PairUploadReadiness(
            moving_id=1,
            fixed_id=0,
            status=PairUploadStatus.READY,
            output_dir=ready_dir1,
            issues=(),
        )
        pair2 = PairUploadReadiness(
            moving_id=2,
            fixed_id=1,
            status=PairUploadStatus.READY,
            output_dir=ready_dir2,
            issues=(),
        )
        report = _make_readiness_report(
            ready_count=2,
            pairs=(pair1, pair2),
            ready_dirs=(ready_dir1, ready_dir2),
        )
        import linumpy_manual_align.ui.widget_session as widget_session_module

        monkeypatch.setattr(
            widget_session_module,
            "assess_upload_readiness",
            lambda *a, **k: report,
            raising=False,
        )
        widget = _make_session_confidence_widget(
            tmp_path,
            [(0, 1), (1, 2)],
            saved_pairs={1, 2},
        )
        _patch_upload_settings(monkeypatch)

        _finish_upload(widget, mids={1, 2})

        assert widget.uploaded_pairs == {1, 2}
        assert widget.pair_combo.itemText(0).startswith("↑ ")
        assert widget.pair_combo.itemText(1).startswith("↑ ")
        assert "2 uploaded" in widget.session_summary_label.text

    def test_reupload_replaces_uploaded_batch(
        self, tmp_path: Path, copy_fixture_tree, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _copy_valid_slice_output(copy_fixture_tree, tmp_path, mid=1)
        _copy_valid_slice_output(copy_fixture_tree, tmp_path, mid=2)
        ready_dir2 = tmp_path / "slice_z02"
        pair2 = PairUploadReadiness(
            moving_id=2,
            fixed_id=1,
            status=PairUploadStatus.READY,
            output_dir=ready_dir2,
            issues=(),
        )
        report = _make_readiness_report(
            ready_count=1,
            pairs=(pair2,),
            ready_dirs=(ready_dir2,),
        )
        import linumpy_manual_align.ui.widget_session as widget_session_module

        monkeypatch.setattr(
            widget_session_module,
            "assess_upload_readiness",
            lambda *a, **k: report,
            raising=False,
        )
        widget = _make_session_confidence_widget(
            tmp_path,
            [(0, 1), (1, 2)],
            saved_pairs={1, 2},
        )
        _patch_upload_settings(monkeypatch)

        _finish_upload(widget, mids={1, 2}, msg="Uploaded 2 transforms to host:/path")
        _finish_upload(widget, mids={2}, msg="Uploaded 1 transforms to host:/path")

        assert widget.uploaded_pairs == {2}
        assert not widget.pair_combo.itemText(0).startswith("↑ ")
        assert widget.pair_combo.itemText(1).startswith("↑ ")

    def test_edit_clears_uploaded_marker(
        self, tmp_path: Path, copy_fixture_tree, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _copy_valid_slice_output(copy_fixture_tree, tmp_path, mid=1)
        ready_dir = tmp_path / "slice_z01"
        pair = PairUploadReadiness(
            moving_id=1,
            fixed_id=0,
            status=PairUploadStatus.READY,
            output_dir=ready_dir,
            issues=(),
        )
        report = _make_readiness_report(
            ready_count=1,
            pairs=(pair,),
            ready_dirs=(ready_dir,),
        )
        import linumpy_manual_align.ui.widget_session as widget_session_module

        monkeypatch.setattr(
            widget_session_module,
            "assess_upload_readiness",
            lambda *a, **k: report,
            raising=False,
        )
        widget = _make_session_confidence_widget(
            tmp_path,
            [(0, 1)],
            saved_pairs={1},
            uploaded_pairs={1},
        )
        widget._refresh_session_state()
        assert widget.pair_combo.itemText(0).startswith("↑ ")

        widget._mark_pair_edited(1)

        assert 1 not in widget.uploaded_pairs
        assert not widget.pair_combo.itemText(0).startswith("↑ ")

    def test_resume_block_shows_full_path(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        widget = _make_session_confidence_widget(tmp_path, [(0, 1)], saved_pairs={1})
        _patch_upload_settings(monkeypatch)

        _finish_upload(widget, mids={1})

        config_text = widget.resume_config_label.text
        assert "params.manual_transforms_dir = '" in config_text
        assert "/scratch/workspace/sub-22/output/manual_transforms" in config_text
        assert "host:" not in config_text
        assert widget.resume_block.isVisible()

    def test_copy_config_line(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        widget = _make_session_confidence_widget(tmp_path, [(0, 1)], saved_pairs={1})
        _patch_upload_settings(monkeypatch)
        _finish_upload(widget, mids={1})

        clipboard = _ClipboardStub()
        import linumpy_manual_align.ui.widget_session as widget_session_module

        monkeypatch.setattr(
            widget_session_module.QApplication,
            "clipboard",
            staticmethod(lambda: clipboard),
        )

        widget._copy_config_line()

        assert clipboard.text == widget.resume_config_label.text
        assert widget.viewer.status == "Copied config line to clipboard"

    def test_resume_guidance_mentions_stack_and_resume(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        widget = _make_session_confidence_widget(tmp_path, [(0, 1)], saved_pairs={1})
        _patch_upload_settings(monkeypatch)
        _finish_upload(widget, mids={1})

        guidance = widget.resume_guidance_label.text
        assert "stack" in guidance
        assert "-resume" in guidance

    def test_local_only_hides_resume_and_upload_counts(
        self, tmp_path: Path, copy_fixture_tree
    ) -> None:
        _copy_valid_slice_output(copy_fixture_tree, tmp_path, mid=1)
        widget = _make_session_confidence_widget(
            tmp_path,
            [(0, 1)],
            saved_pairs={1},
            server_config=None,
        )

        widget._refresh_session_state()

        assert not widget.resume_block.isVisible()
        summary = widget.session_summary_label.text
        assert "ready" not in summary
        assert "uploaded" not in summary

    def test_download_finish_refreshes_session(
        self, tmp_path: Path, copy_fixture_tree, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _copy_valid_slice_output(copy_fixture_tree, tmp_path, mid=1)
        widget = _make_session_confidence_widget(
            tmp_path,
            [(0, 1)],
            saved_pairs=set(),
            server_config=None,
        )
        widget._refresh_session_state()
        assert "0 saved-local" in widget.session_summary_label.text

        local_dir = tmp_path / "server_package"
        aips = local_dir / "manual_align_package" / "aips"
        aips.mkdir(parents=True)
        (aips / "slice_z00.npz").touch()

        monkeypatch.setattr(widget, "_rebuild_pairs", lambda: None)

        widget._on_download_finished(True, "Download complete", local_dir)

        assert "1 saved-local" in widget.session_summary_label.text

    def test_load_existing_package_refreshes_session(
        self, tmp_path: Path, copy_fixture_tree, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _copy_valid_slice_output(copy_fixture_tree, tmp_path, mid=1)
        widget = _make_session_confidence_widget(
            tmp_path,
            [(0, 1)],
            saved_pairs=set(),
            server_config=None,
        )
        widget._refresh_session_state()
        assert "0 saved-local" in widget.session_summary_label.text

        aips = tmp_path / "manual_align_package" / "aips"
        aips.mkdir(parents=True)
        (aips / "slice_z00.npz").touch()

        monkeypatch.setattr(widget, "_rebuild_pairs", lambda: None)

        widget._load_existing_package(aips)

        assert "1 saved-local" in widget.session_summary_label.text

    def test_server_status_unchanged_after_upload(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        widget = _make_session_confidence_widget(tmp_path, [(0, 1)], saved_pairs={1})
        _patch_upload_settings(monkeypatch)
        msg = "Uploaded 2 transforms to host:/path"

        _finish_upload(widget, mids={1, 2}, msg=msg)

        assert widget.server_status_label.text == msg


_MISSING_METADATA_MSG = (
    "No manual_align_metadata.json found at package root or parent directory."
)
_INVALID_JSON_MSG = "manual_align_metadata.json is not valid JSON; using defaults."


def _make_package_metadata_widget(
    tmp_path: Path,
    *,
    level: int = 2,
    pairs: list[tuple[int, int]] | None = None,
):
    """Minimal ServerMixin + PairLoadingMixin widget for package ingest wiring tests."""
    from linumpy_manual_align.remote.cross_section import CrossSectionManager
    from linumpy_manual_align.ui.widget_pair_loading import PairLoadingMixin

    class _PackageMetadataWidget(ServerMixin, PairLoadingMixin):
        pass

    widget = object.__new__(_PackageMetadataWidget)
    widget.output_dir = tmp_path
    widget.pairs = pairs or [(0, 1)]
    widget.level = level
    widget.server_status_label = _StatusLabel()
    widget.viewer = _Viewer()
    widget._cs_mgr = CrossSectionManager()
    widget._btn_mode_z = _BtnModeZ()
    widget.saved_pairs = set()
    widget.slice_ids = []
    widget.existing_transforms = {}
    widget.aips_dir = None
    widget.slice_paths = {}
    widget.pair_paths_xy = {}
    widget.aips_xz_dir = None
    widget.aips_yz_dir = None
    widget.slice_paths_xz = {}
    widget.slice_paths_yz = {}
    widget.pair_paths_xz = {}
    widget.pair_paths_yz = {}
    widget.transforms_dir = None
    return widget


def _patch_package_discovery(monkeypatch: pytest.MonkeyPatch, widget, aips: Path) -> None:
    del monkeypatch, aips
    widget._refresh_saved_pairs = lambda: None
    widget._rebuild_pairs = lambda: None
    widget._refresh_session_state = lambda: None


class TestPackageMetadataWiring:
    """Server load paths use load_manual_align_metadata (CONT-05, CONT-06)."""

    def test_explicit_level_applied_from_metadata(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        widget = _make_package_metadata_widget(tmp_path, level=2)
        aips = tmp_path / "manual_align_package" / "aips"
        aips.mkdir(parents=True)
        (aips / "slice_z00.npz").touch()
        (aips.parent / METADATA_FILENAME).write_text(json.dumps({"pyramid_level": 5}))
        _patch_package_discovery(monkeypatch, widget, aips)

        widget._load_existing_package(aips, base_status="Package loaded")

        assert widget.level == 5

    def test_level_preserved_when_metadata_missing(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        widget = _make_package_metadata_widget(tmp_path, level=2)
        aips = tmp_path / "manual_align_package" / "aips"
        aips.mkdir(parents=True)
        (aips / "slice_z00.npz").touch()
        _patch_package_discovery(monkeypatch, widget, aips)

        widget._load_existing_package(aips, base_status="Package loaded")

        assert widget.level == 2

    def test_level_preserved_when_metadata_has_no_level_key(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        widget = _make_package_metadata_widget(tmp_path, level=2)
        aips = tmp_path / "manual_align_package" / "aips"
        aips.mkdir(parents=True)
        (aips / "slice_z00.npz").touch()
        (aips.parent / METADATA_FILENAME).write_text(
            json.dumps({"slices_remote_dir": "/remote/slices"})
        )
        _patch_package_discovery(monkeypatch, widget, aips)

        widget._load_existing_package(aips, base_status="Package loaded")

        assert widget.level == 2

    def test_missing_metadata_surfaces_warning_on_load_existing(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        widget = _make_package_metadata_widget(tmp_path)
        aips = tmp_path / "manual_align_package" / "aips"
        aips.mkdir(parents=True)
        (aips / "slice_z00.npz").touch()
        base = "Existing package loaded"
        _patch_package_discovery(monkeypatch, widget, aips)

        widget._load_existing_package(aips, base_status=base)

        label = widget.server_status_label.text
        assert base in label
        assert _MISSING_METADATA_MSG in label
        assert "metadata.missing" not in label

    def test_invalid_json_surfaces_warning_on_load_existing(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        widget = _make_package_metadata_widget(tmp_path)
        aips = tmp_path / "manual_align_package" / "aips"
        aips.mkdir(parents=True)
        (aips / "slice_z00.npz").touch()
        (aips.parent / METADATA_FILENAME).write_text("{ not valid json")
        _patch_package_discovery(monkeypatch, widget, aips)

        widget._load_existing_package(aips, base_status="Loaded")

        label = widget.server_status_label.text
        assert _INVALID_JSON_MSG in label
        assert "metadata.invalid_json" not in label

    def test_download_finish_composes_base_status_and_warnings(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        widget = _make_package_metadata_widget(tmp_path)
        widget.btn_download = _ButtonStub()
        widget.btn_upload = _ButtonStub()
        widget.server_progress = _ProgressStub()
        widget._worker = None

        local_dir = tmp_path / "server_package"
        aips = local_dir / "manual_align_package" / "aips"
        aips.mkdir(parents=True)
        (aips / "slice_z00.npz").touch()
        _patch_package_discovery(monkeypatch, widget, aips)

        base = "Download complete"
        widget._on_download_finished(True, base, local_dir)

        label = widget.server_status_label.text
        assert base in label
        assert _MISSING_METADATA_MSG in label

    def test_apply_metadata_called_without_second_parse(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        widget = _make_package_metadata_widget(tmp_path)
        aips = tmp_path / "manual_align_package" / "aips"
        aips.mkdir(parents=True)
        (aips / "slice_z00.npz").touch()
        (aips.parent / METADATA_FILENAME).write_text(json.dumps({"pyramid_level": 1}))
        _patch_package_discovery(monkeypatch, widget, aips)

        apply_calls: list[tuple] = []

        def record_apply(normalized, issues) -> None:
            apply_calls.append((normalized, issues))

        widget._cs_mgr.apply_metadata = record_apply

        widget._load_existing_package(aips, base_status="Loaded")

        assert len(apply_calls) == 1
        normalized, _issues = apply_calls[0]
        assert normalized.pyramid_level == 1
        assert normalized.pyramid_level_explicit is True


class TestStartupMetadataWarnings:
    """Startup cached load preserves ContractIssue warnings (D-01, D-02)."""

    def test_startup_preserves_missing_metadata_warning(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Startup branch passes base_status so warnings are not clobbered."""
        from linumpy_manual_align.remote.cross_section import CrossSectionManager
        from linumpy_manual_align.ui.widget import ManualAlignWidget

        output_dir = tmp_path / "sub" / "manual_transforms"
        output_dir.mkdir(parents=True)
        aips = tmp_path / "sub" / "server_package" / "manual_align_package" / "aips"
        aips.mkdir(parents=True)
        (aips / "slice_z00.npz").touch()

        widget = ManualAlignWidget.__new__(ManualAlignWidget)
        widget.output_dir = output_dir
        widget.level = 2
        widget.server_config = MagicMock()
        widget.pairs = []
        widget.server_status_label = _StatusLabel()
        widget.viewer = _Viewer()
        widget._cs_mgr = CrossSectionManager()
        widget._btn_mode_z = _BtnModeZ()
        widget.saved_pairs = set()
        widget.existing_transforms = {}
        _patch_package_discovery(monkeypatch, widget, aips)

        existing = widget._find_existing_package()
        assert existing is not None
        base_status = f"Existing package loaded from {existing.parent}"
        widget._load_existing_package(existing, base_status=base_status)

        label = widget.server_status_label.text
        assert "Existing package loaded from" in label
        assert _MISSING_METADATA_MSG in label
        assert widget.level == 2


def _make_pair_loading_widget(output_dir: Path):
    """Build a minimal PairLoadingMixin widget for saved-pair discovery tests."""
    from linumpy_manual_align.ui.widget_pair_loading import PairLoadingMixin

    class _PairLoadingWidget(PairLoadingMixin):
        pass

    widget = object.__new__(_PairLoadingWidget)
    widget.output_dir = output_dir
    widget.saved_pairs = set()
    return widget


def _write_transform(slice_dir: Path) -> None:
    """Create a slice directory containing a transform.tfm file."""
    slice_dir.mkdir(parents=True, exist_ok=True)
    (slice_dir / TRANSFORM_FILENAME).write_text("stub", encoding="utf-8")


class TestSavedPairsDiscovery:
    """Contract-helper-based saved-pair discovery (CONT-07, D-10)."""

    def test_saved_pairs_added_when_transform_present(self, tmp_path: Path) -> None:
        """A slice_z## dir built via manual_output_dir with transform.tfm is discovered."""
        _write_transform(manual_output_dir(tmp_path, 1))
        widget = _make_pair_loading_widget(tmp_path)

        widget._refresh_saved_pairs()

        assert 1 in widget.saved_pairs

    def test_saved_pairs_skips_dir_without_transform(self, tmp_path: Path) -> None:
        """A slice_z## dir WITHOUT transform.tfm is not added (transform.tfm filter)."""
        manual_output_dir(tmp_path, 2).mkdir(parents=True)
        widget = _make_pair_loading_widget(tmp_path)

        widget._refresh_saved_pairs()

        assert 2 not in widget.saved_pairs
        assert widget.saved_pairs == set()

    def test_saved_pairs_ignores_non_conforming_names(self, tmp_path: Path) -> None:
        """Directories not matching the strict slice_z layout are ignored."""
        # ``slice_1`` (missing ``z``) and ``slicez01`` (missing underscore) are
        # non-conforming and must never contribute a moving id, even with a
        # transform.tfm present.
        _write_transform(tmp_path / "slice_1")
        _write_transform(tmp_path / "slicez01")
        widget = _make_pair_loading_widget(tmp_path)

        widget._refresh_saved_pairs()

        assert widget.saved_pairs == set()

    def test_saved_pairs_strict_slice_z_parsing(self, tmp_path: Path) -> None:
        """Pins the strict ``slice_z<digits>`` contract (discover_manual_slice_dirs).

        Both ``slice_z01`` and ``slice_z001`` parse to moving id 1; only dirs
        containing transform.tfm are counted, and non ``slice_z`` names are
        excluded entirely.
        """
        _write_transform(manual_output_dir(tmp_path, 1))  # slice_z01
        _write_transform(tmp_path / "slice_z001")  # also parses to id 1
        _write_transform(tmp_path / "notaslice")  # excluded
        widget = _make_pair_loading_widget(tmp_path)

        widget._refresh_saved_pairs()

        assert widget.saved_pairs == {1}

    def test_saved_pairs_missing_output_dir_is_noop(self, tmp_path: Path) -> None:
        """Discovery on a non-existent output_dir leaves saved_pairs empty."""
        widget = _make_pair_loading_widget(tmp_path / "does_not_exist")

        widget._refresh_saved_pairs()

        assert widget.saved_pairs == set()
