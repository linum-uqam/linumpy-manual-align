"""Tests for widget data structures and CLI argument parsing."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from linumpy_manual_align.__main__ import parse_args
from linumpy_manual_align.contracts import SEVERITY_ERROR, SEVERITY_WARNING, ContractIssue, validate_manual_output
from linumpy_manual_align.io.transform_io import save_transform
from linumpy_manual_align.state import _MAX_UNDO_HISTORY, AlignmentState, UndoStack
from linumpy_manual_align.ui.widget_close_guard import CloseGuardMixin
from linumpy_manual_align.ui.widget_undo_save import UndoSaveMixin
import linumpy_manual_align.ui.widget_close_guard as widget_close_guard_module
import linumpy_manual_align.ui.widget_undo_save as widget_undo_save_module


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
    widget.pair_centers = {m: (50.0, 50.0) for m in mids}
    widget._current_offsets = {m: (0, 0) for m in mids}
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
        widget.pair_centers = {m: (50.0, 50.0) for m in mids}
        widget._current_offsets = {m: (0, 0) for m in mids}
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
