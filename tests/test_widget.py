"""Tests for widget data structures and CLI argument parsing."""

from __future__ import annotations

from linumpy_manual_align.__main__ import parse_args
from linumpy_manual_align.widget import AlignmentState, UndoStack


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
        assert args.level == 1
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
