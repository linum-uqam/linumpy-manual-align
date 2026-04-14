"""Alignment state data structures for the manual alignment widget.

``AlignmentState`` is a plain snapshot of the three editable parameters
for one slice pair.  ``UndoStack`` maintains a bounded linear history of
those snapshots to support undo/redo.
"""

from __future__ import annotations

from dataclasses import dataclass, field

_MAX_UNDO_HISTORY = 500


@dataclass
class AlignmentState:
    """Snapshot of alignment parameters for one slice pair."""

    tx: float = 0.0
    ty: float = 0.0
    rotation: float = 0.0


@dataclass
class UndoStack:
    """Simple linear undo/redo stack per slice pair."""

    _history: list[AlignmentState] = field(default_factory=lambda: [AlignmentState()])
    _index: int = 0

    def __init__(self, initial: AlignmentState | None = None):
        self._history = [initial if initial is not None else AlignmentState()]
        self._index = 0

    def push(self, state: AlignmentState) -> None:
        """Push *state* onto the stack, discarding any redo history."""
        self._history = self._history[: self._index + 1]
        self._history.append(AlignmentState(state.tx, state.ty, state.rotation))
        self._index = len(self._history) - 1
        if len(self._history) > _MAX_UNDO_HISTORY:
            trim = len(self._history) - _MAX_UNDO_HISTORY
            self._history = self._history[trim:]
            self._index -= trim

    def undo(self) -> AlignmentState | None:
        """Step backward; returns the previous state or *None* if at the beginning."""
        if self._index > 0:
            self._index -= 1
            s = self._history[self._index]
            return AlignmentState(s.tx, s.ty, s.rotation)
        return None

    def redo(self) -> AlignmentState | None:
        """Step forward; returns the next state or *None* if at the end."""
        if self._index < len(self._history) - 1:
            self._index += 1
            s = self._history[self._index]
            return AlignmentState(s.tx, s.ty, s.rotation)
        return None

    @property
    def current(self) -> AlignmentState:
        return self._history[self._index]
