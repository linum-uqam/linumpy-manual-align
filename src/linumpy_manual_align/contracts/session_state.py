"""Headless session pair state classification and summary formatting."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from enum import StrEnum
from pathlib import Path

from linumpy_manual_align.contracts.layout import format_manual_slice_dir, parse_manual_slice_dir


class PairSessionState(StrEnum):
    """Operator-facing display state for a single session pair."""

    UNCHANGED = "unchanged"
    UNSAVED = "unsaved"
    SAVED_LOCAL = "saved_local"
    INVALID = "invalid"
    READY = "ready"
    UPLOADED = "uploaded"


SESSION_PREFIX: dict[PairSessionState, str] = {
    PairSessionState.UNCHANGED: "○",
    PairSessionState.UNSAVED: "●",
    PairSessionState.SAVED_LOCAL: "✓",
    PairSessionState.INVALID: "✗",
    PairSessionState.READY: "◎",
    PairSessionState.UPLOADED: "↑",
}

SUMMARY_ORDER_LOCAL: tuple[PairSessionState, ...] = (
    PairSessionState.INVALID,
    PairSessionState.UNSAVED,
    PairSessionState.UNCHANGED,
    PairSessionState.SAVED_LOCAL,
)

SUMMARY_ORDER_SERVER: tuple[PairSessionState, ...] = (
    *SUMMARY_ORDER_LOCAL,
    PairSessionState.READY,
    PairSessionState.UPLOADED,
)


def combo_prefix(state: PairSessionState) -> str:
    """Return the unicode prefix icon for *state* followed by one ASCII space."""
    return f"{SESSION_PREFIX[state]} "


def classify_pair_state(
    *,
    invalid: bool,
    in_unsaved: bool,
    in_saved: bool,
    in_uploaded: bool,
    ready: bool,
    server_enabled: bool,
) -> PairSessionState:
    """Resolve the display state for one pair using the D-08 priority order."""
    if invalid:
        return PairSessionState.INVALID
    if in_unsaved:
        return PairSessionState.UNSAVED
    if not in_saved:
        return PairSessionState.UNCHANGED
    if server_enabled and in_uploaded:
        return PairSessionState.UPLOADED
    if server_enabled and ready:
        return PairSessionState.READY
    return PairSessionState.SAVED_LOCAL


def format_session_summary(
    counts: Mapping[PairSessionState, int],
    *,
    server_enabled: bool,
) -> str:
    """Render the persistent session counts summary line."""
    order = SUMMARY_ORDER_SERVER if server_enabled else SUMMARY_ORDER_LOCAL
    segments = (
        f"{counts.get(state, 0)} {state.value.replace('_', '-')}" for state in order
    )
    return " · ".join(segments)


def moving_ids_from_slice_dirs(dirs: Iterable[Path]) -> set[int]:
    """Extract moving IDs from strict ``slice_z##`` directory names."""
    result: set[int] = set()
    for path in dirs:
        moving_id = parse_manual_slice_dir(path.name)
        if moving_id is not None and path.name == format_manual_slice_dir(moving_id):
            result.add(moving_id)
    return result
