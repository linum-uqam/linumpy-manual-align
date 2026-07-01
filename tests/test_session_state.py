"""Headless tests for session pair state classification and summary formatting."""

from __future__ import annotations

from pathlib import Path

import pytest

from linumpy_manual_align.contracts import (
    PairSessionState,
    classify_pair_state,
    combo_prefix,
    format_session_summary,
    moving_ids_from_slice_dirs,
)


@pytest.mark.parametrize(
    ("kwargs", "expected"),
    [
        # invalid beats everything
        (
            {
                "invalid": True,
                "in_unsaved": True,
                "in_saved": True,
                "in_uploaded": True,
                "ready": True,
                "server_enabled": True,
            },
            PairSessionState.INVALID,
        ),
        (
            {
                "invalid": True,
                "in_unsaved": False,
                "in_saved": False,
                "in_uploaded": False,
                "ready": False,
                "server_enabled": False,
            },
            PairSessionState.INVALID,
        ),
        # unsaved beats saved/uploaded/ready
        (
            {
                "invalid": False,
                "in_unsaved": True,
                "in_saved": True,
                "in_uploaded": True,
                "ready": True,
                "server_enabled": True,
            },
            PairSessionState.UNSAVED,
        ),
        (
            {
                "invalid": False,
                "in_unsaved": True,
                "in_saved": False,
                "in_uploaded": False,
                "ready": False,
                "server_enabled": False,
            },
            PairSessionState.UNSAVED,
        ),
        # unchanged when not saved (regardless of ready/uploaded)
        (
            {
                "invalid": False,
                "in_unsaved": False,
                "in_saved": False,
                "in_uploaded": True,
                "ready": True,
                "server_enabled": True,
            },
            PairSessionState.UNCHANGED,
        ),
        (
            {
                "invalid": False,
                "in_unsaved": False,
                "in_saved": False,
                "in_uploaded": False,
                "ready": False,
                "server_enabled": False,
            },
            PairSessionState.UNCHANGED,
        ),
        # uploaded supersedes ready when server enabled
        (
            {
                "invalid": False,
                "in_unsaved": False,
                "in_saved": True,
                "in_uploaded": True,
                "ready": True,
                "server_enabled": True,
            },
            PairSessionState.UPLOADED,
        ),
        (
            {
                "invalid": False,
                "in_unsaved": False,
                "in_saved": True,
                "in_uploaded": True,
                "ready": False,
                "server_enabled": True,
            },
            PairSessionState.UPLOADED,
        ),
        # ready when saved, server enabled, not uploaded
        (
            {
                "invalid": False,
                "in_unsaved": False,
                "in_saved": True,
                "in_uploaded": False,
                "ready": True,
                "server_enabled": True,
            },
            PairSessionState.READY,
        ),
        # saved-local fallback when saved but not ready/uploaded
        (
            {
                "invalid": False,
                "in_unsaved": False,
                "in_saved": True,
                "in_uploaded": False,
                "ready": False,
                "server_enabled": True,
            },
            PairSessionState.SAVED_LOCAL,
        ),
        (
            {
                "invalid": False,
                "in_unsaved": False,
                "in_saved": True,
                "in_uploaded": False,
                "ready": False,
                "server_enabled": False,
            },
            PairSessionState.SAVED_LOCAL,
        ),
        # local-only ignores uploaded/ready flags
        (
            {
                "invalid": False,
                "in_unsaved": False,
                "in_saved": True,
                "in_uploaded": True,
                "ready": True,
                "server_enabled": False,
            },
            PairSessionState.SAVED_LOCAL,
        ),
    ],
)
def test_classify_pair_state_priority(
    kwargs: dict[str, bool], expected: PairSessionState
) -> None:
    """D-08 priority: invalid > unsaved > unchanged > uploaded > ready > saved-local."""
    assert classify_pair_state(**kwargs) == expected


@pytest.mark.parametrize(
    ("state", "expected_prefix"),
    [
        (PairSessionState.UNCHANGED, "○ "),
        (PairSessionState.UNSAVED, "● "),
        (PairSessionState.SAVED_LOCAL, "✓ "),
        (PairSessionState.INVALID, "✗ "),
        (PairSessionState.READY, "◎ "),
        (PairSessionState.UPLOADED, "↑ "),
    ],
)
def test_combo_prefix_icons(state: PairSessionState, expected_prefix: str) -> None:
    """Each session state maps to one unicode prefix plus a single ASCII space."""
    assert combo_prefix(state) == expected_prefix


def test_format_session_summary_local_only() -> None:
    """Local-only summary uses four segments joined by middle dots."""
    counts = {
        PairSessionState.INVALID: 1,
        PairSessionState.UNSAVED: 2,
        PairSessionState.UNCHANGED: 3,
        PairSessionState.SAVED_LOCAL: 4,
    }
    assert format_session_summary(counts, server_enabled=False) == (
        "1 invalid · 2 unsaved · 3 unchanged · 4 saved-local"
    )


def test_format_session_summary_server_enabled() -> None:
    """Server-enabled summary adds ready and uploaded segments."""
    counts = {
        PairSessionState.INVALID: 1,
        PairSessionState.UNSAVED: 2,
        PairSessionState.UNCHANGED: 3,
        PairSessionState.SAVED_LOCAL: 4,
        PairSessionState.READY: 2,
        PairSessionState.UPLOADED: 1,
    }
    assert format_session_summary(counts, server_enabled=True) == (
        "1 invalid · 2 unsaved · 3 unchanged · 4 saved-local · 2 ready · 1 uploaded"
    )


def test_format_session_summary_server_all_zero() -> None:
    """Missing count keys default to zero and still emit every segment."""
    assert format_session_summary({}, server_enabled=True) == (
        "0 invalid · 0 unsaved · 0 unchanged · 0 saved-local · 0 ready · 0 uploaded"
    )


def test_moving_ids_from_slice_dirs_happy_path() -> None:
    """Strict slice_z## directory names yield their moving IDs."""
    dirs = [Path("slice_z01"), Path("slice_z03")]
    assert moving_ids_from_slice_dirs(dirs) == {1, 3}


def test_moving_ids_from_slice_dirs_rejects_loose_and_foreign() -> None:
    """Loose or foreign directory names are skipped."""
    dirs = [Path("slice_z1"), Path("aips"), Path("slice_z02")]
    assert moving_ids_from_slice_dirs(dirs) == {2}


def test_moving_ids_from_slice_dirs_empty() -> None:
    """Empty input yields an empty set."""
    assert moving_ids_from_slice_dirs([]) == set()
