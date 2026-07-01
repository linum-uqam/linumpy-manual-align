"""Regression tests for lazy package __init__ (PEP 562)."""

from __future__ import annotations

import subprocess
import sys

import pytest


def _run_subprocess_check(code: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=False,
    )


def test_top_level_import_does_not_load_api() -> None:
    result = _run_subprocess_check(
        'import linumpy_manual_align; import sys; '
        'assert "linumpy_manual_align.api" not in sys.modules'
    )
    assert result.returncode == 0, result.stderr or result.stdout


def test_contracts_subpackage_import_does_not_load_api() -> None:
    result = _run_subprocess_check(
        'import linumpy_manual_align.contracts; import sys; '
        'assert "linumpy_manual_align.api" not in sys.modules'
    )
    assert result.returncode == 0, result.stderr or result.stdout


def test_create_manual_align_widget_is_callable() -> None:
    import linumpy_manual_align

    widget_factory = getattr(linumpy_manual_align, "create_manual_align_widget")
    assert callable(widget_factory)


def test_unknown_attribute_raises_attribute_error() -> None:
    import linumpy_manual_align

    with pytest.raises(AttributeError):
        _ = linumpy_manual_align.does_not_exist  # noqa: SLF001


def test_dunder_metadata() -> None:
    import linumpy_manual_align

    assert linumpy_manual_align.__version__ == "0.1.0"
    assert "create_manual_align_widget" in linumpy_manual_align.__all__
