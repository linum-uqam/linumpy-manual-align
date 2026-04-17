"""Tests for remote cross-section script helpers."""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def isolated_settings(qapp):
    """Each test gets a fresh in-memory QSettings store."""
    from linumpy_manual_align.settings import settings

    settings._qs.clear()
    yield settings
    settings._qs.clear()


def test_remote_python_env_overrides_qsettings(isolated_settings, monkeypatch):
    """LINUMPY_MANUAL_ALIGN_REMOTE_PYTHON wins over Settings."""
    monkeypatch.setenv("LINUMPY_MANUAL_ALIGN_REMOTE_PYTHON", "/env/bin/python")
    isolated_settings.set("server/remote_python", "/opt/venv/bin/python")
    from linumpy_manual_align.remote.cs_script import _remote_python

    assert _remote_python() == "/env/bin/python"


def test_remote_python_empty_raises(isolated_settings, monkeypatch):
    """Unset remote interpreter must error with an actionable message (no fake python3 default)."""
    monkeypatch.delenv("LINUMPY_MANUAL_ALIGN_REMOTE_PYTHON", raising=False)
    isolated_settings.set("server/remote_python", "")
    from linumpy_manual_align.remote.cs_script import _remote_python

    with pytest.raises(RuntimeError, match="Remote Python is not configured"):
        _remote_python()


def test_remote_python_strips_whitespace(isolated_settings, monkeypatch):
    monkeypatch.delenv("LINUMPY_MANUAL_ALIGN_REMOTE_PYTHON", raising=False)
    isolated_settings.set("server/remote_python", "  /usr/bin/python3  ")
    from linumpy_manual_align.remote.cs_script import _remote_python

    assert _remote_python() == "/usr/bin/python3"
