"""Tests for the settings module (AppSettings / DEFAULTS)."""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def isolated_settings(qapp):
    """Each test gets a fresh in-memory QSettings store."""
    from linumpy_manual_align.settings import settings

    settings._qs.clear()
    yield settings
    settings._qs.clear()


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------


def test_get_returns_default_when_not_set(isolated_settings):
    """get() returns the coded default when no value has been stored."""
    from linumpy_manual_align.settings import DEFAULTS

    for key, expected in DEFAULTS.items():
        result = isolated_settings.get(key)
        assert result == type(expected)(expected), f"Default mismatch for {key!r}"


def test_get_unknown_key_raises(isolated_settings):
    with pytest.raises(KeyError):
        isolated_settings.get("nonexistent/key")


# ---------------------------------------------------------------------------
# Round-trip set / get with type coercion
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "key, value",
    [
        ("shortcuts/translate_fine_px", 3),
        ("shortcuts/translate_coarse_px", 15),
        ("shortcuts/translate_large_px", 100),
        ("shortcuts/rotate_fine_deg", 0.5),
        ("shortcuts/rotate_coarse_deg", 2.0),
        ("shortcuts/rotate_large_deg", 10.0),
        ("shortcuts/cs_nudge_px", 20),
        ("prefetch/steps", 8),
        ("prefetch/evict_radius_multiplier", 20),
        ("spin/tx_ty_step", 2.0),
        ("spin/rot_step", 0.25),
        ("spin/tile_step", 8),
        ("server/default_host", "10.0.0.1"),
        ("server/remote_python", "/usr/bin/python3"),
    ],
)
def test_round_trip(isolated_settings, key, value):
    isolated_settings.set(key, value)
    result = isolated_settings.get(key)
    from linumpy_manual_align.settings import DEFAULTS

    assert type(result) is type(DEFAULTS[key])
    assert result == type(DEFAULTS[key])(value)


def test_set_unknown_key_raises(isolated_settings):
    with pytest.raises(KeyError):
        isolated_settings.set("bogus/key", 42)


# ---------------------------------------------------------------------------
# reset(key) and reset_all()
# ---------------------------------------------------------------------------


def test_reset_key_restores_default(isolated_settings):
    from linumpy_manual_align.settings import DEFAULTS

    isolated_settings.set("shortcuts/translate_coarse_px", 999)
    assert isolated_settings.get("shortcuts/translate_coarse_px") == 999

    isolated_settings.reset("shortcuts/translate_coarse_px")
    assert isolated_settings.get("shortcuts/translate_coarse_px") == DEFAULTS["shortcuts/translate_coarse_px"]


def test_reset_all_restores_all_defaults(isolated_settings):
    from linumpy_manual_align.settings import DEFAULTS

    for key in DEFAULTS:
        default = DEFAULTS[key]
        isolated_settings.set(key, type(default)(default))  # set to default value (forces write)

    isolated_settings.reset_all()

    for key, expected in DEFAULTS.items():
        assert isolated_settings.get(key) == expected, f"Default not restored for {key!r}"


# ---------------------------------------------------------------------------
# changed signal
# ---------------------------------------------------------------------------


def test_changed_signal_fires_on_set(isolated_settings, qapp):
    received: list[tuple] = []
    isolated_settings.changed.connect(lambda k, v: received.append((k, v)))

    isolated_settings.set("shortcuts/translate_fine_px", 7)

    assert len(received) == 1
    assert received[0][0] == "shortcuts/translate_fine_px"
    assert int(received[0][1]) == 7


def test_changed_signal_fires_on_reset(isolated_settings, qapp):
    from linumpy_manual_align.settings import DEFAULTS

    isolated_settings.set("shortcuts/translate_fine_px", 7)
    received: list[tuple] = []
    isolated_settings.changed.connect(lambda k, v: received.append((k, v)))

    isolated_settings.reset("shortcuts/translate_fine_px")

    assert len(received) == 1
    assert received[0][0] == "shortcuts/translate_fine_px"
    assert int(received[0][1]) == DEFAULTS["shortcuts/translate_fine_px"]


def test_changed_signal_fires_for_all_keys_on_reset_all(isolated_settings, qapp):
    from linumpy_manual_align.settings import DEFAULTS

    received_keys: list[str] = []
    isolated_settings.changed.connect(lambda k, _v: received_keys.append(k))

    isolated_settings.reset_all()

    assert set(received_keys) == set(DEFAULTS.keys())
