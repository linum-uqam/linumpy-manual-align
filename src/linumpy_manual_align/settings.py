"""Persistent application settings for linumpy-manual-align.

Wraps :class:`qtpy.QtCore.QSettings` (organisation ``linum-uqam``,
application ``linumpy-manual-align``) to provide typed get/set access
backed by a single :data:`DEFAULTS` dict.

Usage::

    from linumpy_manual_align.settings import settings
    step = settings.get("shortcuts/translate_coarse_px")  # int
    settings.set("shortcuts/translate_coarse_px", 20)
    settings.reset("shortcuts/translate_coarse_px")
    settings.reset_all()

The ``changed`` signal is emitted with ``(key, new_value)`` whenever
:meth:`AppSettings.set` writes a value.
"""

from __future__ import annotations

from qtpy.QtCore import QObject, Signal
from qtpy.QtCore import QSettings as _QSettings

# ---------------------------------------------------------------------------
# Default values — the single source of truth for all user-tunable constants.
# Keys are grouped by "/" prefix and match the QSettings key hierarchy.
# ---------------------------------------------------------------------------

DEFAULTS: dict[str, int | float | str] = {
    # Keyboard shortcut step sizes
    "shortcuts/translate_fine_px": 1,
    "shortcuts/translate_coarse_px": 10,
    "shortcuts/translate_large_px": 50,
    "shortcuts/rotate_fine_deg": 0.1,
    "shortcuts/rotate_coarse_deg": 1.0,
    "shortcuts/rotate_large_deg": 5.0,
    "shortcuts/cs_nudge_px": 10,
    "shortcuts/cs_nudge_fine_px": 1,
    # Cross-section prefetch / cache (spacing along the axis uses shortcuts/cs_nudge_px)
    "prefetch/steps": 5,
    # evict_radius = evict_radius_multiplier * cs_nudge_px
    "prefetch/evict_radius_multiplier": 15,
    # Spinbox single-step increments
    "spin/tx_ty_step": 1.0,
    "spin/rot_step": 0.1,
    "spin/tile_step": 4,
    # Server (empty in repo — values live in each user's QSettings, not in git)
    "server/default_host": "",
    # Must be set to the server's linumpy venv (zarr, ome-zarr, …); see remote/cs_script.
    "server/remote_python": "",
}


class AppSettings(QObject):
    """Thin typed wrapper around :class:`QSettings`.

    Parameters
    ----------
    parent:
        Optional Qt parent.  Usually *None* for the module-level singleton.

    Signals
    -------
    changed(key, value)
        Emitted after :meth:`set` writes a new value.
    """

    changed = Signal(str, object)  # (key, new_value)

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._qs = _QSettings("linum-uqam", "linumpy-manual-align")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get(self, key: str) -> int | float | str:
        """Return the stored value for *key*, or the default if not set.

        The return type is coerced to match :data:`DEFAULTS`; if *key* is
        unknown, :exc:`KeyError` is raised.
        """
        default = DEFAULTS[key]
        raw = self._qs.value(key, default)
        if raw is None:
            raw = default
        return type(default)(raw)  # type: ignore[call-arg]

    def set(self, key: str, value: int | float | str) -> None:
        """Write *value* for *key* and emit :attr:`changed`."""
        if key not in DEFAULTS:
            raise KeyError(f"Unknown settings key: {key!r}")
        self._qs.setValue(key, value)
        self._qs.sync()
        self.changed.emit(key, value)

    def reset(self, key: str) -> None:
        """Remove the stored override for *key* (next :meth:`get` returns default)."""
        self._qs.remove(key)
        self._qs.sync()
        self.changed.emit(key, DEFAULTS[key])

    def reset_all(self) -> None:
        """Remove all stored overrides and emit :attr:`changed` for every key."""
        self._qs.clear()
        self._qs.sync()
        for key, default in DEFAULTS.items():
            self.changed.emit(key, default)


# Module-level singleton — import and use directly.
settings = AppSettings()
