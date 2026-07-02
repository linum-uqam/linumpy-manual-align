"""Documented napari XY-pair layer lifecycle contract (Phase 9 implementation target).

This module is **documentation only** in Phase 8 (EXTR-02). It defines the
``NapariLayerLifecycle`` structural protocol that Phase 9 will implement as
stateless functions taking ``viewer: napari.Viewer`` as the first parameter.

Current ownership (unchanged until Phase 9 REFACTOR-SEQUENCE Step 6):

- **PairLoadingMixin** (`widget_pair_loading.py`): creates fixed and moving
  ``Image`` layers via ``viewer.add_image`` after a full viewer teardown loop
  (``while len(viewer.layers) > 0: viewer.layers.pop(0)``). This full
  teardown is the Phase 8 baseline; incremental layer update on pair switch is
  Phase 9 scope (NAPI-02).
- **OverlayStateMixin** (`widget_overlay.py`): owns composite layer add/remove,
  fixed/moving visibility for overlay modes, and composite data refresh via
  ``_rebuild_layer_visibility`` / ``_refresh_composite``.
- **ManualAlignWidget** retains ``fixed_layer``, ``moving_layer``, and
  ``_composite_layer`` references after Phase 9; ``napari_layers`` operates
  through the viewer while the widget holds layer refs.

**Out of scope:** cross-section remote layers managed by ``CrossSectionMixin``
have a different lifecycle and are excluded from this interface (D-21).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    import numpy as np
    from napari.viewer import Viewer


class NapariLayerLifecycle(Protocol):
    """Structural contract for XY pair layer create, update, remove, and overlay."""

    def create_pair_layers(
        self,
        viewer: Viewer,
        *,
        fixed_data: np.ndarray,
        moving_data: np.ndarray,
        fixed_scale: list[float],
        moving_scale: list[float],
        fid: int,
        mid: int,
        projection_mode: str,
    ) -> tuple[object, object]:
        """Create fixed and moving Image layers after clearing existing viewer layers."""

    def update_pair_data(
        self,
        viewer: Viewer,
        *,
        fixed_data: np.ndarray,
        moving_data: np.ndarray,
    ) -> None:
        """Push new AIP data into existing fixed/moving layers without full teardown."""

    def remove_all_layers(self, viewer: Viewer) -> None:
        """Remove every layer from the viewer (baseline: pop until empty)."""

    def set_overlay_mode(self, viewer: Viewer, *, mode: str) -> None:
        """Switch overlay mode; add/remove composite and toggle fixed/moving visibility."""

    def refresh_composite(self, viewer: Viewer, *, state: object) -> None:
        """Recompute and push composite layer data for non-color overlay modes."""
