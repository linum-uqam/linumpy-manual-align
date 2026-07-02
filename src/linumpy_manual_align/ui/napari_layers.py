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

See also
--------
``docs/architecture/NAPI-HOT-PATHS.md`` — documented napari mutation hot paths,
baseline call graphs (entry point → mixin → napari API), and Phase 9 optimization
targets (NAPI-HOT-PATHS audit deliverable, D-15).
"""

from __future__ import annotations

import contextlib
import logging
from typing import TYPE_CHECKING, Protocol, Sequence

import numpy as np

from linumpy_manual_align.io.image_utils import OVERLAY_COLOR, OVERLAY_DIFF, build_overlay, content_bbox

if TYPE_CHECKING:
    from napari.viewer import Viewer

logger = logging.getLogger(__name__)

FIXED_LAYER_NAME = "Fixed"
MOVING_LAYER_NAME = "Moving"
COMPOSITE_LAYER_NAME = "Composite"
ALLOWED_LAYER_NAMES = frozenset({FIXED_LAYER_NAME, MOVING_LAYER_NAME, COMPOSITE_LAYER_NAME})


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


def remove_all_layers(viewer: Viewer) -> None:
    """Remove every layer from the viewer."""
    viewer.layers.clear()


def create_pair_layers(
    viewer: Viewer,
    *,
    fixed_data: np.ndarray,
    moving_data: np.ndarray,
    fixed_scale: list[float],
    moving_scale: list[float],
    fixed_gamma: float,
    fixed_opacity: float,
    fixed_clim: tuple[float, float],
    moving_gamma: float,
    moving_opacity: float,
    moving_clim: tuple[float, float],
    fid: int,
    mid: int,
    projection_mode: str,
) -> tuple[object, object]:
    """Create fixed and moving Image layers after clearing existing viewer layers."""
    del fid, mid, projection_mode  # stable names (D-06); params kept for call-site compatibility.
    remove_all_layers(viewer)
    fixed_layer = viewer.add_image(
        fixed_data,
        name=FIXED_LAYER_NAME,
        colormap="green",
        blending="additive",
        contrast_limits=fixed_clim,
        gamma=fixed_gamma,
        opacity=fixed_opacity,
        scale=fixed_scale,
    )
    moving_layer = viewer.add_image(
        moving_data,
        name=MOVING_LAYER_NAME,
        colormap="red",
        blending="additive",
        contrast_limits=moving_clim,
        gamma=moving_gamma,
        opacity=moving_opacity,
        scale=list(moving_scale),
    )
    return fixed_layer, moving_layer


def update_pair_data(
    viewer: Viewer,
    *,
    fixed_layer: object,
    moving_layer: object,
    fixed_data: np.ndarray,
    moving_data: np.ndarray,
) -> None:
    """Push new AIP data into existing fixed/moving layers without full teardown."""
    del viewer
    fixed_layer.data = fixed_data  # type: ignore[attr-defined]
    moving_layer.data = moving_data  # type: ignore[attr-defined]


def assert_layer_inventory(
    viewer: Viewer,
    *,
    projection_mode: str,
    overlay_mode: str,
    overlay_color: str = OVERLAY_COLOR,
) -> None:
    """Log a warning when layer count or names are unexpected (D-14)."""
    expected = 2
    if projection_mode == "xy" and overlay_mode != overlay_color:
        expected = 3
    count = len(viewer.layers)
    if count != expected:
        logger.warning(
            "Unexpected napari layer count: got %d, expected %d (mode=%s overlay=%s)",
            count,
            expected,
            projection_mode,
            overlay_mode,
        )
    for layer in viewer.layers:
        if layer.name not in ALLOWED_LAYER_NAMES:
            logger.warning("Unexpected napari layer name: %r", layer.name)


def set_overlay_mode(
    viewer: Viewer,
    *,
    overlay_mode: str,
    fixed_layer: object | None,
    moving_layer: object | None,
    composite_layer: object | None,
    base_image: np.ndarray | None,
    scale: list[float],
    projection_mode: str = "xy",
) -> object | None:
    """Switch overlay mode; add/remove composite and toggle fixed/moving visibility."""
    allow_composite = projection_mode == "xy"
    if not allow_composite:
        if fixed_layer is not None:
            fixed_layer.visible = True  # type: ignore[attr-defined]
        if moving_layer is not None:
            moving_layer.visible = True  # type: ignore[attr-defined]
        if composite_layer is not None:
            with contextlib.suppress(ValueError):
                viewer.layers.remove(composite_layer)  # type: ignore[arg-type]
        return None

    is_color = overlay_mode == OVERLAY_COLOR
    colormap = "inferno" if overlay_mode == OVERLAY_DIFF else "gray"

    if fixed_layer is not None:
        fixed_layer.visible = is_color  # type: ignore[attr-defined]
    if moving_layer is not None:
        moving_layer.visible = is_color  # type: ignore[attr-defined]

    if is_color:
        if composite_layer is not None:
            with contextlib.suppress(ValueError):
                viewer.layers.remove(composite_layer)  # type: ignore[arg-type]
        return None
    if composite_layer is not None:
        composite_layer.colormap = colormap  # type: ignore[attr-defined]
        return composite_layer
    if base_image is not None and fixed_layer is not None:
        comp = np.zeros_like(base_image)
        composite_layer = viewer.add_image(
            comp,
            name=COMPOSITE_LAYER_NAME,
            colormap=colormap,
            blending="translucent",
            contrast_limits=(0.0, 1.0),
            scale=list(scale),
        )
        return composite_layer
    return None


def should_recenter_on_switch(
    *,
    projection_mode: str,
    full_teardown: bool,
    preserve_camera: bool,
) -> bool:
    """Return whether the viewer should reset_view after a pair load.

    XY always recenters so pair switches stay centered on content. In Z (xz/yz),
    any full teardown recenters to avoid cumulative drift from stale camera state;
    ``preserve_camera`` is retained for call-site signature stability but no
    longer suppresses Z recenter on full teardown.
    """
    if projection_mode == "xy":
        return True
    return full_teardown


def should_recenter_after_cross_section(
    *,
    projection_mode: str,
    prev_shape: tuple[int, ...],
    new_shape: tuple[int, ...],
) -> bool:
    """Return whether the cross-section hook should pan the camera after a frame update.

    Z cross-section frames always trigger content-aware camera panning so moving
    tissue stays centered as the slider scrolls within the fixed XZ/YZ canvas.
    ``prev_shape`` and ``new_shape`` are retained for call-site signature stability only.
    """
    del prev_shape, new_shape
    return projection_mode in {"xz", "yz"}


def content_center_world(
    layers: Sequence[tuple[np.ndarray, Sequence[float], Sequence[float]]],
    *,
    threshold: float = 0.02,
) -> tuple[float, float] | None:
    """Return the combined tissue bounding-box center in world coordinates.

    Each layer is ``(data, scale, translate)`` where world position of pixel
    ``(r, c)`` is ``(r * sy + ty, c * sx + tx)``. Returns ``None`` when no
    layer has visible content so callers can leave the camera untouched.
    """
    world_r_min: float | None = None
    world_r_max: float | None = None
    world_c_min: float | None = None
    world_c_max: float | None = None

    for data, scale, translate in layers:
        arr = np.asarray(data)
        if not np.any(arr > threshold):
            continue
        sy, sx = float(scale[0]), float(scale[1])
        ty, tx = float(translate[0]), float(translate[1])
        r1, c1, r2, c2 = content_bbox(arr, threshold=threshold, padding=0)
        wr1 = r1 * sy + ty
        wr2 = r2 * sy + ty
        wc1 = c1 * sx + tx
        wc2 = c2 * sx + tx
        world_r_min = wr1 if world_r_min is None else min(world_r_min, wr1)
        world_r_max = wr2 if world_r_max is None else max(world_r_max, wr2)
        world_c_min = wc1 if world_c_min is None else min(world_c_min, wc1)
        world_c_max = wc2 if world_c_max is None else max(world_c_max, wc2)

    if world_r_min is None or world_c_min is None or world_r_max is None or world_c_max is None:
        return None
    return ((world_r_min + world_r_max) / 2.0, (world_c_min + world_c_max) / 2.0)


def content_fit_zoom_factor(
    layer: tuple[np.ndarray, Sequence[float], Sequence[float]],
    *,
    threshold: float = 0.02,
    min_factor: float = 1.0,
    max_factor: float = 20.0,
) -> float | None:
    """Return the zoom-in factor so tissue column width fills the full canvas width.

    Multiplies the post-``reset_view`` base zoom by ``full_world_width /
    tissue_world_width`` so uncropped Z cross-sections zoom to the moving tissue
    instead of the full ~1351px extent. Returns ``None`` on blank frames.
    """
    data, scale, _translate = layer
    arr = np.asarray(data)
    if not np.any(arr > threshold):
        return None
    sx = float(scale[1])
    _r1, c1, _r2, c2 = content_bbox(arr, threshold=threshold, padding=0)
    tissue_w = (c2 - c1) * sx
    full_w = arr.shape[1] * sx
    if tissue_w <= 0:
        return None
    factor = full_w / tissue_w
    return max(min_factor, min(max_factor, factor))


def refresh_composite(
    viewer: Viewer,
    *,
    overlay_mode: str,
    composite_layer: object | None,
    fixed_base: np.ndarray | None,
    shifted_moving: np.ndarray | None,
    tile_size: int,
) -> None:
    """Recompute and push composite layer data for non-color overlay modes."""
    del viewer
    if overlay_mode == OVERLAY_COLOR or composite_layer is None:
        return
    if fixed_base is None or shifted_moving is None:
        return
    composite_layer.data = build_overlay(  # type: ignore[attr-defined]
        fixed_base,
        shifted_moving,
        mode=overlay_mode,
        tile_size=tile_size,
    )
