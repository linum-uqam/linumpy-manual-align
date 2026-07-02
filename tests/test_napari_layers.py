"""Unit tests for ui/napari_layers stateless layer lifecycle functions."""

from __future__ import annotations

import numpy as np
import pytest

from linumpy_manual_align.io.image_utils import OVERLAY_CHECKER, OVERLAY_COLOR, OVERLAY_DIFF, build_overlay
from linumpy_manual_align.ui import napari_layers


@pytest.fixture
def sample_arrays() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(0)
    fixed = rng.random((32, 48)).astype(np.float32)
    moving = rng.random((32, 48)).astype(np.float32)
    return fixed, moving


def test_remove_all_layers_clears_viewer(make_napari_viewer) -> None:
    viewer = make_napari_viewer(show=False)
    viewer.add_image(np.zeros((4, 4)))
    viewer.add_image(np.ones((4, 4)))
    napari_layers.remove_all_layers(viewer)
    assert len(viewer.layers) == 0


def test_create_pair_layers_returns_two_named_layers(make_napari_viewer, sample_arrays) -> None:
    viewer = make_napari_viewer(show=False)
    fixed, moving = sample_arrays
    scale = [0.01, 0.01]

    fixed_layer, moving_layer = napari_layers.create_pair_layers(
        viewer,
        fixed_data=fixed,
        moving_data=moving,
        fixed_scale=scale,
        moving_scale=scale,
        fixed_gamma=0.7,
        fixed_opacity=0.9,
        fixed_clim=(0.1, 0.9),
        moving_gamma=0.5,
        moving_opacity=0.8,
        moving_clim=(0.0, 1.0),
        fid=1,
        mid=2,
        projection_mode="xy",
    )

    assert len(viewer.layers) == 2
    assert fixed_layer.name == napari_layers.FIXED_LAYER_NAME
    assert moving_layer.name == napari_layers.MOVING_LAYER_NAME
    assert fixed_layer.colormap.name == "green"
    assert moving_layer.colormap.name == "red"
    assert fixed_layer.blending == "additive"
    assert moving_layer.blending == "additive"
    assert fixed_layer.gamma == pytest.approx(0.7)
    assert moving_layer.gamma == pytest.approx(0.5)
    assert fixed_layer.opacity == pytest.approx(0.9)
    assert moving_layer.opacity == pytest.approx(0.8)
    assert tuple(fixed_layer.contrast_limits) == (0.1, 0.9)
    assert tuple(moving_layer.contrast_limits) == (0.0, 1.0)


def test_create_pair_layers_stable_names_all_projection_modes(make_napari_viewer, sample_arrays) -> None:
    viewer = make_napari_viewer(show=False)
    fixed, moving = sample_arrays
    scale = [0.01, 0.01]

    for mode in ("xy", "xz", "yz"):
        fixed_layer, moving_layer = napari_layers.create_pair_layers(
            viewer,
            fixed_data=fixed,
            moving_data=moving,
            fixed_scale=scale,
            moving_scale=scale,
            fixed_gamma=0.6,
            fixed_opacity=1.0,
            fixed_clim=(0.0, 1.0),
            moving_gamma=0.6,
            moving_opacity=1.0,
            moving_clim=(0.0, 1.0),
            fid=3,
            mid=4,
            projection_mode=mode,
        )
        assert fixed_layer.name == napari_layers.FIXED_LAYER_NAME
        assert moving_layer.name == napari_layers.MOVING_LAYER_NAME


def test_create_pair_layers_twice_leaves_exactly_two_layers(make_napari_viewer, sample_arrays) -> None:
    viewer = make_napari_viewer(show=False)
    fixed, moving = sample_arrays
    scale = [0.01, 0.01]
    kwargs = dict(
        fixed_scale=scale,
        moving_scale=scale,
        fixed_gamma=0.6,
        fixed_opacity=1.0,
        fixed_clim=(0.0, 1.0),
        moving_gamma=0.6,
        moving_opacity=1.0,
        moving_clim=(0.0, 1.0),
        fid=0,
        mid=1,
        projection_mode="xy",
    )
    napari_layers.create_pair_layers(viewer, fixed_data=fixed, moving_data=moving, **kwargs)
    napari_layers.create_pair_layers(viewer, fixed_data=moving, moving_data=fixed, **kwargs)
    assert len(viewer.layers) == 2


def test_set_overlay_mode_color_removes_composite(make_napari_viewer, sample_arrays) -> None:
    viewer = make_napari_viewer(show=False)
    fixed, moving = sample_arrays
    scale = [0.01, 0.01]
    fixed_layer, moving_layer = napari_layers.create_pair_layers(
        viewer,
        fixed_data=fixed,
        moving_data=moving,
        fixed_scale=scale,
        moving_scale=scale,
        fixed_gamma=0.6,
        fixed_opacity=1.0,
        fixed_clim=(0.0, 1.0),
        moving_gamma=0.6,
        moving_opacity=1.0,
        moving_clim=(0.0, 1.0),
        fid=0,
        mid=1,
        projection_mode="xy",
    )
    composite = viewer.add_image(np.zeros_like(fixed), name="Composite")
    composite.visible = False

    result = napari_layers.set_overlay_mode(
        viewer,
        overlay_mode=OVERLAY_COLOR,
        fixed_layer=fixed_layer,
        moving_layer=moving_layer,
        composite_layer=composite,
        base_image=fixed,
        scale=scale,
    )

    assert result is None
    assert fixed_layer.visible is True
    assert moving_layer.visible is True
    assert len(viewer.layers) == 2


def test_set_overlay_mode_diff_adds_composite(make_napari_viewer, sample_arrays) -> None:
    viewer = make_napari_viewer(show=False)
    fixed, moving = sample_arrays
    scale = [0.01, 0.01]
    fixed_layer, moving_layer = napari_layers.create_pair_layers(
        viewer,
        fixed_data=fixed,
        moving_data=moving,
        fixed_scale=scale,
        moving_scale=scale,
        fixed_gamma=0.6,
        fixed_opacity=1.0,
        fixed_clim=(0.0, 1.0),
        moving_gamma=0.6,
        moving_opacity=1.0,
        moving_clim=(0.0, 1.0),
        fid=0,
        mid=1,
        projection_mode="xy",
    )

    composite = napari_layers.set_overlay_mode(
        viewer,
        overlay_mode=OVERLAY_DIFF,
        fixed_layer=fixed_layer,
        moving_layer=moving_layer,
        composite_layer=None,
        base_image=fixed,
        scale=scale,
    )

    assert composite is not None
    assert composite.name == napari_layers.COMPOSITE_LAYER_NAME
    assert composite.colormap.name == "inferno"
    assert fixed_layer.visible is False
    assert moving_layer.visible is False
    assert len(viewer.layers) == 3


def test_set_overlay_mode_diff_to_checker_updates_colormap_in_place(make_napari_viewer, sample_arrays) -> None:
    viewer = make_napari_viewer(show=False)
    fixed, moving = sample_arrays
    scale = [0.01, 0.01]
    fixed_layer, moving_layer = napari_layers.create_pair_layers(
        viewer,
        fixed_data=fixed,
        moving_data=moving,
        fixed_scale=scale,
        moving_scale=scale,
        fixed_gamma=0.6,
        fixed_opacity=1.0,
        fixed_clim=(0.0, 1.0),
        moving_gamma=0.6,
        moving_opacity=1.0,
        moving_clim=(0.0, 1.0),
        fid=0,
        mid=1,
        projection_mode="xy",
    )
    composite = napari_layers.set_overlay_mode(
        viewer,
        overlay_mode=OVERLAY_DIFF,
        fixed_layer=fixed_layer,
        moving_layer=moving_layer,
        composite_layer=None,
        base_image=fixed,
        scale=scale,
    )
    assert composite is not None
    assert len(viewer.layers) == 3

    updated = napari_layers.set_overlay_mode(
        viewer,
        overlay_mode=OVERLAY_CHECKER,
        fixed_layer=fixed_layer,
        moving_layer=moving_layer,
        composite_layer=composite,
        base_image=fixed,
        scale=scale,
    )

    assert updated is composite
    assert composite.colormap.name == "gray"
    assert len(viewer.layers) == 3


def test_refresh_composite_noop_in_color_mode(make_napari_viewer, sample_arrays) -> None:
    viewer = make_napari_viewer(show=False)
    fixed, moving = sample_arrays
    composite = viewer.add_image(np.zeros_like(fixed), name="Composite")
    before = composite.data.copy()
    napari_layers.refresh_composite(
        viewer,
        overlay_mode=OVERLAY_COLOR,
        composite_layer=composite,
        fixed_base=fixed,
        shifted_moving=moving,
        tile_size=32,
    )
    np.testing.assert_array_equal(composite.data, before)


def test_refresh_composite_pushes_overlay_data(make_napari_viewer, sample_arrays) -> None:
    viewer = make_napari_viewer(show=False)
    fixed, moving = sample_arrays
    composite = viewer.add_image(np.zeros_like(fixed), name="Composite")
    expected = build_overlay(fixed, moving, mode=OVERLAY_DIFF, tile_size=16)
    napari_layers.refresh_composite(
        viewer,
        overlay_mode=OVERLAY_DIFF,
        composite_layer=composite,
        fixed_base=fixed,
        shifted_moving=moving,
        tile_size=16,
    )
    np.testing.assert_allclose(composite.data, expected)


def test_update_pair_data_preserves_layer_count_and_visuals(make_napari_viewer, sample_arrays) -> None:
    viewer = make_napari_viewer(show=False)
    fixed, moving = sample_arrays
    scale = [0.01, 0.01]
    fixed_layer, moving_layer = napari_layers.create_pair_layers(
        viewer,
        fixed_data=fixed,
        moving_data=moving,
        fixed_scale=scale,
        moving_scale=scale,
        fixed_gamma=0.75,
        fixed_opacity=0.85,
        fixed_clim=(0.1, 0.9),
        moving_gamma=0.65,
        moving_opacity=0.95,
        moving_clim=(0.05, 0.95),
        fid=0,
        mid=1,
        projection_mode="xy",
    )
    new_fixed = fixed + 0.1
    new_moving = moving + 0.2
    napari_layers.update_pair_data(
        viewer,
        fixed_layer=fixed_layer,
        moving_layer=moving_layer,
        fixed_data=new_fixed,
        moving_data=new_moving,
    )
    assert len(viewer.layers) == 2
    np.testing.assert_allclose(fixed_layer.data, new_fixed)
    np.testing.assert_allclose(moving_layer.data, new_moving)
    assert fixed_layer.gamma == pytest.approx(0.75)
    assert moving_layer.opacity == pytest.approx(0.95)


def test_assert_layer_inventory_silent_for_valid_states(make_napari_viewer, sample_arrays, caplog) -> None:
    viewer = make_napari_viewer(show=False)
    fixed, moving = sample_arrays
    scale = [0.01, 0.01]
    napari_layers.create_pair_layers(
        viewer,
        fixed_data=fixed,
        moving_data=moving,
        fixed_scale=scale,
        moving_scale=scale,
        fixed_gamma=0.6,
        fixed_opacity=1.0,
        fixed_clim=(0.0, 1.0),
        moving_gamma=0.6,
        moving_opacity=1.0,
        moving_clim=(0.0, 1.0),
        fid=0,
        mid=1,
        projection_mode="xy",
    )
    with caplog.at_level("WARNING"):
        napari_layers.assert_layer_inventory(
            viewer, projection_mode="xy", overlay_mode=OVERLAY_COLOR
        )
    assert not caplog.records


def test_assert_layer_inventory_warns_on_unexpected_count(make_napari_viewer, sample_arrays, caplog) -> None:
    viewer = make_napari_viewer(show=False)
    fixed, moving = sample_arrays
    scale = [0.01, 0.01]
    napari_layers.create_pair_layers(
        viewer,
        fixed_data=fixed,
        moving_data=moving,
        fixed_scale=scale,
        moving_scale=scale,
        fixed_gamma=0.6,
        fixed_opacity=1.0,
        fixed_clim=(0.0, 1.0),
        moving_gamma=0.6,
        moving_opacity=1.0,
        moving_clim=(0.0, 1.0),
        fid=0,
        mid=1,
        projection_mode="xy",
    )
    viewer.add_image(np.zeros_like(fixed), name="Stray")
    with caplog.at_level("WARNING"):
        napari_layers.assert_layer_inventory(
            viewer, projection_mode="xy", overlay_mode=OVERLAY_COLOR
        )
    assert any("Unexpected napari layer" in r.message for r in caplog.records)


def test_set_overlay_mode_z_mode_removes_composite(make_napari_viewer, sample_arrays) -> None:
    viewer = make_napari_viewer(show=False)
    fixed, moving = sample_arrays
    scale = [0.01, 0.01]
    fixed_layer, moving_layer = napari_layers.create_pair_layers(
        viewer,
        fixed_data=fixed,
        moving_data=moving,
        fixed_scale=scale,
        moving_scale=scale,
        fixed_gamma=0.6,
        fixed_opacity=1.0,
        fixed_clim=(0.0, 1.0),
        moving_gamma=0.6,
        moving_opacity=1.0,
        moving_clim=(0.0, 1.0),
        fid=0,
        mid=1,
        projection_mode="xy",
    )
    composite = napari_layers.set_overlay_mode(
        viewer,
        overlay_mode=OVERLAY_DIFF,
        fixed_layer=fixed_layer,
        moving_layer=moving_layer,
        composite_layer=None,
        base_image=fixed,
        scale=scale,
        projection_mode="xy",
    )
    assert len(viewer.layers) == 3

    result = napari_layers.set_overlay_mode(
        viewer,
        overlay_mode=OVERLAY_DIFF,
        fixed_layer=fixed_layer,
        moving_layer=moving_layer,
        composite_layer=composite,
        base_image=fixed,
        scale=scale,
        projection_mode="xz",
    )

    assert result is None
    assert fixed_layer.visible is True
    assert moving_layer.visible is True
    assert len(viewer.layers) == 2


def test_can_incremental_update_gate(make_napari_viewer, sample_arrays) -> None:
    from linumpy_manual_align.ui.widget_pair_loading import PairLoadingMixin

    class _Stub(PairLoadingMixin):
        pass

    widget = object.__new__(_Stub)
    fixed, moving = sample_arrays
    scale = [0.01, 0.01]
    widget.fixed_layer = None
    widget.moving_layer = None
    assert widget._can_incremental_update(
        fixed_aip=fixed, moving_aip=moving, fixed_scale=scale, moving_scale=scale
    ) is False

    viewer = make_napari_viewer(show=False)
    fl, ml = napari_layers.create_pair_layers(
        viewer,
        fixed_data=fixed,
        moving_data=moving,
        fixed_scale=scale,
        moving_scale=scale,
        fixed_gamma=0.6,
        fixed_opacity=1.0,
        fixed_clim=(0.0, 1.0),
        moving_gamma=0.6,
        moving_opacity=1.0,
        moving_clim=(0.0, 1.0),
        fid=0,
        mid=1,
        projection_mode="xy",
    )
    widget.fixed_layer = fl
    widget.moving_layer = ml
    assert widget._can_incremental_update(
        fixed_aip=fixed, moving_aip=moving, fixed_scale=scale, moving_scale=scale
    ) is True
    assert widget._can_incremental_update(
        fixed_aip=fixed[:16, :16],
        moving_aip=moving,
        fixed_scale=scale,
        moving_scale=scale,
    ) is False


def test_xy_z_xy_round_trip_no_orphans(make_napari_viewer, sample_arrays) -> None:
    from linumpy_manual_align.ui.widget_overlay import OverlayStateMixin
    from linumpy_manual_align.ui.widget_pair_loading import PairLoadingMixin

    class _RoundTripWidget(PairLoadingMixin, OverlayStateMixin):
        pass

    viewer = make_napari_viewer(show=False)
    fixed, moving = sample_arrays
    scale = [0.01, 0.01]

    widget = object.__new__(_RoundTripWidget)
    widget.viewer = viewer
    widget._projection_mode = "xy"
    widget._overlay_mode = OVERLAY_DIFF
    widget._original_fixed_aip = fixed
    widget._original_moving_aip = moving
    widget.fixed_layer = None
    widget.moving_layer = None
    widget._composite_layer = None
    widget.spin_tile = type("T", (), {"value": lambda self: 32})()

    widget.fixed_layer, widget.moving_layer = napari_layers.create_pair_layers(
        viewer,
        fixed_data=fixed,
        moving_data=moving,
        fixed_scale=scale,
        moving_scale=scale,
        fixed_gamma=0.6,
        fixed_opacity=1.0,
        fixed_clim=(0.0, 1.0),
        moving_gamma=0.6,
        moving_opacity=1.0,
        moving_clim=(0.0, 1.0),
        fid=0,
        mid=1,
        projection_mode="xy",
    )
    widget._composite_layer = napari_layers.set_overlay_mode(
        viewer,
        overlay_mode=OVERLAY_DIFF,
        fixed_layer=widget.fixed_layer,
        moving_layer=widget.moving_layer,
        composite_layer=None,
        base_image=fixed,
        scale=scale,
        projection_mode="xy",
    )
    assert len(viewer.layers) == 3

    cs_mgr = type("CS", (), {"close": lambda self: None, "readers": {1: object()}})()
    cs_mgr.close_called = False
    original_close = cs_mgr.close

    def _track_close() -> None:
        cs_mgr.close_called = True
        original_close()

    cs_mgr.close = _track_close  # type: ignore[method-assign]
    widget._cs_mgr = cs_mgr

    widget._projection_mode = "xz"
    widget._rebuild_layer_visibility()
    assert widget._composite_layer is None
    assert len(viewer.layers) == 2

    widget._projection_mode = "xy"
    widget._composite_layer = None
    widget.fixed_layer, widget.moving_layer = napari_layers.create_pair_layers(
        viewer,
        fixed_data=fixed,
        moving_data=moving,
        fixed_scale=scale,
        moving_scale=scale,
        fixed_gamma=0.6,
        fixed_opacity=1.0,
        fixed_clim=(0.0, 1.0),
        moving_gamma=0.6,
        moving_opacity=1.0,
        moving_clim=(0.0, 1.0),
        fid=0,
        mid=1,
        projection_mode="xy",
    )
    widget._rebuild_layer_visibility()
    assert len(viewer.layers) == 3
    assert cs_mgr.close_called is False
    napari_layers.assert_layer_inventory(
        viewer, projection_mode="xy", overlay_mode=OVERLAY_DIFF
    )


@pytest.mark.parametrize(
    ("projection_mode", "full_teardown", "preserve_camera", "expected"),
    [
        ("xy", False, True, True),
        ("xy", False, False, True),
        ("xy", True, True, True),
        ("xy", True, False, True),
        ("xz", True, False, True),
        ("xz", True, True, True),
        ("xz", False, True, False),
        ("yz", True, True, True),
    ],
)
def test_should_recenter_on_switch(
    projection_mode: str,
    full_teardown: bool,
    preserve_camera: bool,
    expected: bool,
) -> None:
    assert (
        napari_layers.should_recenter_on_switch(
            projection_mode=projection_mode,
            full_teardown=full_teardown,
            preserve_camera=preserve_camera,
        )
        is expected
    )


class _FakeViewer:
    def __init__(self) -> None:
        self.reset_calls: list[None] = []

    def reset_view(self) -> None:
        self.reset_calls.append(None)


class _FakeCamera:
    def __init__(self, center: tuple[float, ...] = (0.0, 0.0, 0.0), zoom: float = 3.0) -> None:
        self.center = center
        self.zoom = zoom


class _FakeViewerWithCamera:
    def __init__(self, center: tuple[float, ...] = (0.0, 0.0, 0.0), zoom: float = 3.0) -> None:
        self.camera = _FakeCamera(center=center, zoom=zoom)


class _FakeViewerResetCamera:
    def __init__(self, base_zoom: float, full_center: tuple[float, ...] = (0.0, 0.0, 0.0)) -> None:
        self.camera = _FakeCamera(center=full_center, zoom=1.0)
        self.reset_calls: list[None] = []
        self._base_zoom = base_zoom
        self._full_center = full_center

    def reset_view(self) -> None:
        self.reset_calls.append(None)
        self.camera.zoom = self._base_zoom
        self.camera.center = self._full_center


class _FakeImageLayer:
    def __init__(
        self,
        data: np.ndarray,
        scale: tuple[float, float] = (1.0, 1.0),
        translate: tuple[float, float] = (0.0, 0.0),
    ) -> None:
        self.data = data
        self.scale = scale
        self.translate = translate


def _content_block_array(rows: int, cols: int, col_start: int, col_end: int) -> np.ndarray:
    arr = np.zeros((rows, cols), dtype=np.float32)
    arr[:, col_start:col_end] = 1.0
    return arr


class _RecenterStub:
    pass


def test_maybe_recenter_after_load_recenters_in_xy() -> None:
    from linumpy_manual_align.ui.widget_pair_loading import PairLoadingMixin

    class _Stub(PairLoadingMixin, _RecenterStub):
        pass

    widget = object.__new__(_Stub)
    widget._projection_mode = "xy"
    widget.viewer = _FakeViewer()

    widget._maybe_recenter_after_load(full_teardown=False, preserve_camera=True)
    widget._maybe_recenter_after_load(full_teardown=True, preserve_camera=True)

    assert len(widget.viewer.reset_calls) == 2


def test_maybe_recenter_after_load_z_recenters_on_full_teardown_only() -> None:
    from linumpy_manual_align.ui.widget_pair_loading import PairLoadingMixin

    class _Stub(PairLoadingMixin, _RecenterStub):
        pass

    widget = object.__new__(_Stub)
    widget._projection_mode = "xz"
    widget.viewer = _FakeViewer()

    widget._maybe_recenter_after_load(full_teardown=False, preserve_camera=True)
    assert len(widget.viewer.reset_calls) == 0

    widget._maybe_recenter_after_load(full_teardown=True, preserve_camera=True)
    assert len(widget.viewer.reset_calls) == 1


def test_maybe_recenter_after_load_recenters_on_z_full_teardown() -> None:
    from linumpy_manual_align.ui.widget_pair_loading import PairLoadingMixin

    class _Stub(PairLoadingMixin, _RecenterStub):
        pass

    widget = object.__new__(_Stub)
    widget._projection_mode = "xz"
    widget.viewer = _FakeViewer()

    widget._maybe_recenter_after_load(full_teardown=True, preserve_camera=False)

    assert len(widget.viewer.reset_calls) == 1


def test_maybe_recenter_after_load_recenters_on_incremental_xy() -> None:
    from linumpy_manual_align.ui.widget_pair_loading import PairLoadingMixin

    class _Stub(PairLoadingMixin, _RecenterStub):
        pass

    widget = object.__new__(_Stub)
    widget._projection_mode = "xy"
    widget.viewer = _FakeViewer()

    widget._maybe_recenter_after_load(full_teardown=False, preserve_camera=True)
    widget._maybe_recenter_after_load(full_teardown=False, preserve_camera=False)

    assert len(widget.viewer.reset_calls) == 2


def test_load_pair_preserve_camera_skips_restore_in_xy() -> None:
    from linumpy_manual_align.ui.widget_interaction import InteractionMixin

    class _Stub(InteractionMixin):
        pass

    widget = object.__new__(_Stub)
    widget._projection_mode = "xy"
    widget.viewer = type("Cam", (), {"camera": type("C", (), {"zoom": 1.0, "center": (0.0, 0.0)})()})()
    load_calls: list[tuple[int, bool]] = []
    restore_calls: list[tuple[float, tuple]] = []

    def _load_pair(idx: int, preserve_camera: bool = False) -> None:
        load_calls.append((idx, preserve_camera))

    def _restore_camera(zoom: float, center: tuple) -> None:
        restore_calls.append((zoom, center))

    widget._load_pair = _load_pair  # type: ignore[method-assign]
    widget._restore_camera = _restore_camera  # type: ignore[method-assign]

    widget._load_pair_preserve_camera(3)

    assert load_calls == [(3, False)]
    assert restore_calls == []


def test_load_pair_preserve_camera_skips_restore_in_z() -> None:
    from linumpy_manual_align.ui.widget_interaction import InteractionMixin

    class _Stub(InteractionMixin):
        pass

    widget = object.__new__(_Stub)
    widget._projection_mode = "xz"
    widget.viewer = type("Cam", (), {"camera": type("C", (), {"zoom": 2.5, "center": (1.0, 2.0)})()})()
    load_calls: list[tuple[int, bool]] = []
    restore_calls: list[tuple[float, tuple]] = []

    def _load_pair(idx: int, preserve_camera: bool = False) -> None:
        load_calls.append((idx, preserve_camera))

    def _restore_camera(zoom: float, center: tuple) -> None:
        restore_calls.append((zoom, center))

    widget._load_pair = _load_pair  # type: ignore[method-assign]
    widget._restore_camera = _restore_camera  # type: ignore[method-assign]

    widget._load_pair_preserve_camera(1)

    assert load_calls == [(1, False)]
    assert restore_calls == []


def test_xy_z_xy_round_trip_recenters() -> None:
    from linumpy_manual_align.ui.widget_pair_loading import PairLoadingMixin

    class _Stub(PairLoadingMixin, _RecenterStub):
        pass

    widget = object.__new__(_Stub)
    widget.viewer = _FakeViewer()

    widget._projection_mode = "xy"
    widget._maybe_recenter_after_load(full_teardown=False, preserve_camera=True)
    assert len(widget.viewer.reset_calls) == 1

    widget._projection_mode = "xz"
    assert (
        napari_layers.should_recenter_on_switch(
            projection_mode="xz", full_teardown=False, preserve_camera=True
        )
        is False
    )
    widget._maybe_recenter_after_load(full_teardown=False, preserve_camera=True)
    assert len(widget.viewer.reset_calls) == 1

    widget._projection_mode = "xy"
    widget._maybe_recenter_after_load(full_teardown=True, preserve_camera=True)
    assert len(widget.viewer.reset_calls) == 2


@pytest.mark.parametrize(
    ("projection_mode", "prev_shape", "new_shape", "expected"),
    [
        ("xz", (10, 20), (10, 25), True),
        ("yz", (10, 20), (12, 20), True),
        ("xz", (10, 20), (10, 20), True),
        ("xy", (10, 20), (10, 25), False),
    ],
)
def test_should_recenter_after_cross_section(
    projection_mode: str,
    prev_shape: tuple[int, int],
    new_shape: tuple[int, int],
    expected: bool,
) -> None:
    assert (
        napari_layers.should_recenter_after_cross_section(
            projection_mode=projection_mode,
            prev_shape=prev_shape,
            new_shape=new_shape,
        )
        is expected
    )


def test_content_center_world_single_layer_tracks_content_column() -> None:
    left = _content_block_array(20, 40, 2, 9)
    right = _content_block_array(20, 40, 30, 39)

    left_center = napari_layers.content_center_world([(left, (1.0, 1.0), (0.0, 0.0))])
    right_center = napari_layers.content_center_world([(right, (1.0, 1.0), (0.0, 0.0))])

    assert left_center is not None
    assert right_center is not None
    assert left_center[1] < right_center[1]


def test_content_center_world_applies_scale_and_translate() -> None:
    data = np.zeros((20, 20), dtype=np.float32)
    data[5:11, 4:9] = 1.0
    scale = (2.0, 2.0)
    translate = (10.0, 5.0)

    center = napari_layers.content_center_world([(data, scale, translate)])

    assert center is not None
    r1, c1, r2, c2 = 5, 4, 11, 9
    expected = (
        ((r1 * scale[0] + translate[0]) + (r2 * scale[0] + translate[0])) / 2.0,
        ((c1 * scale[1] + translate[1]) + (c2 * scale[1] + translate[1])) / 2.0,
    )
    assert center == pytest.approx(expected)


def test_content_center_world_combines_layers() -> None:
    left = _content_block_array(20, 40, 2, 9)
    right = _content_block_array(20, 40, 30, 39)

    left_only = napari_layers.content_center_world([(left, (1.0, 1.0), (0.0, 0.0))])
    right_only = napari_layers.content_center_world([(right, (1.0, 1.0), (0.0, 0.0))])
    combined = napari_layers.content_center_world(
        [(left, (1.0, 1.0), (0.0, 0.0)), (right, (1.0, 1.0), (0.0, 0.0))]
    )

    assert left_only is not None
    assert right_only is not None
    assert combined is not None
    assert left_only[1] < combined[1] < right_only[1]


def test_content_center_world_none_when_blank() -> None:
    blank = np.zeros((10, 10), dtype=np.float32)
    assert napari_layers.content_center_world([]) is None
    assert napari_layers.content_center_world([(blank, (1.0, 1.0), (0.0, 0.0))]) is None


def test_content_fit_zoom_factor() -> None:
    half = _content_block_array(40, 1000, 250, 750)
    assert napari_layers.content_fit_zoom_factor((half, (1.0, 1.0), (0.0, 0.0))) == pytest.approx(2.0)

    narrow = _content_block_array(40, 1000, 400, 500)
    assert napari_layers.content_fit_zoom_factor((narrow, (1.0, 1.0), (0.0, 0.0))) == pytest.approx(10.0)
    assert napari_layers.content_fit_zoom_factor((narrow, (0.01, 0.01), (0.0, 0.0))) == pytest.approx(10.0)

    tiny = _content_block_array(40, 1000, 495, 505)
    assert napari_layers.content_fit_zoom_factor((tiny, (1.0, 1.0), (0.0, 0.0))) == pytest.approx(20.0)

    full = _content_block_array(40, 1000, 0, 1000)
    assert napari_layers.content_fit_zoom_factor((full, (1.0, 1.0), (0.0, 0.0))) == pytest.approx(1.0)

    blank = np.zeros((10, 10), dtype=np.float32)
    assert napari_layers.content_fit_zoom_factor((blank, (1.0, 1.0), (0.0, 0.0))) is None


def test_content_fit_after_load_z_centers_and_zooms() -> None:
    from linumpy_manual_align.ui.widget_pair_loading import PairLoadingMixin

    class _Stub(PairLoadingMixin, _RecenterStub):
        pass

    moving_data = _content_block_array(40, 1351, 244, 340)
    triple = (moving_data, (1.0, 1.0), (0.0, 0.0))
    expected_center = napari_layers.content_center_world([triple])
    expected_factor = napari_layers.content_fit_zoom_factor(triple)

    widget = object.__new__(_Stub)
    widget._projection_mode = "xz"
    widget.moving_layer = _FakeImageLayer(moving_data)
    widget.viewer = _FakeViewerResetCamera(base_zoom=0.5, full_center=(0.0, 0.0, 675.0))

    widget._maybe_recenter_after_load(full_teardown=True, preserve_camera=False)

    assert len(widget.viewer.reset_calls) == 1
    assert expected_center is not None
    assert expected_factor is not None
    assert widget.viewer.camera.center[-2] == pytest.approx(expected_center[0])
    assert widget.viewer.camera.center[-1] == pytest.approx(expected_center[1])
    assert widget.viewer.camera.zoom == pytest.approx(0.5 * expected_factor)
    assert widget.viewer.camera.zoom > 0.5


def test_content_fit_after_load_z_blank_frame_leaves_reset() -> None:
    from linumpy_manual_align.ui.widget_pair_loading import PairLoadingMixin

    class _Stub(PairLoadingMixin, _RecenterStub):
        pass

    widget = object.__new__(_Stub)
    widget._projection_mode = "xz"
    widget.moving_layer = _FakeImageLayer(np.zeros((40, 1351), dtype=np.float32))
    widget.viewer = _FakeViewerResetCamera(base_zoom=0.5, full_center=(0.0, 0.0, 675.0))

    widget._maybe_recenter_after_load(full_teardown=True, preserve_camera=False)

    assert len(widget.viewer.reset_calls) == 1
    assert widget.viewer.camera.zoom == 0.5
    assert widget.viewer.camera.center == (0.0, 0.0, 675.0)


def test_maybe_recenter_after_load_xy_still_reset_only() -> None:
    from linumpy_manual_align.ui.widget_pair_loading import PairLoadingMixin

    class _Stub(PairLoadingMixin, _RecenterStub):
        pass

    widget = object.__new__(_Stub)
    widget._projection_mode = "xy"
    widget.viewer = _FakeViewer()

    widget._maybe_recenter_after_load(full_teardown=True, preserve_camera=False)

    assert len(widget.viewer.reset_calls) == 1


def test_maybe_recenter_after_cross_section_pans_to_content_in_z() -> None:
    from linumpy_manual_align.ui.widget_cross_section import CrossSectionMixin

    class _Stub(CrossSectionMixin):
        pass

    fixed_data = _content_block_array(20, 40, 10, 18)
    moving_data = _content_block_array(20, 40, 12, 20)
    expected = napari_layers.content_center_world(
        [(moving_data, (1.0, 1.0), (0.0, 0.0))]
    )

    widget = object.__new__(_Stub)
    widget._projection_mode = "xz"
    widget.fixed_layer = _FakeImageLayer(fixed_data)
    widget.moving_layer = _FakeImageLayer(moving_data)
    widget.viewer = _FakeViewerWithCamera(center=(0.0, 0.0, 50.0), zoom=3.0)

    widget._maybe_recenter_after_cross_section((10, 20), (10, 20))

    assert expected is not None
    assert widget.viewer.camera.center[-2] == pytest.approx(expected[0])
    assert widget.viewer.camera.center[-1] == pytest.approx(expected[1])
    assert widget.viewer.camera.zoom == 3.0


def test_cross_section_same_shape_scroll_pans_follows_content() -> None:
    from linumpy_manual_align.ui.widget_cross_section import CrossSectionMixin

    class _Stub(CrossSectionMixin):
        pass

    widget = object.__new__(_Stub)
    widget._projection_mode = "xz"
    widget.fixed_layer = _FakeImageLayer(np.zeros((20, 40), dtype=np.float32))
    widget.moving_layer = _FakeImageLayer(_content_block_array(20, 40, 2, 9))
    widget.viewer = _FakeViewerWithCamera(center=(0.0, 0.0, 0.0), zoom=2.5)

    widget._maybe_recenter_after_cross_section((20, 40), (20, 40))
    col_left = widget.viewer.camera.center[-1]

    widget.moving_layer.data = _content_block_array(20, 40, 30, 39)
    widget._maybe_recenter_after_cross_section((20, 40), (20, 40))
    col_right = widget.viewer.camera.center[-1]

    assert col_right > col_left
    assert widget.viewer.camera.zoom == 2.5


def test_maybe_recenter_after_cross_section_noop_in_xy() -> None:
    from linumpy_manual_align.ui.widget_cross_section import CrossSectionMixin

    class _Stub(CrossSectionMixin):
        pass

    widget = object.__new__(_Stub)
    widget._projection_mode = "xy"
    widget.fixed_layer = _FakeImageLayer(_content_block_array(20, 40, 10, 18))
    widget.moving_layer = _FakeImageLayer(_content_block_array(20, 40, 12, 20))
    initial_center = (0.0, 0.0, 50.0)
    widget.viewer = _FakeViewerWithCamera(center=initial_center, zoom=3.0)

    widget._maybe_recenter_after_cross_section((10, 20), (10, 25))

    assert widget.viewer.camera.center == initial_center


def test_cross_section_wide_static_fixed_shifting_moving_tracks_moving_only() -> None:
    from linumpy_manual_align.ui.widget_cross_section import CrossSectionMixin

    class _Stub(CrossSectionMixin):
        pass

    scale = (1.0, 1.0)
    translate = (0.0, 0.0)
    fixed_data = _content_block_array(40, 1351, 256, 1094)
    moving_f1 = _content_block_array(40, 1351, 600, 700)
    moving_f2 = _content_block_array(40, 1351, 700, 800)

    widget = object.__new__(_Stub)
    widget._projection_mode = "xz"
    widget.fixed_layer = _FakeImageLayer(fixed_data, scale=scale, translate=translate)
    widget.moving_layer = _FakeImageLayer(moving_f1, scale=scale, translate=translate)
    widget.viewer = _FakeViewerWithCamera(center=(0.0, 0.0, 0.0), zoom=4.0)

    moving_center_f1 = napari_layers.content_center_world([(moving_f1, scale, translate)])
    union_center_f1 = napari_layers.content_center_world(
        [(fixed_data, scale, translate), (moving_f1, scale, translate)]
    )

    widget._maybe_recenter_after_cross_section((40, 1351), (40, 1351))
    col1 = widget.viewer.camera.center[-1]

    moving_center_f2 = napari_layers.content_center_world([(moving_f2, scale, translate)])
    union_center_f2 = napari_layers.content_center_world(
        [(fixed_data, scale, translate), (moving_f2, scale, translate)]
    )

    widget.moving_layer.data = moving_f2
    widget._maybe_recenter_after_cross_section((40, 1351), (40, 1351))
    col2 = widget.viewer.camera.center[-1]

    assert moving_center_f1 is not None
    assert moving_center_f2 is not None
    assert union_center_f1 is not None
    assert union_center_f2 is not None
    assert col1 == pytest.approx(moving_center_f1[1])
    assert col2 == pytest.approx(moving_center_f2[1])

    moving_delta = abs(moving_center_f2[1] - moving_center_f1[1])
    union_delta = abs(union_center_f2[1] - union_center_f1[1])
    camera_delta = abs(col2 - col1)

    assert camera_delta == pytest.approx(moving_delta)
    assert union_delta < moving_delta
    assert widget.viewer.camera.zoom == 4.0
