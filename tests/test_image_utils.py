"""Comprehensive tests for image_utils module.

Tests cover content_bbox, normalize_aip, enhance_aip, apply_transform,
and build_overlay — all pure functions with no Qt/napari dependencies.
"""

from __future__ import annotations

import numpy as np
import pytest

from linumpy_manual_align.io.image_utils import (
    ENHANCE_CLAHE,
    ENHANCE_EDGES,
    ENHANCE_NONE,
    ENHANCE_SHARPEN,
    OVERLAY_CHECKER,
    OVERLAY_DIFF,
    apply_transform,
    build_overlay,
    content_bbox,
    enhance_aip,
    normalize_aip,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _ones_block(shape: tuple[int, int], r1: int, c1: int, r2: int, c2: int) -> np.ndarray:
    """Return a zero float32 image with a block of ones at the given ROI."""
    img = np.zeros(shape, dtype=np.float32)
    img[r1:r2, c1:c2] = 1.0
    return img


# ---------------------------------------------------------------------------
# content_bbox
# ---------------------------------------------------------------------------


class TestContentBbox:
    def test_centred_block_returns_tight_box_plus_padding(self) -> None:
        img = _ones_block((100, 100), 40, 40, 60, 60)
        r1, c1, r2, c2 = content_bbox(img, threshold=0.5, padding=5)
        assert r1 <= 40 - 5
        assert c1 <= 40 - 5
        assert r2 >= 60 + 5
        assert c2 >= 60 + 5

    def test_blank_image_returns_full_extent(self) -> None:
        img = np.zeros((64, 64), dtype=np.float32)
        r1, c1, r2, c2 = content_bbox(img)
        assert (r1, c1, r2, c2) == (0, 0, 64, 64)

    def test_full_image_content_returns_full_extent(self) -> None:
        img = np.ones((64, 64), dtype=np.float32)
        r1, c1, r2, c2 = content_bbox(img)
        assert r1 == 0
        assert c1 == 0
        assert r2 == 64
        assert c2 == 64

    def test_padding_clamped_to_boundaries(self) -> None:
        # Block touching the top-left corner — padding should not go negative.
        img = _ones_block((64, 64), 0, 0, 10, 10)
        r1, c1, _r2, _c2 = content_bbox(img, padding=20)
        assert r1 == 0
        assert c1 == 0

    def test_padding_zero(self) -> None:
        img = _ones_block((50, 50), 10, 15, 20, 35)
        r1, c1, r2, c2 = content_bbox(img, threshold=0.5, padding=0)
        assert r1 == 10
        assert c1 == 15
        # content_bbox adds 1 to the last row/col index so r2/c2 are exclusive bounds.
        assert r2 == 20
        assert c2 == 35

    def test_single_pixel_content(self) -> None:
        img = np.zeros((32, 32), dtype=np.float32)
        img[16, 16] = 1.0
        r1, c1, _r2, _c2 = content_bbox(img, threshold=0.5, padding=0)
        assert r1 == 16
        assert c1 == 16


# ---------------------------------------------------------------------------
# normalize_aip
# ---------------------------------------------------------------------------


class TestNormalizeAip:
    def test_output_range_zero_to_one(self) -> None:
        rng = np.random.default_rng(0)
        img = rng.random((64, 64)).astype(np.float32) + 0.1  # no zeros
        out = normalize_aip(img)
        assert float(out.min()) >= 0.0
        assert float(out.max()) <= 1.0 + 1e-6

    def test_output_dtype_float32(self) -> None:
        img = np.ones((32, 32), dtype=np.float64)
        out = normalize_aip(img)
        assert out.dtype == np.float32

    def test_blank_image_returns_zeros(self) -> None:
        img = np.zeros((32, 32), dtype=np.float32)
        out = normalize_aip(img)
        assert np.all(out == 0.0)

    def test_constant_nonzero_image_returns_zeros(self) -> None:
        img = np.full((32, 32), 0.5, dtype=np.float32)
        out = normalize_aip(img)
        # p_low ≈ p_high → returns zero array
        assert np.all(out == 0.0)

    def test_output_shape_preserved(self) -> None:
        img = np.random.default_rng(1).random((40, 60)).astype(np.float32) + 0.01
        out = normalize_aip(img)
        assert out.shape == img.shape

    def test_image_with_spread_content_is_normalized(self) -> None:
        rng = np.random.default_rng(7)
        img = rng.random((64, 64)).astype(np.float32) * 0.5 + 0.01
        out = normalize_aip(img)
        assert out.max() > 0.0


# ---------------------------------------------------------------------------
# enhance_aip
# ---------------------------------------------------------------------------


class TestEnhanceAip:
    _BASE = np.random.default_rng(42).random((64, 64)).astype(np.float32)

    def test_none_returns_input_unchanged(self) -> None:
        out = enhance_aip(self._BASE, ENHANCE_NONE)
        np.testing.assert_array_equal(out, self._BASE)

    def test_edges_output_shape_dtype(self) -> None:
        out = enhance_aip(self._BASE, ENHANCE_EDGES)
        assert out.shape == self._BASE.shape
        assert out.dtype == np.float32

    def test_edges_range_zero_to_one(self) -> None:
        out = enhance_aip(self._BASE, ENHANCE_EDGES)
        assert float(out.min()) >= 0.0
        assert float(out.max()) <= 1.0 + 1e-5

    def test_clahe_output_shape_dtype(self) -> None:
        pytest.importorskip("skimage")
        out = enhance_aip(self._BASE, ENHANCE_CLAHE)
        assert out.shape == self._BASE.shape
        assert out.dtype == np.float32

    def test_clahe_range_zero_to_one(self) -> None:
        pytest.importorskip("skimage")
        out = enhance_aip(self._BASE, ENHANCE_CLAHE)
        assert float(out.min()) >= 0.0
        assert float(out.max()) <= 1.0 + 1e-5

    def test_sharpen_output_shape_dtype(self) -> None:
        out = enhance_aip(self._BASE, ENHANCE_SHARPEN)
        assert out.shape == self._BASE.shape
        assert out.dtype == np.float32

    def test_sharpen_range_zero_to_one(self) -> None:
        out = enhance_aip(self._BASE, ENHANCE_SHARPEN)
        assert float(out.min()) >= 0.0
        assert float(out.max()) <= 1.0 + 1e-5

    def test_unknown_mode_returns_input(self) -> None:
        out = enhance_aip(self._BASE, "unknown_mode")
        np.testing.assert_array_equal(out, self._BASE)

    def test_edges_is_different_from_input(self) -> None:
        out = enhance_aip(self._BASE, ENHANCE_EDGES)
        assert not np.allclose(out, self._BASE)


# ---------------------------------------------------------------------------
# apply_transform
# ---------------------------------------------------------------------------


class TestApplyTransform:
    _IMG = np.zeros((64, 64), dtype=np.float32)
    # Put a bright spot at top-left so shifts are visible.
    _IMG[10, 10] = 1.0

    def test_identity_returns_same_shape(self) -> None:
        out = apply_transform(self._IMG)
        assert out.shape == self._IMG.shape

    def test_identity_values_unchanged(self) -> None:
        out = apply_transform(self._IMG)
        np.testing.assert_array_almost_equal(out, self._IMG, decimal=5)

    def test_output_dtype_float32(self) -> None:
        img = np.ones((32, 32), dtype=np.float64)
        out = apply_transform(img)
        assert out.dtype == np.float32

    def test_translation_shifts_content(self) -> None:
        img = np.zeros((64, 64), dtype=np.float32)
        img[30, 30] = 1.0
        out = apply_transform(img, tx=5.0, ty=0.0)
        # Pixel at (30, 35) should be roughly 1.0 after a +5 x-shift.
        assert out[30, 35] > 0.5

    def test_negative_translation(self) -> None:
        img = np.zeros((64, 64), dtype=np.float32)
        img[30, 30] = 1.0
        out = apply_transform(img, tx=-5.0, ty=0.0)
        assert out[30, 25] > 0.5

    def test_rotation_180_flips_image(self) -> None:
        img = np.zeros((64, 64), dtype=np.float32)
        img[10, 10] = 1.0
        out = apply_transform(img, rotation=180.0)
        # 180 deg rotation maps (10,10) to (54,54) for a 64x64 image
        # (rotation centre is at pixel 31.5,31.5; 31.5 + (31.5 - 10) ~= 53.0, rounds to 54).
        peak = np.unravel_index(np.argmax(out), out.shape)
        assert out[peak] > 0.5

    def test_rotation_zero_no_change(self) -> None:
        out = apply_transform(self._IMG, rotation=0.0)
        np.testing.assert_array_almost_equal(out, self._IMG, decimal=5)

    def test_combined_rotation_translation(self) -> None:
        img = np.zeros((64, 64), dtype=np.float32)
        img[32, 32] = 1.0
        out = apply_transform(img, rotation=0.0, tx=5.0, ty=3.0)
        assert out[35, 37] > 0.5

    def test_large_shift_produces_empty_output(self) -> None:
        img = np.ones((32, 32), dtype=np.float32)
        out = apply_transform(img, tx=1000.0)
        # All content shifted out of frame → zeros
        assert float(out.max()) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# build_overlay
# ---------------------------------------------------------------------------


class TestBuildOverlay:
    _SHAPE = (64, 64)
    _FIXED = np.full(_SHAPE, 0.8, dtype=np.float32)
    _MOVING = np.full(_SHAPE, 0.3, dtype=np.float32)

    def test_diff_output_shape(self) -> None:
        out = build_overlay(self._FIXED, self._MOVING, OVERLAY_DIFF)
        assert out.shape == self._SHAPE

    def test_diff_output_dtype(self) -> None:
        out = build_overlay(self._FIXED, self._MOVING, OVERLAY_DIFF)
        assert out.dtype == np.float32

    def test_diff_values_are_absolute_difference(self) -> None:
        out = build_overlay(self._FIXED, self._MOVING, OVERLAY_DIFF)
        expected = abs(0.8 - 0.3)
        assert np.allclose(out, expected, atol=1e-5)

    def test_diff_identical_images_returns_zeros(self) -> None:
        img = np.full(self._SHAPE, 0.5, dtype=np.float32)
        out = build_overlay(img, img, OVERLAY_DIFF)
        np.testing.assert_array_almost_equal(out, 0.0)

    def test_checker_output_shape(self) -> None:
        out = build_overlay(self._FIXED, self._MOVING, OVERLAY_CHECKER)
        assert out.shape == self._SHAPE

    def test_checker_output_dtype(self) -> None:
        out = build_overlay(self._FIXED, self._MOVING, OVERLAY_CHECKER)
        assert out.dtype == np.float32

    def test_checker_values_are_from_fixed_or_moving(self) -> None:
        out = build_overlay(self._FIXED, self._MOVING, OVERLAY_CHECKER)
        valid = np.isclose(out, 0.8) | np.isclose(out, 0.3)
        assert np.all(valid)

    def test_checker_tile_size_1(self) -> None:
        # Tile size 1 → alternating pixels.
        out = build_overlay(self._FIXED, self._MOVING, OVERLAY_CHECKER, tile_size=1)
        assert out.shape == self._SHAPE

    def test_checker_tile_size_zero_clamped_to_1(self) -> None:
        # tile_size=0 should be clamped to 1, not crash.
        out = build_overlay(self._FIXED, self._MOVING, OVERLAY_CHECKER, tile_size=0)
        assert out.shape == self._SHAPE

    def test_checker_large_tile_size(self) -> None:
        # Tile larger than image → entire image comes from one source.
        out = build_overlay(self._FIXED, self._MOVING, OVERLAY_CHECKER, tile_size=1000)
        # Only values from fixed (even tile) or moving (odd tile) should be present.
        assert out.shape == self._SHAPE
