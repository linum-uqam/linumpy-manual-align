"""Image processing helpers for the manual alignment widget.

All functions are pure (no Qt / napari dependencies) so they can be
unit-tested without a display.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import affine_transform, gaussian_filter, sobel
from scipy.ndimage import shift as ndimage_shift

# Overlay mode identifiers (kept here so widget and tests share the same constants)
OVERLAY_COLOR = "color"
OVERLAY_DIFF = "diff"
OVERLAY_CHECKER = "checker"

# Enhancement mode identifiers
ENHANCE_NONE = "none"
ENHANCE_EDGES = "edges"
ENHANCE_CLAHE = "clahe"
ENHANCE_SHARPEN = "sharpen"


def content_bbox(
    img: np.ndarray,
    threshold: float = 0.02,
    padding: int = 20,
) -> tuple[int, int, int, int]:
    """Return ``(r1, c1, r2, c2)`` tight bounding box of non-zero content.

    A uniform padding of *padding* pixels is added on every side (clamped to
    the image boundaries).  Returns the full image extent when no content is
    detected.
    """
    mask = img > threshold
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    if not rows.any() or not cols.any():
        return 0, 0, img.shape[0], img.shape[1]
    r1 = max(0, int(np.where(rows)[0][0]) - padding)
    r2 = min(img.shape[0], int(np.where(rows)[0][-1]) + padding + 1)
    c1 = max(0, int(np.where(cols)[0][0]) - padding)
    c2 = min(img.shape[1], int(np.where(cols)[0][-1]) + padding + 1)
    return r1, c1, r2, c2


def normalize_aip(img: np.ndarray) -> np.ndarray:
    """Stretch *img* to [0, 1] using the 1st / 99th percentile of non-zero values.

    Returns a zero array when the image is blank or constant.
    """
    p_low = float(np.percentile(img[img > 0], 1)) if np.any(img > 0) else 0.0
    p_high = float(np.percentile(img, 99))
    if p_high <= p_low:
        return np.zeros_like(img, dtype=np.float32)
    return np.clip((img - p_low) / (p_high - p_low), 0, 1).astype(np.float32)


def enhance_aip(img: np.ndarray, mode: str) -> np.ndarray:
    """Apply a visual enhancement to a normalized [0, 1] float32 AIP.

    Each mode targets a different alignment difficulty:

    ``ENHANCE_EDGES``
        Sobel gradient magnitude.  Converts the image to a map of tissue
        boundaries regardless of absolute intensity.  Ideal for oblique cuts
        where tissue edges are the clearest alignment landmarks.

    ``ENHANCE_CLAHE``
        Contrast-Limited Adaptive Histogram Equalization (CLAHE).  Equalises
        local contrast so that both bright core regions and dim periphery are
        visible at the same time.  Helps with projection blur where the tissue
        top and bottom occupy different intensities.

    ``ENHANCE_SHARPEN``
        Unsharp mask (Gaussian high-pass added back to the image).  A mild
        crispening that sharpens blurry features without the binary look of
        edge detection.  Good as a general-purpose enhancement.

    Parameters
    ----------
    img : np.ndarray
        Normalized [0, 1] float32 2-D array.
    mode : str
        One of the ``ENHANCE_*`` constants.

    Returns
    -------
    np.ndarray
        Enhanced image, normalized to [0, 1], same shape and dtype as *img*.
    """
    if mode == ENHANCE_NONE:
        return img

    if mode == ENHANCE_EDGES:
        gx = sobel(img, axis=1)
        gy = sobel(img, axis=0)
        edges = np.hypot(gx, gy).astype(np.float32)
        return normalize_aip(edges)

    if mode == ENHANCE_CLAHE:
        from skimage.exposure import equalize_adapthist

        return equalize_adapthist(img, clip_limit=0.03).astype(np.float32)

    if mode == ENHANCE_SHARPEN:
        blurred = gaussian_filter(img, sigma=2.0)
        # Unsharp mask: boost by 1.5x the high-frequency detail
        sharpened = img + 1.5 * (img - blurred)
        return np.clip(sharpened, 0, 1).astype(np.float32)

    return img


def apply_transform(
    moving: np.ndarray,
    *,
    rotation: float = 0.0,
    tx: float = 0.0,
    ty: float = 0.0,
) -> np.ndarray:
    """Return *moving* with rotation and translation baked into pixel data.

    Used exclusively for the composite overlay modes (Difference, Checkerboard)
    where a single combined image must be computed.

    Rotation is performed about pixel ``(H/2 - ty, W/2 - tx)`` so the
    world-space pivot stays at the image centre after the ``(ty, tx)``
    translation is applied.

    Parameters
    ----------
    moving:
        Source image, shape (H, W).
    rotation:
        Counter-clockwise rotation in degrees.
    tx, ty:
        Pixel shift at the current pyramid level.

    Returns
    -------
    np.ndarray
        Transformed image, same shape as *moving*.
    """
    out: np.ndarray = moving
    if abs(rotation) > 0.01:
        angle_rad = np.radians(-rotation)
        cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
        # affine_transform maps output→input; use R(-rotation) for CCW display rotation.
        m = np.array([[cos_a, sin_a], [-sin_a, cos_a]])
        # Always rotate about the image centre (H/2, W/2) regardless of tx/ty.
        # The shift step below then moves the brain to its translated position.
        cy = moving.shape[0] / 2.0
        cx = moving.shape[1] / 2.0
        c = np.array([cy, cx])
        out = affine_transform(out, m, offset=c - m @ c, mode="constant", cval=0, order=1).astype(np.float32)
    if abs(tx) > 1e-6 or abs(ty) > 1e-6:
        out = ndimage_shift(out, [ty, tx], order=1, mode="constant", cval=0)
    return out.astype(np.float32)


def build_overlay(
    fixed: np.ndarray,
    shifted_moving: np.ndarray,
    mode: str,
    tile_size: int = 16,
) -> np.ndarray:
    """Combine *fixed* and *shifted_moving* into a single composite image.

    Parameters
    ----------
    fixed, shifted_moving:
        Normalized [0, 1] float32 images of the same shape.
    mode:
        ``OVERLAY_DIFF`` — absolute difference (bright = misaligned).
        ``OVERLAY_CHECKER`` — spatial checkerboard alternating tiles.
    tile_size:
        Edge length in pixels of each checkerboard tile (ignored for
        ``OVERLAY_DIFF``).

    Returns
    -------
    np.ndarray
        Composite float32 image, shape identical to inputs.
    """
    if mode == OVERLAY_DIFF:
        return np.abs(fixed - shifted_moving).astype(np.float32)

    # Checkerboard
    tile_size = max(1, tile_size)
    rows, cols = fixed.shape[:2]
    ri = np.arange(rows)
    ci = np.arange(cols)
    mask = ((ri[:, None] // tile_size) + (ci[None, :] // tile_size)) % 2 == 0
    return np.where(mask, fixed, shifted_moving).astype(np.float32)
