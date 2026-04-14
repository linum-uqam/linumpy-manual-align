"""Image processing helpers for the manual alignment widget.

All functions are pure (no Qt / napari dependencies) so they can be
unit-tested without a display.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter, sobel
from scipy.ndimage import rotate as ndimage_rotate
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
    """Return *moving* with rotation and sub-pixel translation baked in.

    Parameters
    ----------
    moving:
        Source image, shape (H, W).
    rotation:
        Counter-clockwise rotation in degrees.
    tx:
        Horizontal pixel shift (positive = right).
    ty:
        Vertical pixel shift (positive = down).

    Returns
    -------
    np.ndarray
        Transformed image, same shape as *moving*.
    """
    out = moving
    if abs(rotation) > 0.01:
        out = ndimage_rotate(out, -rotation, reshape=False, order=1, mode="constant", cval=0)
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
