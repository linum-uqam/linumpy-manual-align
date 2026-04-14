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


def build_moving_affine(
    rotation_deg: float,
    tx: float,
    ty: float,
    scale_yx: tuple[float, float] | list[float],
    shape_hw: tuple[int, int],
) -> np.ndarray:
    """Build the complete 3x3 data→world affine for the moving napari layer.

    Encodes: scale → rotate CCW by *rotation_deg* about the image centre →
    translate by *(ty, tx)* pixels.  Setting this matrix on ``layer.affine``
    means no pixel resampling is ever required; napari renders the transform
    natively in hardware.

    Parameters
    ----------
    rotation_deg:
        Counter-clockwise rotation in degrees.
    tx, ty:
        Translation in pixels at the current pyramid level
        (positive tx = rightward, positive ty = downward).
    scale_yx:
        Physical pixel spacing ``(scale_y, scale_x)`` at the working level.
    shape_hw:
        Image shape ``(H, W)`` at the working level.

    Returns
    -------
    np.ndarray
        3x3 homogeneous affine matrix (data coords → world coords).
    """
    sy, sx = float(scale_yx[0]), float(scale_yx[1])
    H, W = shape_hw

    # Rotation centre = image centre in world space (at zero translation)
    cy_w = H / 2.0 * sy
    cx_w = W / 2.0 * sx

    theta = np.radians(rotation_deg)
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)

    # World-space translation
    ty_w = ty * sy
    tx_w = tx * sx

    # Full data→world transform:
    #   world_row = cos*r*sy  - sin*c*sx  + (1-cos)*cy_w + sin*cx_w + ty_w
    #   world_col = sin*r*sy  + cos*c*sx  - sin*cy_w + (1-cos)*cx_w + tx_w
    return np.array(
        [
            [cos_t * sy, -sin_t * sx, (1.0 - cos_t) * cy_w + sin_t * cx_w + ty_w],
            [sin_t * sy, cos_t * sx, -sin_t * cy_w + (1.0 - cos_t) * cx_w + tx_w],
            [0.0, 0.0, 1.0],
        ]
    )


def apply_transform(
    moving: np.ndarray,
    *,
    rotation: float = 0.0,
    tx: float = 0.0,
    ty: float = 0.0,
) -> np.ndarray:
    """Return *moving* with rotation and translation baked into pixel data.

    Used exclusively for the composite overlay modes (Difference, Checkerboard)
    where a single combined image must be computed.  For the direct-layer
    display path, use ``build_moving_affine`` instead — no pixel resampling.

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
        cy = moving.shape[0] / 2.0 - ty
        cx = moving.shape[1] / 2.0 - tx
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
