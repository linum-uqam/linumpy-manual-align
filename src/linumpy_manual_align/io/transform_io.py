"""Transform I/O helpers for manual alignment tool.

Loads and saves SimpleITK Euler3DTransform .tfm files and companion
pairwise_registration_metrics.json files, compatible with the linumpy
stacking pipeline (linum_stack_slices_motor.py).

Also provides AIP discovery and loading helpers shared between the widget
and the CLI (discover_aips, discover_pair_aips, load_aip_from_npz).
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import SimpleITK as sitk

# Shared regex for matching slice_z##  directory / file names.
_SLICE_PATTERN = re.compile(r"slice_z(\d+)")


def adjust_for_rotation_center(
    tx: float,
    ty: float,
    rot_deg: float,
    tfm_center: tuple[float, float],
    img_center: tuple[float, float],
) -> tuple[float, float]:
    """Correct ``(tx, ty)`` for a mismatch between a transform's rotation centre and the image centre.

    The linumpy stacking pipeline saves transforms whose rotation centre is the
    image centre at the full resolution used during registration. When that
    centre differs from the image centre at the current display level, the
    translation must be adjusted so that the visual result is the same.

    Inputs and outputs are in the widget's content-shift convention (matching
    :func:`load_transform` / :func:`save_transform`). The adjustment is the
    negation of the sitk point-map derivation ``t_sitk' = t_sitk + (I - R) * dc``
    because widget ``tx`` is the negation of sitk ``tx``.

    Parameters
    ----------
    tx, ty:
        Translation in widget content-shift convention.
    rot_deg:
        Rotation angle in degrees.
    tfm_center:
        ``(cx, cy)`` rotation centre stored in the transform file (pixels).
    img_center:
        ``(cx, cy)`` centre of the image as currently displayed (pixels).

    Returns
    -------
    tuple[float, float]
        Adjusted ``(tx, ty)`` in widget convention.
    """
    if abs(rot_deg) <= 0.01:
        return tx, ty
    dcx = tfm_center[0] - img_center[0]
    dcy = tfm_center[1] - img_center[1]
    rad = np.radians(rot_deg)
    cos_r, sin_r = np.cos(rad), np.sin(rad)
    # Signs flipped relative to the sitk-convention derivation because widget
    # tx is -sitk_tx.
    tx -= (1.0 - cos_r) * dcx + sin_r * dcy
    ty -= -sin_r * dcx + (1.0 - cos_r) * dcy
    return tx, ty


def load_transform(tfm_path: Path) -> tuple[float, float, float, tuple[float, float]]:
    """Load a .tfm file and return (tx, ty, rotation_deg, center_xy).

    The returned ``tx, ty`` are in the widget-internal content-shift
    convention (positive ``tx`` means content moves to the right, matching
    napari ``layer.translate`` and ``scipy.ndimage.shift``). This is the
    inverse of the SimpleITK output->input (point-map) convention used on
    disk and by the rest of the linumpy pipeline
    (``linum_register_pairwise.py``,
    ``linumpy.stitching.stacking.apply_2d_transform``, ...), so the values
    are negated when read.

    Save/load is symmetric around the widget convention:
    ``load_transform(path_returned_by_save_transform(x)) == x``.

    Parameters
    ----------
    tfm_path : Path
        Path to the .tfm file.

    Returns
    -------
    tuple[float, float, float, tuple[float, float]]
        (tx, ty, rotation_deg, (center_x, center_y)) in widget content-shift
        convention (pixels and degrees).
    """
    tfm = sitk.ReadTransform(str(tfm_path))
    params = tfm.GetParameters()
    # params: [rx, ry, rz, tx, ty, tz] for Euler3DTransform
    rotation_deg = float(np.degrees(params[2]))
    # Convert SimpleITK (on-disk) convention to widget content-shift convention.
    tx = -float(params[3])
    ty = -float(params[4])
    fixed_params = tfm.GetFixedParameters()
    cx = float(fixed_params[0]) if len(fixed_params) > 0 else 0.0
    cy = float(fixed_params[1]) if len(fixed_params) > 1 else 0.0
    return tx, ty, rotation_deg, (cx, cy)


def load_offsets(offsets_path: Path) -> tuple[int, int]:
    """Load Z-overlap offsets from an offsets.txt file.

    Parameters
    ----------
    offsets_path : Path
        Path to offsets.txt.

    Returns
    -------
    tuple[int, int]
        (fixed_z, moving_z) offsets, or (0, 0) if file doesn't exist.
    """
    if not offsets_path.exists():
        return (0, 0)
    vals = np.loadtxt(str(offsets_path), dtype=int)
    if vals.size >= 2:
        return (int(vals[0]), int(vals[1]))
    return (0, 0)


def save_transform(
    output_dir: Path,
    tx: float,
    ty: float,
    rotation_deg: float,
    center: tuple[float, float],
    level: int = 0,
    offsets: tuple[int, int] = (0, 0),
) -> Path:
    """Save a manual alignment transform as a .tfm file.

    Translation values are scaled by 2^level to convert from working
    resolution to full resolution pixels. Rotation is scale-invariant.

    The inputs ``tx, ty`` are expressed in the widget's internal
    scipy-shift convention (positive means content moves to the right /
    down in napari's display). The on-disk ``.tfm`` file uses SimpleITK's
    output->input (point-map) convention, which is the negation. This
    function converts by negating ``tx, ty`` when writing so that
    downstream consumers that apply the stored transform via
    ``sitk.ResampleImageFilter`` (e.g.
    ``linumpy.stitching.stacking.apply_2d_transform``) produce the visual
    alignment the user saw in the widget.

    Also writes a companion pairwise_registration_metrics.json with
    source="manual" so the stacking pipeline can identify these. The
    ``translation_x / translation_y`` fields in the metrics JSON are
    likewise stored in SimpleITK convention to match the .tfm file.

    Parameters
    ----------
    output_dir : Path
        Directory to write into (e.g. manual_transforms/slice_z04/).
    tx, ty : float
        Translation in pixels at the *working* resolution (pyramid level),
        in widget/scipy-shift convention.
    rotation_deg : float
        Rotation in degrees.
    center : tuple[float, float]
        (cx, cy) rotation center at working resolution.
    level : int
        Pyramid level used for alignment (0 = full res, 1 = 2x, ...).
    offsets : tuple[int, int]
        Z-overlap offsets (fixed_z, moving_z) to carry over from automated transform.

    Returns
    -------
    Path
        Path to the written .tfm file.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    scale = 2**level
    # Widget-internal tx/ty are in scipy-shift convention; the on-disk tfm
    # must be in SimpleITK convention (negated) so downstream sitk.Resample
    # consumers reproduce the widget's visual alignment.
    full_tx = -tx * scale
    full_ty = -ty * scale
    full_cx = center[0] * scale
    full_cy = center[1] * scale

    transform = sitk.Euler3DTransform()
    transform.SetCenter([full_cx, full_cy, 0.0])
    transform.SetRotation(0.0, 0.0, np.radians(rotation_deg))
    transform.SetTranslation([full_tx, full_ty, 0.0])

    tfm_path = output_dir / "transform.tfm"
    sitk.WriteTransform(transform, str(tfm_path))

    # Write companion offsets.txt (carries over automated offsets or zeros for new transforms)
    offsets_path = output_dir / "offsets.txt"
    np.savetxt(str(offsets_path), list(offsets), fmt="%d")

    # Write metrics JSON
    mag = float(np.sqrt(full_tx**2 + full_ty**2))
    metrics = {
        "step_name": "pairwise_registration",
        "output_path": str(output_dir),
        "source": "manual",
        "metrics": {
            # Stored in SimpleITK convention to match the .tfm file.
            "translation_x": {"value": full_tx, "unit": "pixels"},
            "translation_y": {"value": full_ty, "unit": "pixels"},
            "translation_magnitude": {"value": mag, "unit": "pixels"},
            "rotation": {"value": rotation_deg, "unit": "degrees"},
            "registration_confidence": {"value": 1.0},
            "z_correlation": {"value": 1.0},
            "registration_error": {"value": 0.0},
        },
        "overall_status": "ok",
        "manual_alignment": {
            "pyramid_level": level,
            # Widget-internal (scipy-shift) values for display/debug.
            "working_tx": tx,
            "working_ty": ty,
            "center_working": list(center),
        },
    }
    metrics_path = output_dir / "pairwise_registration_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2))

    return tfm_path


def load_pairwise_metrics(metrics_path: Path) -> dict:
    """Load pairwise registration metrics JSON.

    Returns
    -------
    dict
        Parsed metrics dict, or empty dict if file doesn't exist.
    """
    if not metrics_path.exists():
        return {}
    return json.loads(metrics_path.read_text())


def discover_slices(input_dir: Path) -> dict[int, Path]:
    """Discover common-space slice files by pattern.

    Parameters
    ----------
    input_dir : Path
        Directory containing slice_z##.ome.zarr files.

    Returns
    -------
    dict[int, Path]
        Ordered mapping from slice ID to path.
    """
    slices = {}
    for p in sorted(input_dir.iterdir()):
        m = _SLICE_PATTERN.search(p.name)
        if m and p.name.endswith(".ome.zarr"):
            slices[int(m.group(1))] = p
    return dict(sorted(slices.items()))


def discover_transforms(transforms_dir: Path) -> dict[int, Path]:
    """Discover existing pairwise transform directories.

    Parameters
    ----------
    transforms_dir : Path
        Directory containing slice_z## subdirectories with .tfm files.

    Returns
    -------
    dict[int, Path]
        Mapping from slice ID to transform directory.
    """
    transforms = {}
    for p in sorted(transforms_dir.iterdir()):
        if p.is_dir():
            m = _SLICE_PATTERN.search(p.name)
            if m:
                tfm_files = list(p.glob("*.tfm"))
                if tfm_files:
                    transforms[int(m.group(1))] = p
    return dict(sorted(transforms.items()))


def discover_aips(aips_dir: Path) -> dict[int, Path]:
    """Discover pre-computed AIP .npz files in *aips_dir*.

    Parameters
    ----------
    aips_dir : Path
        Directory containing ``slice_z##.npz`` files.

    Returns
    -------
    dict[int, Path]
        Ordered mapping from slice ID to .npz path.
    """
    aips: dict[int, Path] = {}
    for p in sorted(aips_dir.iterdir()):
        m = _SLICE_PATTERN.search(p.name)
        if m and p.name.endswith(".npz"):
            aips[int(m.group(1))] = p
    return dict(sorted(aips.items()))


def discover_pair_aips(aips_dir: Path) -> dict[tuple[int, int], dict[str, Path]]:
    """Discover paired XZ/YZ NPZ files (``pair_z{fid:02d}_z{mid:02d}_{role}.npz``).

    Parameters
    ----------
    aips_dir : Path
        Directory containing paired ``pair_z##_z##_fixed.npz`` / ``…_moving.npz`` files.

    Returns
    -------
    dict[tuple[int, int], dict[str, Path]]
        ``{(fid, mid): {"fixed": Path, "moving": Path}}``.
    """
    pair_pattern = re.compile(r"pair_z(\d+)_z(\d+)_(fixed|moving)\.npz$")
    result: dict[tuple[int, int], dict[str, Path]] = {}
    for p in sorted(aips_dir.iterdir()):
        m = pair_pattern.match(p.name)
        if m:
            key = (int(m.group(1)), int(m.group(2)))
            result.setdefault(key, {})[m.group(3)] = p
    return result


def load_aip_from_npz(npz_path: Path) -> tuple[np.ndarray, list[float]]:
    """Load a pre-computed AIP from an .npz file.

    Parameters
    ----------
    npz_path : Path
        Path to an .npz file containing ``aip`` and ``scale`` arrays.

    Returns
    -------
    tuple[np.ndarray, list[float]]
        ``(aip, scale_yx)`` where *aip* is float32 and *scale_yx* is a
        two-element list ``[sy, sx]``.
    """
    data = np.load(str(npz_path))
    aip = data["aip"].astype(np.float32)
    scale = data["scale"]
    scale_yx = list(scale[1:]) if len(scale) == 3 else list(scale)
    return aip, [float(v) for v in scale_yx]


def get_metric(metrics: dict, key: str) -> float | None:
    """Extract a scalar value from a pairwise-registration metrics dict.

    Parameters
    ----------
    metrics : dict
        Dict as returned by :func:`load_pairwise_metrics`.
    key : str
        Key inside ``metrics["metrics"]``, e.g. ``"translation_magnitude"``.

    Returns
    -------
    float or None
        The ``"value"`` field, or *None* if the key is absent or malformed.
    """
    try:
        return float(metrics["metrics"][key]["value"])
    except (KeyError, TypeError, ValueError):
        return None
