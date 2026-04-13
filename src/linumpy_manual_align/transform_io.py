"""Transform I/O helpers for manual alignment tool.

Loads and saves SimpleITK Euler3DTransform .tfm files and companion
pairwise_registration_metrics.json files, compatible with the linumpy
stacking pipeline (linum_stack_slices_motor.py).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import SimpleITK as sitk


def load_transform(tfm_path: Path) -> tuple[float, float, float, tuple[float, float]]:
    """Load a .tfm file and return (tx, ty, rotation_deg, center_xy).

    Parameters
    ----------
    tfm_path : Path
        Path to the .tfm file.

    Returns
    -------
    tuple[float, float, float, tuple[float, float]]
        (tx, ty, rotation_deg, (center_x, center_y)) in pixels and degrees.
    """
    tfm = sitk.ReadTransform(str(tfm_path))
    params = tfm.GetParameters()
    # params: [rx, ry, rz, tx, ty, tz] for Euler3DTransform
    rotation_deg = float(np.degrees(params[2]))
    tx = float(params[3])
    ty = float(params[4])
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

    Also writes a companion pairwise_registration_metrics.json with
    source="manual" so the stacking pipeline can identify these.

    Parameters
    ----------
    output_dir : Path
        Directory to write into (e.g. manual_transforms/slice_z04/).
    tx, ty : float
        Translation in pixels at the *working* resolution (pyramid level).
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
    full_tx = tx * scale
    full_ty = ty * scale
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
    import re

    pattern = re.compile(r"slice_z(\d+)")
    slices = {}
    for p in sorted(input_dir.iterdir()):
        m = pattern.search(p.name)
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
    import re

    pattern = re.compile(r"slice_z(\d+)")
    transforms = {}
    for p in sorted(transforms_dir.iterdir()):
        if p.is_dir():
            m = pattern.search(p.name)
            if m:
                tfm_files = list(p.glob("*.tfm"))
                if tfm_files:
                    transforms[int(m.group(1))] = p
    return dict(sorted(transforms.items()))
