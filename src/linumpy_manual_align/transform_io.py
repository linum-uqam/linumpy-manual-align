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


def load_transform(tfm_path: Path) -> tuple[float, float, float]:
    """Load a .tfm file and return (tx, ty, rotation_deg).

    Parameters
    ----------
    tfm_path : Path
        Path to the .tfm file.

    Returns
    -------
    tuple[float, float, float]
        (tx, ty, rotation_deg) in pixels and degrees.
    """
    tfm = sitk.ReadTransform(str(tfm_path))
    params = tfm.GetParameters()
    # params: [rx, ry, rz, tx, ty, tz] for Euler3DTransform
    rotation_deg = float(np.degrees(params[2]))
    tx = float(params[3])
    ty = float(params[4])
    return tx, ty, rotation_deg


def save_transform(
    output_dir: Path,
    tx: float,
    ty: float,
    rotation_deg: float,
    center: tuple[float, float],
    level: int = 0,
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

    # Write companion offsets.txt (zeros — manual transforms don't have Z-offset info)
    offsets_path = output_dir / "offsets.txt"
    np.savetxt(str(offsets_path), [0, 0], fmt="%d")

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
