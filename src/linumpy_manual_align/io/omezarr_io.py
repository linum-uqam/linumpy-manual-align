"""OME-Zarr I/O helpers."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def load_aip_from_ome_zarr(zarr_path: Path, level: int) -> tuple[np.ndarray, list[float]]:
    """Load a volume from OME-Zarr and compute its AIP.

    Returns
    -------
    tuple[np.ndarray, list[float]]
        AIP image and YX scale.
    """
    try:
        from ome_zarr.io import parse_url
        from ome_zarr.reader import Reader
    except ImportError as exc:
        raise RuntimeError(
            "OME-Zarr reading requires the optional 'ome-zarr' dependency. Install with: uv pip install -e '.[zarr]'"
        ) from exc

    loc = parse_url(str(zarr_path), mode="r")
    if loc is None:
        raise ValueError(f"Could not parse OME-Zarr location: {zarr_path}")

    nodes = list(Reader(loc)())
    if not nodes:
        raise ValueError(f"No OME-Zarr nodes found: {zarr_path}")

    node = nodes[0]
    if level >= len(node.data):
        raise ValueError(f"Requested level {level} not found in OME-Zarr: {zarr_path}")

    vol = np.asarray(node.data[level], dtype=np.float32)

    # Keep only ZYX dimensions. If extra leading dimensions exist (e.g. C, T),
    # take index 0 for those axes.
    while vol.ndim > 3:
        vol = vol[0]
    if vol.ndim != 3:
        raise ValueError(f"Expected 3D ZYX data in OME-Zarr level {level}, got shape {vol.shape}")

    scale = [1.0, 1.0, 1.0]

    # Primary path: Reader-provided per-level coordinate transformations.
    ct_by_level = node.metadata.get("coordinateTransformations", [])
    if level < len(ct_by_level):
        transforms = ct_by_level[level]
        if isinstance(transforms, dict):
            transforms = [transforms]
        for transform in transforms:
            if transform.get("type") == "scale":
                vals = transform.get("scale", [])
                if len(vals) >= 3:
                    scale = [float(vals[-3]), float(vals[-2]), float(vals[-1])]
                break

    # Fallback path: root multiscales metadata.
    if scale == [1.0, 1.0, 1.0]:
        root_attrs = dict(getattr(node.zarr, "root_attrs", {}))
        multiscales = root_attrs.get("multiscales", [])
        datasets = multiscales[0].get("datasets", []) if multiscales else []
        if level < len(datasets):
            for transform in datasets[level].get("coordinateTransformations", []):
                if transform.get("type") == "scale":
                    vals = transform.get("scale", [])
                    if len(vals) >= 3:
                        scale = [float(vals[-3]), float(vals[-2]), float(vals[-1])]
                    break

    aip = vol.mean(axis=0).astype(np.float32)
    return aip, scale[1:]
