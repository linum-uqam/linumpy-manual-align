"""Tests for OME-Zarr loading helpers."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from linumpy_manual_align.io.omezarr_io import load_aip_from_ome_zarr


def _create_minimal_ome_zarr(path: Path, data: np.ndarray, scale_zyx: list[float]) -> None:
    zarr = pytest.importorskip("zarr")

    root = zarr.open_group(str(path), mode="w")
    root.create_array("0", data=data, chunks=data.shape)
    root.attrs["multiscales"] = [
        {
            "version": "0.4",
            "axes": [
                {"name": "z", "type": "space"},
                {"name": "y", "type": "space"},
                {"name": "x", "type": "space"},
            ],
            "datasets": [
                {
                    "path": "0",
                    "coordinateTransformations": [
                        {
                            "type": "scale",
                            "scale": scale_zyx,
                        }
                    ],
                }
            ],
        }
    ]


class TestLoadAipFromOmeZarr:
    def test_loads_aip_and_scale(self, tmp_path: Path) -> None:
        pytest.importorskip("ome_zarr")

        vol = np.arange(3 * 4 * 5, dtype=np.float32).reshape(3, 4, 5)
        zarr_path = tmp_path / "slice_z00.ome.zarr"
        _create_minimal_ome_zarr(zarr_path, vol, [2.0, 0.4, 0.6])

        aip, scale_yx = load_aip_from_ome_zarr(zarr_path, level=0)

        assert aip.shape == (4, 5)
        assert np.allclose(aip, vol.mean(axis=0))
        assert scale_yx == [0.4, 0.6]

    def test_raises_on_missing_level(self, tmp_path: Path) -> None:
        pytest.importorskip("ome_zarr")

        vol = np.ones((2, 3, 4), dtype=np.float32)
        zarr_path = tmp_path / "slice_z00.ome.zarr"
        _create_minimal_ome_zarr(zarr_path, vol, [1.0, 1.0, 1.0])

        with pytest.raises(ValueError, match="Requested level"):
            load_aip_from_ome_zarr(zarr_path, level=3)
