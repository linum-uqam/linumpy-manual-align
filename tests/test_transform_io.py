"""Tests for transform_io module: save/load round-trip, discovery helpers."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from linumpy_manual_align.transform_io import (
    discover_aips,
    discover_pair_aips,
    discover_slices,
    discover_transforms,
    get_metric,
    load_aip_from_npz,
    load_offsets,
    load_pairwise_metrics,
    load_transform,
    save_transform,
)


class TestSaveLoadRoundTrip:
    """Verify that saving and loading a transform preserves the parameters."""

    def test_identity_transform(self, tmp_path: Path) -> None:
        out_dir = tmp_path / "slice_z00"
        save_transform(out_dir, tx=0.0, ty=0.0, rotation_deg=0.0, center=(100.0, 100.0), level=0)

        tx, ty, rot, center = load_transform(out_dir / "transform.tfm")
        assert abs(tx) < 1e-6
        assert abs(ty) < 1e-6
        assert abs(rot) < 1e-6
        assert abs(center[0] - 100.0) < 1e-6
        assert abs(center[1] - 100.0) < 1e-6

    def test_translation_only(self, tmp_path: Path) -> None:
        out_dir = tmp_path / "slice_z01"
        save_transform(out_dir, tx=10.5, ty=-7.3, rotation_deg=0.0, center=(256.0, 256.0), level=0)

        tx, ty, rot, _center = load_transform(out_dir / "transform.tfm")
        assert abs(tx - 10.5) < 1e-4
        assert abs(ty - (-7.3)) < 1e-4
        assert abs(rot) < 1e-4

    def test_rotation_only(self, tmp_path: Path) -> None:
        out_dir = tmp_path / "slice_z02"
        save_transform(out_dir, tx=0.0, ty=0.0, rotation_deg=2.5, center=(128.0, 128.0), level=0)

        _tx, _ty, rot, _center = load_transform(out_dir / "transform.tfm")
        # With rotation around non-origin center, tx/ty may be nonzero in the .tfm params
        # but the rotation should be preserved exactly
        assert abs(rot - 2.5) < 1e-4

    def test_full_transform(self, tmp_path: Path) -> None:
        out_dir = tmp_path / "slice_z03"
        save_transform(out_dir, tx=15.0, ty=-20.0, rotation_deg=1.7, center=(200.0, 150.0), level=0)

        tx, ty, rot, _center = load_transform(out_dir / "transform.tfm")
        assert abs(tx - 15.0) < 1e-4
        assert abs(ty - (-20.0)) < 1e-4
        assert abs(rot - 1.7) < 1e-4

    def test_level_scaling(self, tmp_path: Path) -> None:
        """Working-resolution values are scaled by 2^level when saved."""
        out_dir = tmp_path / "slice_z04"
        save_transform(out_dir, tx=5.0, ty=-3.0, rotation_deg=0.5, center=(100.0, 100.0), level=2)

        tx, ty, rot, _center = load_transform(out_dir / "transform.tfm")
        # tx/ty should be 4x the working-resolution values (level=2 → 2^2=4)
        assert abs(tx - 20.0) < 1e-4
        assert abs(ty - (-12.0)) < 1e-4
        assert abs(rot - 0.5) < 1e-4


class TestSaveOutputFiles:
    """Verify that save_transform writes all expected companion files."""

    def test_output_files_exist(self, tmp_path: Path) -> None:
        out_dir = tmp_path / "slice_z05"
        save_transform(out_dir, tx=1.0, ty=2.0, rotation_deg=0.0, center=(50.0, 50.0), level=0)

        assert (out_dir / "transform.tfm").exists()
        assert (out_dir / "offsets.txt").exists()
        assert (out_dir / "pairwise_registration_metrics.json").exists()

    def test_metrics_json_content(self, tmp_path: Path) -> None:
        out_dir = tmp_path / "slice_z06"
        save_transform(out_dir, tx=3.0, ty=4.0, rotation_deg=1.0, center=(50.0, 50.0), level=1)

        metrics = json.loads((out_dir / "pairwise_registration_metrics.json").read_text())
        assert metrics["source"] == "manual"
        assert metrics["overall_status"] == "ok"
        assert metrics["manual_alignment"]["pyramid_level"] == 1
        assert abs(metrics["manual_alignment"]["working_tx"] - 3.0) < 1e-6
        # Full-res values should be 2x working (level=1)
        assert abs(metrics["metrics"]["translation_x"]["value"] - 6.0) < 1e-6


class TestLoadPairwiseMetrics:
    def test_nonexistent_returns_empty(self, tmp_path: Path) -> None:
        result = load_pairwise_metrics(tmp_path / "nonexistent.json")
        assert result == {}

    def test_valid_json(self, tmp_path: Path) -> None:
        path = tmp_path / "metrics.json"
        path.write_text(json.dumps({"source": "manual", "metrics": {"mag": 10}}))
        result = load_pairwise_metrics(path)
        assert result["source"] == "manual"


class TestDiscoverSlices:
    def test_discovers_ome_zarr(self, tmp_path: Path) -> None:
        # Create fake slice directories
        for i in [0, 1, 5, 10]:
            (tmp_path / f"slice_z{i:02d}.ome.zarr").mkdir()
        # Non-matching files should be ignored
        (tmp_path / "other_file.txt").touch()
        (tmp_path / "slice_z99.nii.gz").touch()

        slices = discover_slices(tmp_path)
        assert list(slices.keys()) == [0, 1, 5, 10]

    def test_empty_directory(self, tmp_path: Path) -> None:
        assert discover_slices(tmp_path) == {}


class TestDiscoverTransforms:
    def test_discovers_transform_dirs(self, tmp_path: Path) -> None:
        for i in [3, 7, 12]:
            d = tmp_path / f"slice_z{i:02d}"
            d.mkdir()
            (d / "transform.tfm").touch()
        # Dir without .tfm should be ignored
        (tmp_path / "slice_z99").mkdir()

        transforms = discover_transforms(tmp_path)
        assert list(transforms.keys()) == [3, 7, 12]

    def test_empty_directory(self, tmp_path: Path) -> None:
        assert discover_transforms(tmp_path) == {}


class TestLoadOffsets:
    def test_nonexistent_returns_zeros(self, tmp_path: Path) -> None:
        result = load_offsets(tmp_path / "offsets.txt")
        assert result == (0, 0)

    def test_valid_offsets(self, tmp_path: Path) -> None:
        offsets_path = tmp_path / "offsets.txt"
        np.savetxt(str(offsets_path), [5, 12], fmt="%d")
        result = load_offsets(offsets_path)
        assert result == (5, 12)

    def test_offsets_roundtrip(self, tmp_path: Path) -> None:
        """Offsets saved by save_transform can be loaded by load_offsets."""
        out_dir = tmp_path / "slice_z07"
        save_transform(out_dir, tx=1.0, ty=2.0, rotation_deg=0.0, center=(50.0, 50.0), level=0, offsets=(3, 7))
        result = load_offsets(out_dir / "offsets.txt")
        assert result == (3, 7)


# ---------------------------------------------------------------------------
# discover_aips
# ---------------------------------------------------------------------------


class TestDiscoverAips:
    def test_discovers_npz_files(self, tmp_path: Path) -> None:
        for i in [0, 3, 7]:
            (tmp_path / f"slice_z{i:02d}.npz").touch()
        # Files that don't match the pattern should be ignored.
        (tmp_path / "other.npz").touch()
        (tmp_path / "slice_z00.ome.zarr").mkdir()

        aips = discover_aips(tmp_path)
        assert list(aips.keys()) == [0, 3, 7]

    def test_returns_correct_paths(self, tmp_path: Path) -> None:
        p = tmp_path / "slice_z05.npz"
        p.touch()
        aips = discover_aips(tmp_path)
        assert aips[5] == p

    def test_empty_directory(self, tmp_path: Path) -> None:
        assert discover_aips(tmp_path) == {}

    def test_sorted_order(self, tmp_path: Path) -> None:
        for i in [10, 2, 7]:
            (tmp_path / f"slice_z{i:02d}.npz").touch()
        aips = discover_aips(tmp_path)
        assert list(aips.keys()) == [2, 7, 10]


# ---------------------------------------------------------------------------
# discover_pair_aips
# ---------------------------------------------------------------------------


class TestDiscoverPairAips:
    def test_discovers_paired_files(self, tmp_path: Path) -> None:
        (tmp_path / "pair_z00_z01_fixed.npz").touch()
        (tmp_path / "pair_z00_z01_moving.npz").touch()
        result = discover_pair_aips(tmp_path)
        assert (0, 1) in result
        assert "fixed" in result[(0, 1)]
        assert "moving" in result[(0, 1)]

    def test_ignores_non_pair_files(self, tmp_path: Path) -> None:
        (tmp_path / "slice_z00.npz").touch()
        (tmp_path / "pair_z00_z01_fixed.npz").touch()
        result = discover_pair_aips(tmp_path)
        assert (0, 1) in result
        assert len(result) == 1

    def test_multiple_pairs(self, tmp_path: Path) -> None:
        for fid, mid in [(0, 1), (1, 2), (5, 6)]:
            for role in ("fixed", "moving"):
                (tmp_path / f"pair_z{fid:02d}_z{mid:02d}_{role}.npz").touch()
        result = discover_pair_aips(tmp_path)
        assert len(result) == 3

    def test_empty_directory(self, tmp_path: Path) -> None:
        assert discover_pair_aips(tmp_path) == {}


# ---------------------------------------------------------------------------
# load_aip_from_npz
# ---------------------------------------------------------------------------


class TestLoadAipFromNpz:
    def _write_npz(self, path: Path, aip: np.ndarray, scale: np.ndarray) -> None:
        np.savez(str(path), aip=aip, scale=scale)

    def test_basic_load_yx_scale(self, tmp_path: Path) -> None:
        aip = np.ones((32, 32), dtype=np.float32)
        scale = np.array([1.0, 0.5])
        p = tmp_path / "slice_z00.npz"
        self._write_npz(p, aip, scale)
        loaded_aip, loaded_scale = load_aip_from_npz(p)
        assert loaded_aip.shape == (32, 32)
        assert loaded_aip.dtype == np.float32
        assert loaded_scale == [1.0, 0.5]

    def test_3d_scale_drops_z(self, tmp_path: Path) -> None:
        """When scale has 3 elements (Z, Y, X), Z is dropped."""
        aip = np.zeros((16, 16), dtype=np.uint16)
        scale = np.array([3.0, 1.5, 1.5])
        p = tmp_path / "slice_z01.npz"
        self._write_npz(p, aip, scale)
        _, loaded_scale = load_aip_from_npz(p)
        assert len(loaded_scale) == 2
        assert loaded_scale == [1.5, 1.5]

    def test_output_dtype_is_float32(self, tmp_path: Path) -> None:
        aip = np.ones((8, 8), dtype=np.uint8)
        p = tmp_path / "slice_z02.npz"
        self._write_npz(p, aip, np.array([1.0, 1.0]))
        loaded_aip, _ = load_aip_from_npz(p)
        assert loaded_aip.dtype == np.float32


# ---------------------------------------------------------------------------
# get_metric
# ---------------------------------------------------------------------------


class TestGetMetric:
    def test_returns_float_for_valid_key(self) -> None:
        metrics = {"metrics": {"translation_magnitude": {"value": 42.7}}}
        assert get_metric(metrics, "translation_magnitude") == pytest.approx(42.7)

    def test_returns_none_for_missing_key(self) -> None:
        metrics = {"metrics": {}}
        assert get_metric(metrics, "nonexistent") is None

    def test_returns_none_for_empty_dict(self) -> None:
        assert get_metric({}, "any_key") is None

    def test_returns_none_for_non_numeric_value(self) -> None:
        metrics = {"metrics": {"bad": {"value": "not_a_number"}}}
        assert get_metric(metrics, "bad") is None

    def test_returns_none_when_value_key_missing(self) -> None:
        metrics = {"metrics": {"tx": {"other_field": 5.0}}}
        assert get_metric(metrics, "tx") is None

    def test_handles_integer_value(self) -> None:
        metrics = {"metrics": {"rotation": {"value": 3}}}
        val = get_metric(metrics, "rotation")
        assert val == pytest.approx(3.0)
        assert isinstance(val, float)
