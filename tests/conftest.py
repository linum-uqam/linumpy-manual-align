"""Shared test fixtures for linumpy-manual-align."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest


@pytest.fixture
def fake_data_package(tmp_path: Path) -> Path:
    """Create a minimal fake data package with AIPs and transforms."""
    pkg = tmp_path / "manual_align_package"
    aips = pkg / "aips"
    aips.mkdir(parents=True)
    transforms = pkg / "transforms"
    transforms.mkdir(parents=True)

    # Create fake AIP .npz files for 3 consecutive slices
    for i in range(3):
        aip = np.random.default_rng(i).random((64, 64)).astype(np.float32)
        scale = np.array([1.0, 0.01, 0.01])
        np.savez(str(aips / f"slice_z{i:02d}.npz"), aip=aip, scale=scale)

    return pkg
