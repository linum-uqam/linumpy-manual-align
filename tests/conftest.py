"""Shared test fixtures for linumpy-manual-align."""

from __future__ import annotations

import os
import shutil
from pathlib import Path

import numpy as np
import pytest

# Ensure Qt can start on headless CI runners (no-op when a real display exists).
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


@pytest.fixture(scope="session")
def qapp():
    """Session-scoped QApplication instance, required for QObject-based tests."""
    from napari.qt import get_qapp

    return get_qapp()


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


@pytest.fixture(scope="session")
def fixtures_root() -> Path:
    """Path to committed test fixtures under tests/fixtures/."""
    return Path(__file__).parent / "fixtures"


@pytest.fixture
def copy_fixture_tree(tmp_path: Path, fixtures_root: Path):
    """Return a callable that copies a committed fixture tree into tmp_path."""

    def _copy(name: str, dest: Path | None = None) -> Path:
        src = fixtures_root / name
        target = dest or tmp_path / name
        shutil.copytree(src, target)
        return target

    return _copy
