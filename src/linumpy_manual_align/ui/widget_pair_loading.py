"""AIP discovery and napari layer loading."""

from __future__ import annotations

import contextlib
import logging
import re
from pathlib import Path

import numpy as np

from linumpy_manual_align.io.image_utils import (
    content_bbox,
    enhance_aip,
    normalize_aip,
)
from linumpy_manual_align.io.omezarr_io import load_aip_from_ome_zarr
from linumpy_manual_align.io.transform_io import (
    adjust_for_rotation_center,
    discover_aips,
    discover_pair_aips,
    load_aip_from_npz,
    load_offsets,
    load_transform,
)
from linumpy_manual_align.state import AlignmentState, UndoStack
from linumpy_manual_align.ui.widget_typing import ManualAlignWidget

logger = logging.getLogger(__name__)


class PairLoadingMixin:
    def _refresh_saved_pairs(self: ManualAlignWidget) -> None:
        """Scan output_dir and pre-populate saved_pairs with already-written transforms."""
        if not self.output_dir.exists():
            return
        pattern = re.compile(r"slice_z(\d+)$")
        for p in self.output_dir.iterdir():
            m = pattern.match(p.name)
            if m and (p / "transform.tfm").exists():
                self.saved_pairs.add(int(m.group(1)))

    def _discover_axis_aip_dirs(self: ManualAlignWidget, pkg_root: Path) -> None:
        """Discover aips_xz/ and aips_yz/ directories in the package root."""
        for name, attr, paths_attr, pair_attr in [
            ("aips_xz", "aips_xz_dir", "slice_paths_xz", "pair_paths_xz"),
            ("aips_yz", "aips_yz_dir", "slice_paths_yz", "pair_paths_yz"),
        ]:
            candidate = pkg_root / name
            if candidate.exists():
                setattr(self, attr, candidate)
                setattr(self, paths_attr, discover_aips(candidate))
                setattr(self, pair_attr, discover_pair_aips(candidate))

        has_axis_aips = bool(self.slice_paths_xz or self.slice_paths_yz or self.pair_paths_xz or self.pair_paths_yz)
        self._btn_mode_z.setEnabled(has_axis_aips)
        if has_axis_aips:
            self._btn_mode_z.setToolTip("Depth alignment: adjust Z-overlap offsets, view XZ/YZ cross-sections")

    def _get_automated_state(self: ManualAlignWidget, mid: int) -> AlignmentState:
        """Return the best available initial AlignmentState for *mid*.

        Priority:
        1. A previously saved manual transform in ``output_dir/slice_z{mid:02d}/transform.tfm``.
        2. The automated pipeline transform from ``existing_transforms``.
        3. Zero state (no transform available).

        Side-effect: marks *mid* as saved in ``saved_pairs`` when a manual
        transform is found so the UI shows the correct "✓ SAVED" indicator.
        """
        scale = 2**self.level
        img_center = self.pair_centers.get(mid)

        # 1. Previously saved manual transform takes priority
        manual_tfm = self.output_dir / f"slice_z{mid:02d}" / "transform.tfm"
        if manual_tfm.exists():
            tx, ty, rot, tfm_center = load_transform(manual_tfm)
            if img_center is not None:
                tx, ty = adjust_for_rotation_center(tx, ty, rot, tfm_center, (img_center[0] * scale, img_center[1] * scale))
            self.saved_pairs.add(mid)
            return AlignmentState(tx=tx / scale, ty=ty / scale, rotation=rot)

        # 2. Automated transform
        if mid not in self.existing_transforms:
            return AlignmentState()
        tfm_dir = self.existing_transforms[mid]
        tfm_files = list(tfm_dir.glob("*.tfm"))
        if not tfm_files:
            return AlignmentState()
        tx, ty, rot, tfm_center = load_transform(tfm_files[0])
        if img_center is not None:
            tx, ty = adjust_for_rotation_center(tx, ty, rot, tfm_center, (img_center[0] * scale, img_center[1] * scale))
        return AlignmentState(tx=tx / scale, ty=ty / scale, rotation=rot)

    def _load_pair(self: ManualAlignWidget, idx: int, preserve_camera: bool = False) -> None:
        """Load a slice pair and display as red/green AIP overlay."""
        self.current_pair_idx = idx
        fid, mid = self.pairs[idx]

        self.viewer.status = f"Loading z{fid:02d} / z{mid:02d}..."

        # Select AIP paths based on projection mode.
        # For XZ/YZ: prefer paired files (both slices at the same column),
        # fall back to per-slice files, then fall back to XY.
        _pair_key = (fid, mid)
        if self._projection_mode in ("xz", "yz"):
            pair_store = self.pair_paths_xz if self._projection_mode == "xz" else self.pair_paths_yz
            per_slice_store = self.slice_paths_xz if self._projection_mode == "xz" else self.slice_paths_yz
            if _pair_key in pair_store and "fixed" in pair_store[_pair_key] and "moving" in pair_store[_pair_key]:
                # Paired files: use explicit fixed/moving paths
                _fixed_npz = pair_store[_pair_key]["fixed"]
                _moving_npz = pair_store[_pair_key]["moving"]
                _use_paired = True
            elif fid in per_slice_store and mid in per_slice_store:
                _fixed_npz = per_slice_store[fid]
                _moving_npz = per_slice_store[mid]
                _use_paired = False
            else:
                logger.warning(f"No {self._projection_mode.upper()} AIPs for pair z{fid:02d}→z{mid:02d}")
                _fixed_npz = _moving_npz = None
                _use_paired = False
        else:
            _fixed_npz = _moving_npz = None
            _use_paired = False

        if self._projection_mode in ("xz", "yz") and _fixed_npz is not None and _moving_npz is not None:
            fixed_aip, fixed_scale_yx = load_aip_from_npz(_fixed_npz)
            moving_aip, moving_scale_yx = load_aip_from_npz(_moving_npz)
        elif self._use_precomputed_aips:
            # Prefer paired XY AIPs (edge-limited depth slabs) when available.
            # Fall back to per-slice full-depth projections for backward compatibility.
            _xy_pair = self.pair_paths_xy.get(_pair_key, {})
            if "fixed" in _xy_pair and "moving" in _xy_pair:
                fixed_aip, fixed_scale_yx = load_aip_from_npz(_xy_pair["fixed"])
                moving_aip, moving_scale_yx = load_aip_from_npz(_xy_pair["moving"])
            else:
                fixed_aip, fixed_scale_yx = load_aip_from_npz(self.slice_paths[fid])
                moving_aip, moving_scale_yx = load_aip_from_npz(self.slice_paths[mid])
        else:
            fixed_aip, fixed_scale_yx = load_aip_from_ome_zarr(self.slice_paths[fid], level=self.level)
            moving_aip, moving_scale_yx = load_aip_from_ome_zarr(self.slice_paths[mid], level=self.level)

        # Normalize to [0, 1] for display
        fixed_aip = normalize_aip(fixed_aip)
        moving_aip = normalize_aip(moving_aip)

        # Crop both images to the brain content region of the fixed AIP so that
        # the napari canvas tightly fits the brain and does not include the large
        # dark tiled-mosaic border.  The same (row, col) box is applied to the
        # moving AIP so that relative tx/ty shifts are unchanged.
        #
        # Only crop in XY mode.  For XZ/YZ the row axis is the Z-depth of the
        # volume, and cropping it removes the empty Z-space that makes the
        # separation between the two slices visible.
        if self._projection_mode == "xy":
            r1, c1, r2, c2 = content_bbox(fixed_aip)
            self._content_bbox_wl = (r1, c1, r2, c2)
            fixed_aip = fixed_aip[r1:r2, c1:c2].copy()
            moving_aip = moving_aip[r1:r2, c1:c2].copy()
        else:
            r1, c1 = 0, 0
            # Load the full-depth XY AIP for the fixed slice to determine the brain's
            # Y and X extent, used to restrict the cross-section sliders to tissue only.
            _xy_fixed: np.ndarray | None = None
            if self._use_precomputed_aips and fid in self.slice_paths:
                with contextlib.suppress(Exception):
                    _xy_fixed, _ = load_aip_from_npz(self.slice_paths[fid])
            elif not self._use_precomputed_aips and fid in self.slice_paths:
                with contextlib.suppress(Exception):
                    _xy_fixed, _ = load_aip_from_ome_zarr(self.slice_paths[fid], level=self.level)
            if _xy_fixed is not None:
                _xy_fixed = normalize_aip(_xy_fixed)
                self._content_bbox_wl = content_bbox(_xy_fixed)
            else:
                self._content_bbox_wl = None
        # Preserve crop origin (working-level pixels) for transform-centre accounting
        self._crop_rc: tuple[int, int] = (r1, c1)

        # Store raw (normalized, cropped) AIPs so enhancement can be changed without reloading from disk
        self._raw_fixed_aip = fixed_aip.copy()
        self._raw_moving_aip = moving_aip.copy()

        # Apply current enhancement to derive the display AIPs
        fixed_aip = enhance_aip(fixed_aip, self._enhance_mode)
        moving_aip = enhance_aip(moving_aip, self._enhance_mode)

        self._original_fixed_aip = fixed_aip.copy()
        self._original_moving_aip = moving_aip.copy()

        # Store image centre for this pair (only XY view, needed for rotation).
        # Include the crop offset so the centre is expressed in the original
        # (full-working-level) coordinate system — required for save_transform
        # and adjust_for_rotation_center to produce correct full-resolution values.
        if self._projection_mode == "xy":
            self.pair_centers[mid] = (
                c1 + moving_aip.shape[1] / 2.0,
                r1 + moving_aip.shape[0] / 2.0,
            )  # (cx, cy) in original working-level pixel coords

        # Snapshot layer visual settings so they survive the pair switch.
        # On first load the layers don't exist yet, so fall back to defaults.
        fixed_gamma = self.fixed_layer.gamma if self.fixed_layer is not None else 0.6
        fixed_opacity = self.fixed_layer.opacity if self.fixed_layer is not None else 1.0
        fixed_clim = tuple(self.fixed_layer.contrast_limits) if self.fixed_layer is not None else (0.0, 1.0)
        moving_gamma = self.moving_layer.gamma if self.moving_layer is not None else 0.6
        moving_opacity = self.moving_layer.opacity if self.moving_layer is not None else 1.0
        moving_clim = tuple(self.moving_layer.contrast_limits) if self.moving_layer is not None else (0.0, 1.0)

        # Remove existing layers (including stale composite)
        self._composite_layer = None
        while len(self.viewer.layers) > 0:
            self.viewer.layers.pop(0)

        view_suffix = {"xy": "", "xz": " (XZ)", "yz": " (YZ)"}.get(self._projection_mode, "")

        self._moving_scale_yx = list(moving_scale_yx)

        # Recreate layers, restoring any settings the user adjusted in napari's layer controls.
        self.fixed_layer = self.viewer.add_image(
            fixed_aip,
            name=f"Fixed z{fid:02d}{view_suffix}",
            colormap="green",
            blending="additive",
            contrast_limits=fixed_clim,
            gamma=fixed_gamma,
            opacity=fixed_opacity,
            scale=fixed_scale_yx,
        )
        self.moving_layer = self.viewer.add_image(
            moving_aip,
            name=f"Moving z{mid:02d}{view_suffix}",
            colormap="red",
            blending="additive",
            contrast_limits=moving_clim,
            gamma=moving_gamma,
            opacity=moving_opacity,
            scale=list(moving_scale_yx),
        )

        # Adjust visibility / add composite layer based on current overlay mode
        self._rebuild_layer_visibility()

        # Initialize or restore undo stack
        if mid not in self.undo_stacks:
            initial = self._get_automated_state(mid)
            self.undo_stacks[mid] = UndoStack(initial)

        # Load Z-offsets for this pair — prefer a manually saved offsets.txt
        if mid not in self._current_offsets:
            manual_offsets_path = self.output_dir / f"slice_z{mid:02d}" / "offsets.txt"
            if manual_offsets_path.exists():
                offsets = load_offsets(manual_offsets_path)
            elif mid in self.existing_transforms:
                offsets = load_offsets(self.existing_transforms[mid] / "offsets.txt")
            else:
                offsets = (0, 0)
            self._current_offsets[mid] = offsets

        # Update Z-offset spinboxes
        with self._suppress_z_events():
            self.spin_fixed_z.setValue(self._current_offsets[mid][0])
            self.spin_moving_z.setValue(self._current_offsets[mid][1])
        self._update_z_relative_label()

        state = self.undo_stacks[mid].current
        self._apply_state(state, push=False)

        # Update UI
        with self._suppress_events():
            self.pair_combo.setCurrentIndex(idx)
        self._update_status()

        # Interactive cross-section sliders
        if self._projection_mode in ("xz", "yz"):
            if self._cs_mgr.slices_remote_dir is not None:
                # Resolve the cross-section position for this pair.  The priority
                # chain inside _update_initial_cs_position is:
                #   1. _cs_positions[mid] (in-memory — set whenever the user moves
                #      the slider, so an XY↔Z round-trip restores automatically)
                #   2. cs_position.txt on disk (persists across reloads)
                #   3. NPZ tissue centroid (first-time default)
                #   4. 0
                # This means we can call it unconditionally: when the user already
                # scrolled, priority 1 returns their saved position immediately.
                self._update_initial_cs_position(fid, mid)
                self._update_cs_slider_visibility()
                self._ensure_readers_for_pair()
                # Reader already open (e.g. view-mode switch XY↔Z or XZ↔YZ): init slider now
                if self._cs_mgr.has_reader(mid):
                    self._init_cs_slider_from_reader(mid)
            else:
                self._set_cs_sliders_visible(False)
        else:
            self._set_cs_sliders_visible(False)

        if not preserve_camera:
            self.viewer.reset_view()
        self.viewer.status = f"Pair z{fid:02d} → z{mid:02d} loaded ({self._projection_mode.upper()} view)"
