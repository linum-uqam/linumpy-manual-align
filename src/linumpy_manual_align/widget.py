"""Napari dock widget for interactive manual slice alignment.

Provides two complementary alignment modes:

XY Alignment
    Displays consecutive slice AIP projections as a red/green overlay.
    The user adjusts TX, TY, and rotation of the moving slice.
    Three overlay modes (Color, Difference, Checkerboard) and four
    enhancement modes (None, Edges, CLAHE, Sharpen) aid visibility.

Z Alignment
    Displays XZ and YZ center cross-sections of each slice to verify
    the depth-overlap (Z-offset) between consecutive slices.
    Fixed-Z and Moving-Z spinboxes control the overlap voxel indices.

Saves corrected transforms as SimpleITK .tfm files compatible with
the linumpy stacking pipeline.
"""

from __future__ import annotations

import contextlib
import logging
import re
from pathlib import Path

import napari
import numpy as np
from qtpy.QtCore import QTimer
from qtpy.QtWidgets import (
    QFileDialog,
    QLabel,
    QMessageBox,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from linumpy_manual_align.cross_section import CrossSectionManager
from linumpy_manual_align.image_utils import (
    ENHANCE_CLAHE,
    ENHANCE_EDGES,
    ENHANCE_NONE,
    ENHANCE_SHARPEN,
    OVERLAY_CHECKER,
    OVERLAY_COLOR,
    OVERLAY_DIFF,
    apply_transform,
    build_overlay,
    content_bbox,
    enhance_aip,
    normalize_aip,
)
from linumpy_manual_align.omezarr_io import load_aip_from_ome_zarr
from linumpy_manual_align.server_transfer import ScpWorker
from linumpy_manual_align.state import AlignmentState, UndoStack
from linumpy_manual_align.transform_io import (
    adjust_for_rotation_center,
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
from linumpy_manual_align.ui_builder import (
    build_display_group,
    build_mode_row,
    build_navigation_row,
    build_save_row,
    build_scroll_content,
    build_server_group,
    build_xy_page,
    build_z_page,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Widget
# ---------------------------------------------------------------------------


# (key, method_name, args) — used by _install_keybindings
_KEYBINDINGS: list[tuple[str, str, tuple]] = [
    # Navigation
    ("n", "_next_pair", ()),
    ("p", "_prev_pair", ()),
    ("s", "_save_current", ()),
    # Fine translation (1 px) / Z fine nudge (1 voxel) — mode-aware
    ("Right", "_nudge_translate", (1, 0)),
    ("Left", "_nudge_translate", (-1, 0)),
    ("Up", "_nudge_translate", (0, -1)),
    ("Down", "_nudge_translate", (0, 1)),
    # Coarse translation (10 px) / Z coarse nudge (10 voxels) — mode-aware
    ("Alt-Right", "_nudge_translate", (10, 0)),
    ("Alt-Left", "_nudge_translate", (-10, 0)),
    ("Alt-Up", "_nudge_translate", (0, -10)),
    ("Alt-Down", "_nudge_translate", (0, 10)),
    # Large translation (50 px) — XY only
    ("Control-Right", "_nudge_translate", (50, 0)),
    ("Control-Left", "_nudge_translate", (-50, 0)),
    ("Control-Up", "_nudge_translate", (0, -50)),
    ("Control-Down", "_nudge_translate", (0, 50)),
    # Rotation  (] = CW, [ = CCW; positive rotation = CW in apply_transform)
    ("]", "_nudge_rotate", (0.1,)),
    ("[", "_nudge_rotate", (-0.1,)),
    ("Alt-]", "_nudge_rotate", (1.0,)),
    ("Alt-[", "_nudge_rotate", (-1.0,)),
    ("Control-]", "_nudge_rotate", (5.0,)),
    ("Control-[", "_nudge_rotate", (-5.0,)),
    # Z alignment: switch XZ ↔ YZ view
    ("v", "_toggle_z_proj", ()),
    # Toggle between XY and Z alignment modes
    ("m", "_toggle_alignment_mode", ()),
    # Interactive cross-section slider nudge (step 10 px) — Z alignment only
    ("Alt-,", "_nudge_cs_position", (-10,)),
    ("Alt-.", "_nudge_cs_position", (10,)),
]


class ManualAlignWidget(QWidget):
    """Napari dock widget for manual slice alignment."""

    def __init__(
        self,
        viewer: napari.Viewer,
        input_dir: Path | None,
        transforms_dir: Path | None,
        output_dir: Path,
        level: int = 1,
        filter_slices: list[int] | None = None,
        aips_dir: Path | None = None,
        aips_xz_dir: Path | None = None,
        aips_yz_dir: Path | None = None,
        server_config: object = None,
    ):
        super().__init__()
        self.viewer = viewer
        self.input_dir = Path(input_dir) if input_dir else None
        self.transforms_dir = Path(transforms_dir) if transforms_dir else None
        self.output_dir = Path(output_dir)
        self.level = level
        self.aips_dir = Path(aips_dir) if aips_dir else None
        self.aips_xz_dir = Path(aips_xz_dir) if aips_xz_dir else None
        self.aips_yz_dir = Path(aips_yz_dir) if aips_yz_dir else None
        self.server_config = server_config

        # Discover slices — from AIP package or OME-Zarr directory
        if self.aips_dir is not None:
            self.slice_paths = discover_aips(self.aips_dir)
        elif self.input_dir is not None:
            self.slice_paths = discover_slices(self.input_dir)
        else:
            # Empty startup — will be populated after server download
            self.slice_paths = {}

        # Discover axis-specific AIPs (XZ, YZ projections)
        self.slice_paths_xz = discover_aips(self.aips_xz_dir) if self.aips_xz_dir else {}
        self.slice_paths_yz = discover_aips(self.aips_yz_dir) if self.aips_yz_dir else {}
        # Paired XZ/YZ files — share a common column per pair for correct visual alignment
        self.pair_paths_xz: dict[tuple[int, int], dict[str, Path]] = (
            discover_pair_aips(self.aips_xz_dir) if self.aips_xz_dir else {}
        )
        self.pair_paths_yz: dict[tuple[int, int], dict[str, Path]] = (
            discover_pair_aips(self.aips_yz_dir) if self.aips_yz_dir else {}
        )

        self.slice_ids = list(self.slice_paths.keys())
        self.existing_transforms = discover_transforms(self.transforms_dir) if self.transforms_dir else {}
        self._filter_slices = filter_slices

        self.pairs: list[tuple[int, int]] = []
        self._build_pairs()

        if not self.pairs:
            logger.info("Starting in empty state — no slice pairs found. Download data from server.")

        # Per-pair state
        self.undo_stacks: dict[int, UndoStack] = {}  # keyed by moving_id
        self.saved_pairs: set[int] = set()  # moving_ids that have been saved
        self.unsaved_changes: set[int] = set()  # moving_ids with unsaved modifications
        self.pair_centers: dict[int, tuple[float, float]] = {}  # (cx, cy) per moving_id

        self._refresh_saved_pairs()

        # Layers (set during pair loading)
        self.fixed_layer: napari.layers.Image | None = None
        self.moving_layer: napari.layers.Image | None = None
        self._composite_layer: napari.layers.Image | None = None
        # Raw (normalized, pre-enhancement) AIPs kept so we can re-enhance without disk I/O
        self._raw_fixed_aip: np.ndarray | None = None
        self._raw_moving_aip: np.ndarray | None = None
        # Enhanced AIPs — what is actually displayed and used for compositing
        self._moving_scale_yx: list[float] = [1.0, 1.0]
        self._original_moving_aip: np.ndarray | None = None  # before rotation
        self._suppress_spinbox_event = False
        self._suppress_z_offset_event = False
        self._worker: ScpWorker | None = None  # prevent GC of background thread
        self._close_confirmed = False

        # ── Interactive XZ/YZ via remote OME-Zarr ──────────────────────────────
        # CrossSectionManager owns readers, cache, prefetch, and metadata.
        self._cs_mgr = CrossSectionManager(parent=self)
        self._cs_mgr.reader_ready.connect(self._on_reader_ready)
        self._cs_mgr.reader_failed.connect(self._on_reader_failed)
        self._cs_mgr.cross_section_ready.connect(self._on_cross_section_ready)
        self._cs_mgr.cross_section_failed.connect(self._on_cross_section_failed)
        # Moving slice cross-section position (slider-controlled; fixed slice uses static NPZ).
        self._cross_section_y: int = 0
        self._cross_section_x: int = 0
        # Debounce timer for slider-driven requests.
        self._cs_debounce_timer: QTimer | None = None

        # Projection mode and Z-overlap state
        self._projection_mode = "xy"  # "xy", "xz", "yz"
        self._current_offsets: dict[int, tuple[int, int]] = {}  # mid -> (fixed_z, moving_z)

        # Overlay display mode and AIP enhancement mode
        self._overlay_mode = OVERLAY_COLOR
        self._enhance_mode = ENHANCE_NONE

        # Current pair index
        self.current_pair_idx = 0

        self._build_ui()
        self._install_keybindings()
        self._install_close_guard()
        if self.pairs:
            self._load_pair(0)
        else:
            # No data yet — check if an existing package is already on disk
            if self.server_config is not None and not self.pairs:
                existing = self._find_existing_package()
                if existing is not None:
                    self._load_existing_package(existing)
                    self.server_status_label.setText(f"Existing package loaded from {existing.parent}")
            self._update_status()

    # ----- UI construction -----

    @property
    def _use_precomputed_aips(self) -> bool:
        """True when pre-computed NPZ AIPs are available (``aips_dir`` is set)."""
        return self.aips_dir is not None

    @contextlib.contextmanager
    def _suppress_events(self):  # type: ignore[return]
        """Context manager that suppresses spinbox-changed signals.

        Guarantees the flag is restored even if an exception is raised inside
        the block, eliminating the fragile set/finally pattern.
        """
        self._suppress_spinbox_event = True
        try:
            yield
        finally:
            self._suppress_spinbox_event = False

    @contextlib.contextmanager
    def _suppress_z_events(self):  # type: ignore[return]
        """Context manager that suppresses Z-offset spinbox-changed signals."""
        self._suppress_z_offset_event = True
        try:
            yield
        finally:
            self._suppress_z_offset_event = False

    def _set_cs_sliders_visible(self, visible: bool) -> None:
        """Show or hide the interactive cross-section slider rows."""
        for widget in (
            self.slider_cs_y,
            self._lbl_cs_y,
            self._cs_y_form_row_label,
            self.slider_cs_x,
            self._lbl_cs_x,
            self._cs_x_form_row_label,
            self._cs_loading_label,
        ):
            widget.setVisible(visible)

    def _update_cs_slider_visibility(self) -> None:
        """Show the correct cross-section slider for the current projection mode."""
        if self._cs_mgr.slices_remote_dir is None:
            return
        xz = self._projection_mode == "xz"
        yz = self._projection_mode == "yz"
        self.slider_cs_y.setVisible(xz)
        self._lbl_cs_y.setVisible(xz)
        self._cs_y_form_row_label.setVisible(xz)
        self.slider_cs_x.setVisible(yz)
        self._lbl_cs_x.setVisible(yz)
        self._cs_x_form_row_label.setVisible(yz)
        self._cs_loading_label.setVisible(xz or yz)

    def _refresh_saved_pairs(self) -> None:
        """Scan output_dir and pre-populate saved_pairs with already-written transforms."""
        if not self.output_dir.exists():
            return
        pattern = re.compile(r"slice_z(\d+)$")
        for p in self.output_dir.iterdir():
            m = pattern.match(p.name)
            if m and (p / "transform.tfm").exists():
                self.saved_pairs.add(int(m.group(1)))

    def _build_pairs(self) -> None:
        """Populate ``self.pairs`` from ``self.slice_ids``, respecting any active slice filter."""
        self.pairs = [
            (self.slice_ids[i], self.slice_ids[i + 1])
            for i in range(len(self.slice_ids) - 1)
            if self._filter_slices is None or self.slice_ids[i + 1] in self._filter_slices
        ]

    def _pair_label(self, fid: int, mid: int) -> str:
        """Return the display label for a pair (fid → mid), including metrics if available."""
        label = f"z{fid:02d} → z{mid:02d}"
        if mid in self.existing_transforms:
            metrics_path = self.existing_transforms[mid] / "pairwise_registration_metrics.json"
            metrics = load_pairwise_metrics(metrics_path)
            mag = get_metric(metrics, "translation_magnitude")
            if mag is not None:
                label += f"  ({mag:.0f}px)"
        return label

    def _build_ui(self) -> None:
        # ── Outer scroll area ────────────────────────────────────────────────
        outer = QVBoxLayout()
        outer.setContentsMargins(0, 0, 0, 0)
        self.setLayout(outer)
        scroll, layout = build_scroll_content()
        outer.addWidget(scroll)

        # ── Pair navigation ───────────────────────────────────────────────────
        nav_layout, nav = build_navigation_row(
            pairs=self.pairs,
            pair_labels=[self._pair_label(fid, mid) for fid, mid in self.pairs],
            on_prev=self._prev_pair,
            on_next=self._next_pair,
            on_combo_changed=self._on_pair_changed,
        )
        self.btn_prev = nav.btn_prev
        self.btn_next = nav.btn_next
        self.pair_combo = nav.pair_combo
        layout.addLayout(nav_layout)

        # ── Mode toggle buttons ───────────────────────────────────────────────
        has_axis_aips = bool(self.slice_paths_xz or self.slice_paths_yz or self.pair_paths_xz or self.pair_paths_yz)
        mode_layout, mode = build_mode_row(
            has_axis_aips=has_axis_aips,
            on_xy_toggled=lambda checked: self._on_mode_btn_toggled("xy", checked),
            on_z_toggled=lambda checked: self._on_mode_btn_toggled("z", checked),
        )
        self._btn_mode_xy = mode.btn_mode_xy
        self._btn_mode_z = mode.btn_mode_z
        layout.addLayout(mode_layout)

        # ── Stacked content (sizes to active page only) ───────────────────────
        self._mode_stack = QStackedWidget()
        layout.addWidget(self._mode_stack)

        # Page 0 - XY Alignment
        xy_page, xy = build_xy_page(
            on_spinbox_changed=self._on_spinbox_changed,
            on_rotation_changed=self._on_rotation_changed,
            on_rotation_slider_changed=self._on_rotation_slider_changed,
            on_load_auto=self._load_automated_transform,
            on_reset=self._reset_transform,
            on_undo=self._undo,
            on_redo=self._redo,
        )
        self.spin_tx = xy.spin_tx
        self.spin_ty = xy.spin_ty
        self.spin_rot = xy.spin_rot
        self.rot_slider = xy.rot_slider
        self.btn_load_auto = xy.btn_load_auto
        self.btn_reset = xy.btn_reset
        self.btn_undo = xy.btn_undo
        self.btn_redo = xy.btn_redo
        self._mode_stack.addWidget(xy_page)

        # Page 1 - Z Alignment
        z_page, z = build_z_page(
            parent_widget=self,
            on_proj_changed=self._on_z_proj_changed,
            on_fixed_z_changed=self._on_z_offset_changed,
            on_moving_z_changed=self._on_z_offset_changed,
        )
        self._btn_proj_xz = z.btn_proj_xz
        self._btn_proj_yz = z.btn_proj_yz
        self._proj_btn_group = z.proj_btn_group
        self.spin_fixed_z = z.spin_fixed_z
        self.spin_moving_z = z.spin_moving_z
        self.z_relative_label = z.z_relative_label
        self.slider_cs_y = z.slider_cs_y
        self._lbl_cs_y = z.lbl_cs_y
        self._cs_y_form_row_label = z.cs_y_form_row_label
        self.slider_cs_x = z.slider_cs_x
        self._lbl_cs_x = z.lbl_cs_x
        self._cs_x_form_row_label = z.cs_x_form_row_label
        self._cs_loading_label = z.cs_loading_label
        self._set_cs_sliders_visible(False)
        self._mode_stack.addWidget(z_page)

        # ── Display ───────────────────────────────────────────────────────────
        disp_group, disp = build_display_group(
            on_overlay_changed=self._on_overlay_mode_changed,
            on_enhance_changed=self._on_enhance_changed,
            on_tile_size_changed=self._on_tile_size_changed,
        )
        self.combo_overlay = disp.combo_overlay
        self.combo_enhance = disp.combo_enhance
        self.spin_tile = disp.spin_tile
        self._tile_row_label = disp.tile_row_label
        layout.addWidget(disp_group)

        # ── Save ──────────────────────────────────────────────────────────────
        save_layout, save = build_save_row(
            on_save=self._save_current,
            on_save_all=self._save_all_and_exit,
        )
        self.btn_save = save.btn_save
        self.btn_save_all = save.btn_save_all
        layout.addLayout(save_layout)

        # ── Server ────────────────────────────────────────────────────────────
        server_group, srv = build_server_group(
            server_config=self.server_config,
            on_browse=self._browse_server_config,
            on_host_changed=self._on_host_changed,
            on_download=self._download_from_server,
            on_upload=self._upload_to_server,
        )
        self.config_path_edit = srv.config_path_edit
        self.btn_browse_config = srv.btn_browse_config
        self.host_edit = srv.host_edit
        self.btn_download = srv.btn_download
        self.btn_upload = srv.btn_upload
        self.server_progress = srv.server_progress
        self.server_status_label = srv.server_status_label
        layout.addWidget(server_group)

        # ── Status ────────────────────────────────────────────────────────────
        self.status_label = QLabel("")
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)

        # Wire cross-section slider signals (debounced via QTimer)
        self._cs_debounce_timer = QTimer(self)
        self._cs_debounce_timer.setSingleShot(True)
        self._cs_debounce_timer.timeout.connect(self._on_cs_slider_settled)

        def _on_y_changed(v: int) -> None:
            self._lbl_cs_y.setText(str(v))
            self._cross_section_y = v
            self._cs_debounce_timer.start(150)

        def _on_x_changed(v: int) -> None:
            self._lbl_cs_x.setText(str(v))
            self._cross_section_x = v
            self._cs_debounce_timer.start(150)

        self.slider_cs_y.valueChanged.connect(_on_y_changed)
        self.slider_cs_x.valueChanged.connect(_on_x_changed)

    # ----- AIP discovery -----

    def _discover_axis_aip_dirs(self, pkg_root: Path) -> None:
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

        # Enable the Z Alignment button if axis AIPs are now available
        has_axis_aips = bool(self.slice_paths_xz or self.slice_paths_yz or self.pair_paths_xz or self.pair_paths_yz)
        self._btn_mode_z.setEnabled(has_axis_aips)
        if has_axis_aips:
            self._btn_mode_z.setToolTip("Depth alignment: adjust Z-overlap offsets, view XZ/YZ cross-sections")

    # ----- Pair loading -----

    def _get_automated_state(self, mid: int) -> AlignmentState:
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

    def _load_pair(self, idx: int, preserve_camera: bool = False) -> None:
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
            fixed_aip = fixed_aip[r1:r2, c1:c2].copy()
            moving_aip = moving_aip[r1:r2, c1:c2].copy()
        else:
            r1, c1 = 0, 0
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
                # Estimate tissue centroid from the static NPZ so the initial
                # remote fetch lands on tissue rather than the volume edge.
                self._update_initial_cs_position(fid, mid)
                self._update_cs_slider_visibility()
                self._ensure_readers_for_pair()
                # Reader already open (e.g. view-mode switch XZ↔YZ): init slider now
                if self._cs_mgr.has_reader(mid):
                    self._init_cs_slider_from_reader(mid)
            else:
                self._set_cs_sliders_visible(False)
        else:
            self._set_cs_sliders_visible(False)

        if not preserve_camera:
            self.viewer.reset_view()
        self.viewer.status = f"Pair z{fid:02d} → z{mid:02d} loaded ({self._projection_mode.upper()} view)"

    # ----- Overlay helpers -----

    def _make_shifted_moving(self, state: AlignmentState) -> np.ndarray:
        """Return the moving AIP with rotation + pixel-level shift baked in (for composite modes)."""
        moving = self._original_moving_aip
        if moving is None:
            return np.zeros((1, 1), dtype=np.float32)
        return apply_transform(moving, rotation=state.rotation, tx=state.tx, ty=state.ty)

    def _rebuild_layer_visibility(self) -> None:
        is_color = self._overlay_mode == OVERLAY_COLOR
        colormap = "inferno" if self._overlay_mode == OVERLAY_DIFF else "gray"

        if self.fixed_layer is not None:
            self.fixed_layer.visible = is_color
        if self.moving_layer is not None:
            self.moving_layer.visible = is_color

        if is_color:
            # Switching to color mode — remove composite if present.
            if self._composite_layer is not None:
                with contextlib.suppress(ValueError):
                    self.viewer.layers.remove(self._composite_layer)
                self._composite_layer = None
        elif self._composite_layer is not None:
            # Already have a composite layer; just update the colormap in-place
            # (avoids tearing down and recreating the layer when switching Diff ↔ Checker).
            self._composite_layer.colormap = colormap
        elif self._original_fixed_aip is not None and self.fixed_layer is not None:
            comp = np.zeros_like(self._original_fixed_aip)
            self._composite_layer = self.viewer.add_image(
                comp,
                name="Composite",
                colormap=colormap,
                blending="translucent",
                contrast_limits=(0.0, 1.0),
                scale=list(self.fixed_layer.scale),
            )

    def _refresh_composite(self, state: AlignmentState) -> None:
        """Recompute and push composite image data for non-color overlay modes."""
        if self._overlay_mode == OVERLAY_COLOR or self._composite_layer is None:
            return
        if self._original_fixed_aip is None or self._original_moving_aip is None:
            return
        shifted = self._make_shifted_moving(state)
        self._composite_layer.data = build_overlay(
            self._original_fixed_aip, shifted, mode=self._overlay_mode, tile_size=self.spin_tile.value()
        )

    # ----- State application -----

    def _apply_state(self, state: AlignmentState, push: bool = True) -> None:
        """Apply an alignment state to the moving layer."""
        if self.moving_layer is None or self._original_moving_aip is None or not self.pairs:
            return

        mid = self.pairs[self.current_pair_idx][1]

        if push:
            self.undo_stacks[mid].push(state)
            self.unsaved_changes.add(mid)

        sy, sx = self._moving_scale_yx[0], self._moving_scale_yx[1]

        if self._projection_mode == "xy":
            # Bake ONLY rotation into pixel data (about the image centre, tx=ty=0).
            # Translation is applied via layer.translate so napari can size the canvas
            # correctly — baking a large shift would push content into one corner and
            # leave most of the frame as empty black padding.
            # Pure-translation layer.translate is reliable (no QR decomposition issue).
            baked = apply_transform(
                self._original_moving_aip,
                rotation=state.rotation,
                tx=0,
                ty=0,
            )
            self.moving_layer.data = baked
            self.moving_layer.rotate = 0.0
            self.moving_layer.translate = [state.ty * sy, state.tx * sx]
        else:
            # XZ/YZ mode: pure translation only (no rotation).
            # The export script flips Z ([::-1]) so row 0 = deepest voxel.
            # A higher Z-index therefore maps to a LOWER row index, so the
            # correct shift to bring the overlap regions into alignment is
            # (moving_z - fixed_z) / scale, not (fixed_z - moving_z).
            offsets = self._current_offsets.get(mid, (0, 0))
            dz_display = (offsets[1] - offsets[0]) / 2**self.level
            horiz = state.tx if self._projection_mode == "xz" else state.ty
            self.moving_layer.rotate = 0.0
            self.moving_layer.translate = [dz_display * sy, horiz * sx]

        # Update composite overlay if active
        if self._overlay_mode != OVERLAY_COLOR and self._projection_mode == "xy":
            self._refresh_composite(state)

        # Sync spinboxes
        with self._suppress_events():
            self.spin_tx.setValue(state.tx)
            self.spin_ty.setValue(state.ty)
            self.spin_rot.setValue(state.rotation)
            self.rot_slider.setValue(int(state.rotation * 10))

    def _current_state(self) -> AlignmentState:
        return AlignmentState(
            tx=self.spin_tx.value(),
            ty=self.spin_ty.value(),
            rotation=self.spin_rot.value(),
        )

    # ----- Event handlers -----

    def _on_pair_changed(self, idx: int) -> None:
        if idx >= 0 and idx != self.current_pair_idx:
            self._load_pair_preserve_camera(idx)

    def _on_spinbox_changed(self) -> None:
        if self._suppress_spinbox_event:
            return
        state = self._current_state()
        self._apply_state(state, push=True)
        self._update_status()

    def _on_rotation_changed(self) -> None:
        if self._suppress_spinbox_event:
            return
        state = self._current_state()
        with self._suppress_events():
            self.rot_slider.setValue(int(state.rotation * 10))
        self._apply_state(state, push=True)
        self._update_status()

    def _on_rotation_slider_changed(self, value: int) -> None:
        if self._suppress_spinbox_event:
            return
        rot = value / 10.0
        with self._suppress_events():
            self.spin_rot.setValue(rot)
        state = AlignmentState(tx=self.spin_tx.value(), ty=self.spin_ty.value(), rotation=rot)
        self._apply_state(state, push=True)
        self._update_status()

    # ----- Projection and Z-offset handlers -----

    def _on_mode_btn_toggled(self, mode: str, checked: bool) -> None:
        """Handle XY / Z mode toggle buttons (mutually exclusive)."""
        if not checked:
            return
        if mode == "xy":
            self._btn_mode_z.setChecked(False)
            self._projection_mode = "xy"
            self._mode_stack.setCurrentIndex(0)
        else:
            self._btn_mode_xy.setChecked(False)
            self._projection_mode = "xz" if self._btn_proj_xz.isChecked() else "yz"
            self._mode_stack.setCurrentIndex(1)
        if self.pairs:
            self._load_pair(self.current_pair_idx)

    def _on_z_proj_changed(self, btn_id: int) -> None:
        """Switch between XZ and YZ within the Z Alignment page."""
        if self._projection_mode == "xy":
            return  # ignore if XY mode is active
        self._projection_mode = "xz" if btn_id == 0 else "yz"
        if self.pairs:
            self._load_pair_preserve_camera(self.current_pair_idx)

    def _on_z_offset_changed(self) -> None:
        """Handle Z-offset spinbox changes."""
        if self._suppress_z_offset_event or not self.pairs:
            return
        mid = self.pairs[self.current_pair_idx][1]
        self._current_offsets[mid] = (self.spin_fixed_z.value(), self.spin_moving_z.value())
        self.unsaved_changes.add(mid)
        self._update_z_relative_label()

        # Re-apply current state to update display (Z-offset affects XZ/YZ views)
        state = self._current_state()
        self._apply_state(state, push=False)
        self._update_status()

    def _update_z_relative_label(self) -> None:
        """Update the relative Z-shift label."""
        fixed_z = self.spin_fixed_z.value()
        moving_z = self.spin_moving_z.value()
        diff = fixed_z - moving_z
        self.z_relative_label.setText(f"Relative shift: {diff:+d} voxels")

    # ----- Interactive XZ/YZ cross-section (remote OME-Zarr) -----

    def _update_initial_cs_position(self, fid: int, mid: int) -> None:
        """Load the tissue centroid column from the static NPZ for this pair.

        The export script stores the Y (XZ) or X (YZ) index at which the
        cross-section was taken as ``center_pos``.  We use that as the initial
        slider position so the first remote fetch lands on tissue.
        Falls back to 0 (letting _on_reader_ready use ny//2) when absent.
        """
        pair_key = (fid, mid)
        axis = self._projection_mode  # "xz" or "yz"
        pair_store = self.pair_paths_xz if axis == "xz" else self.pair_paths_yz
        per_slice_store = self.slice_paths_xz if axis == "xz" else self.slice_paths_yz

        npz_path: Path | None = None
        if pair_key in pair_store and "fixed" in pair_store[pair_key]:
            npz_path = pair_store[pair_key]["fixed"]
        elif fid in per_slice_store:
            npz_path = per_slice_store[fid]

        # Reset so _on_reader_ready falls back to ny//2 if center_pos is absent
        self._cross_section_y = 0
        self._cross_section_x = 0

        if npz_path is None:
            return

        try:
            data = np.load(str(npz_path))
            if "center_pos" not in data:
                return
            cp = int(data["center_pos"])
            if axis == "xz":
                self._cross_section_y = cp
            else:
                self._cross_section_x = cp
        except Exception:
            pass

    def _ensure_readers_for_pair(self) -> None:
        """Start a SliceReaderWorker for the moving slice of the current pair.

        The fixed slice is always shown from the static downloaded NPZ — no reader needed.
        """
        if not self.pairs or self.server_config is None:
            return
        _fid, mid = self.pairs[self.current_pair_idx]
        if self._cs_mgr.has_reader(mid):
            return
        self._cs_mgr.ensure_reader(self.server_config, mid)
        self._cs_loading_label.setText("Opening remote reader for moving slice…")

    def _init_cs_slider_from_reader(self, mid: int) -> None:
        """Initialise the active cross-section slider from the already-open reader for *mid*.

        Sets slider range, initial value (tissue centroid + XY-transform offset),
        and requests the first cross-section fetch.  Safe to call whenever the reader
        is already open but the slider has not yet been wired (e.g. after a view-mode switch).
        """
        shape = self._cs_mgr.reader_shape(mid)
        if shape is None:
            return
        self._cs_loading_label.setText("")
        _nz, ny, nx = shape
        axis = self._projection_mode  # "xz" or "yz"
        center = self._cross_section_y if axis == "xz" else self._cross_section_x
        init = center + self._cs_moving_offset(axis)

        if axis == "xz":
            init = max(0, min(ny - 1, init))
            self.slider_cs_y.blockSignals(True)
            self.slider_cs_y.setMaximum(ny - 1)
            self.slider_cs_y.setValue(init)
            self.slider_cs_y.blockSignals(False)
            self._lbl_cs_y.setText(str(init))
            self._cross_section_y = init
        else:
            init = max(0, min(nx - 1, init))
            self.slider_cs_x.blockSignals(True)
            self.slider_cs_x.setMaximum(nx - 1)
            self.slider_cs_x.setValue(init)
            self.slider_cs_x.blockSignals(False)
            self._lbl_cs_x.setText(str(init))
            self._cross_section_x = init

        self._cs_mgr.request(mid, axis, init)

    def _on_reader_ready(self, sid: int, reader: object) -> None:
        """Called when ``CrossSectionManager`` finishes opening a reader."""
        if not self.pairs:
            return
        _fid, mid = self.pairs[self.current_pair_idx]
        if sid != mid:
            return
        self._init_cs_slider_from_reader(mid)

    def _on_reader_failed(self, sid: int, msg: str) -> None:
        """Called when a remote reader fails to open."""
        self._cs_loading_label.setText(f"<span style='color:red;'>Reader failed for z{sid:02d}</span>")

    def _on_cross_section_ready(self, sid: int, axis: str, pos: int, img: object) -> None:
        """Refresh display when a cross-section image arrives in the manager cache."""
        if not self.pairs:
            return
        fid, mid = self.pairs[self.current_pair_idx]
        if sid not in (fid, mid):
            return
        if self._projection_mode != axis:
            return

        # Enable the appropriate slider now that we have real data
        if axis == "xz":
            self.slider_cs_y.setEnabled(True)
        else:
            self.slider_cs_x.setEnabled(True)

        self._refresh_cross_section()

    def _on_cross_section_failed(self, sid: int, axis: str, pos: int, msg: str) -> None:
        logger.warning(f"Cross-section fetch failed z{sid:02d} {axis}[{pos}]: {msg}")

    def _cs_moving_offset(self, axis: str) -> int:
        """Pixel offset to add to the fixed slider position to show matching tissue in the moving slice.

        The moving slice has been shifted by (tx, ty) relative to the fixed slice (at working
        resolution self.level).  To see the same anatomical tissue in both cross-sections we must
        fetch the moving slice at a different column:
          XZ view (Y slider):  moving_y = fixed_y - ty * scale
          YZ view (X slider):  moving_x = fixed_x - tx * scale
        where scale = 2 ** (self.level - cs_level) converts from working-res pixels to
        cross-section-level pixels.
        """
        state = self._current_state()
        scale = 2 ** (self.level - self._cs_mgr.cs_level)
        if axis == "xz":
            return -round(state.ty * scale)
        return -round(state.tx * scale)

    def _refresh_cross_section(self) -> None:
        """Update the moving layer from the cache at the current slider position.

        The fixed layer is always the static downloaded NPZ — we never overwrite it.
        """
        if not self.pairs or self._projection_mode == "xy":
            return
        _fid, mid = self.pairs[self.current_pair_idx]
        axis = self._projection_mode
        pos = self._cross_section_y if axis == "xz" else self._cross_section_x
        img = self._cs_mgr.get_cached(mid, axis, pos)
        if img is not None and self.moving_layer is not None:
            self.moving_layer.data = normalize_aip(img.astype(np.float32))

    def _on_cs_slider_settled(self) -> None:
        """Called after the debounce timer fires — fetch moving slice at the new position."""
        if not self.pairs or self._projection_mode == "xy":
            return
        _fid, mid = self.pairs[self.current_pair_idx]
        axis = self._projection_mode
        pos = self._cross_section_y if axis == "xz" else self._cross_section_x
        self._cs_mgr.request(mid, axis, pos)
        self._cs_mgr.prefetch_around(mid, axis, pos)

    def _nudge_cs_position(self, delta: int) -> None:
        """Shift the active cross-section slider by *delta* pixels."""
        if self._projection_mode == "xz":
            new_val = max(0, min(self.slider_cs_y.maximum(), self._cross_section_y + delta))
            self.slider_cs_y.setValue(new_val)
        elif self._projection_mode == "yz":
            new_val = max(0, min(self.slider_cs_x.maximum(), self._cross_section_x + delta))
            self.slider_cs_x.setValue(new_val)

    # ----- Navigation -----

    def _prev_pair(self) -> None:
        if self.current_pair_idx > 0:
            self._load_pair_preserve_camera(self.current_pair_idx - 1)

    def _next_pair(self) -> None:
        if self.current_pair_idx < len(self.pairs) - 1:
            self._load_pair_preserve_camera(self.current_pair_idx + 1)

    # ----- Undo / redo -----

    def _undo(self) -> None:
        if not self.pairs:
            return
        mid = self.pairs[self.current_pair_idx][1]
        stack = self.undo_stacks.get(mid)
        if stack:
            state = stack.undo()
            if state:
                self._apply_state(state, push=False)
                self._update_status()

    def _redo(self) -> None:
        if not self.pairs:
            return
        mid = self.pairs[self.current_pair_idx][1]
        stack = self.undo_stacks.get(mid)
        if stack:
            state = stack.redo()
            if state:
                self._apply_state(state, push=False)
                self._update_status()

    # ----- Transform actions -----

    def _load_automated_transform(self) -> None:
        """Load the existing automated transform as starting point."""
        if not self.pairs:
            return
        mid = self.pairs[self.current_pair_idx][1]
        if mid not in self.existing_transforms:
            self.viewer.status = f"No automated transform for z{mid:02d}"
            return

        tfm_dir = self.existing_transforms[mid]
        tfm_files = list(tfm_dir.glob("*.tfm"))
        if not tfm_files:
            self.viewer.status = f"No .tfm file in {tfm_dir}"
            return

        tx, ty, rot, tfm_center = load_transform(tfm_files[0])
        scale = 2**self.level
        img_center = self.pair_centers.get(mid)
        if img_center is not None:
            tx, ty = adjust_for_rotation_center(tx, ty, rot, tfm_center, (img_center[0] * scale, img_center[1] * scale))
        state = AlignmentState(tx=tx / scale, ty=ty / scale, rotation=rot)
        self._apply_state(state, push=True)
        self.viewer.status = f"Loaded automated transform for z{mid:02d}: tx={state.tx:.1f} ty={state.ty:.1f} rot={rot:.2f}°"
        self._update_status()

    def _reset_transform(self) -> None:
        state = AlignmentState()
        self._apply_state(state, push=True)
        self._update_status()

    # ----- Save -----

    def _save_current(self) -> None:
        """Save the current transform for the current pair."""
        if not self.pairs:
            return
        _fid, mid = self.pairs[self.current_pair_idx]
        state = self._current_state()

        cx, cy = self.pair_centers.get(mid, (0.0, 0.0))
        offsets = self._current_offsets.get(mid, (0, 0))

        out_dir = self.output_dir / f"slice_z{mid:02d}"
        save_transform(out_dir, state.tx, state.ty, state.rotation, center=(cx, cy), level=self.level, offsets=offsets)
        self.saved_pairs.add(mid)
        self.unsaved_changes.discard(mid)
        self.viewer.status = f"Saved transform for z{mid:02d} → {out_dir}"
        self._update_status()

    def _save_all_and_exit(self) -> None:
        """Save all modified pairs and close."""
        unsaved = [mid for _fid, mid in self.pairs if mid in self.unsaved_changes and self.undo_stacks.get(mid) is not None]
        total_saved = len(self.saved_pairs)
        msg = QMessageBox(self)
        msg.setWindowTitle("Save All & Exit")
        msg.setIcon(QMessageBox.Question)
        if unsaved:
            msg.setText(
                f"Save {len(unsaved)} modified pair(s) and exit?\n\n{total_saved} pair(s) were already saved in this session."
            )
        else:
            msg.setText(f"No unsaved changes. Exit?\n\n{total_saved} pair(s) were saved in this session.")
        msg.setStandardButtons(QMessageBox.Ok | QMessageBox.Cancel)
        msg.setDefaultButton(QMessageBox.Ok)
        msg.button(QMessageBox.Ok).setText("Save && Exit" if unsaved else "Exit")
        if msg.exec() != QMessageBox.Ok:
            return

        count = 0
        for _fid, mid in self.pairs:
            if mid not in self.unsaved_changes:
                continue
            stack = self.undo_stacks.get(mid)
            if stack is None:
                continue
            state = stack.current

            cx, cy = self.pair_centers.get(mid, (0.0, 0.0))
            offsets = self._current_offsets.get(mid, (0, 0))

            out_dir = self.output_dir / f"slice_z{mid:02d}"
            save_transform(out_dir, state.tx, state.ty, state.rotation, center=(cx, cy), level=self.level, offsets=offsets)
            self.saved_pairs.add(mid)
            count += 1

        self.unsaved_changes.clear()
        self._close_confirmed = True
        self.viewer.status = f"Saved {count} transforms to {self.output_dir}"
        logger.info(f"Saved {count} manual transforms to {self.output_dir}")
        self.viewer.close()

    # ----- Server transfer -----

    def _download_from_server(self) -> None:
        """Download the manual alignment data package from the server (in background thread)."""
        from linumpy_manual_align.server_transfer import download_manual_align_package

        if self.server_config is None:
            return

        self.btn_download.setEnabled(False)
        self.btn_upload.setEnabled(False)
        self.server_progress.show()
        self.server_status_label.setText("<i>Downloading...</i>")
        self.viewer.status = "Downloading data from server..."

        # Always download to the same fixed root so repeated downloads overwrite
        # in-place instead of nesting deeper (aips_dir.parent would keep moving
        # deeper each time because aips_dir is updated after every download).
        local_dir = self.output_dir.parent / "server_package"

        self._worker = ScpWorker(download_manual_align_package, (self.server_config, local_dir, self.level))
        self._worker.finished.connect(lambda ok, msg: self._on_download_finished(ok, msg, local_dir))
        self._worker.start()

    def _on_download_finished(self, ok: bool, msg: str, local_dir: Path) -> None:
        """Handle completion of background download."""
        self.server_progress.hide()
        self.server_status_label.setText(msg)
        self.viewer.status = msg
        self.btn_download.setEnabled(True)
        self.btn_upload.setEnabled(True)
        self._worker = None

        if ok:
            pkg_aips = local_dir / "manual_align_package" / "aips"
            if not pkg_aips.exists():
                pkg_aips = local_dir / "aips"
            if pkg_aips.exists():
                self.aips_dir = pkg_aips
                self.slice_paths = discover_aips(pkg_aips)
                self.slice_ids = list(self.slice_paths.keys())

                # Discover axis-specific AIPs
                self._discover_axis_aip_dirs(pkg_aips.parent)

                # Load remote zarr metadata for interactive cross-section sliders
                self._load_remote_cs_metadata(pkg_aips.parent)

                # Reload transforms too if available
                pkg_tfm = local_dir / "manual_align_package" / "transforms"
                if not pkg_tfm.exists():
                    pkg_tfm = local_dir / "transforms"
                if pkg_tfm.exists():
                    self.transforms_dir = pkg_tfm
                    self.existing_transforms = discover_transforms(pkg_tfm)
                # Rebuild pairs and reload
                self._rebuild_pairs()

    def _upload_to_server(self) -> None:
        """Upload saved manual transforms to the server (in background thread)."""
        from linumpy_manual_align.server_transfer import upload_manual_transforms

        if self.server_config is None:
            return

        if not self.output_dir.exists() or not list(self.output_dir.glob("slice_z*")):
            self.server_status_label.setText("No saved transforms to upload")
            return

        self.btn_download.setEnabled(False)
        self.btn_upload.setEnabled(False)
        self.server_progress.show()
        self.server_status_label.setText("<i>Uploading...</i>")
        self.viewer.status = "Uploading transforms to server..."

        self._worker = ScpWorker(upload_manual_transforms, (self.server_config, self.output_dir))
        self._worker.finished.connect(self._on_upload_finished)
        self._worker.start()

    def _on_upload_finished(self, ok: bool, msg: str) -> None:
        """Handle completion of background upload."""
        self.server_progress.hide()
        self.server_status_label.setText(msg)
        self.viewer.status = msg
        self.btn_download.setEnabled(True)
        self.btn_upload.setEnabled(True)
        self._worker = None

    def _rebuild_pairs(self) -> None:
        """Rebuild pair list after slice discovery changes, then reload UI."""
        self._build_pairs()

        if not self.pairs:
            self.viewer.status = "No slice pairs found after download"
            return

        # Rebuild combo box
        self.pair_combo.blockSignals(True)
        self.pair_combo.clear()
        for fid, mid in self.pairs:
            self.pair_combo.addItem(self._pair_label(fid, mid))
        self.pair_combo.blockSignals(False)

        self.current_pair_idx = 0
        self._load_pair(0)

    # ----- Status display -----

    def _update_status(self) -> None:
        if not self.pairs:
            self.status_label.setText(
                "<i>No data loaded. Use the Server section to download or launch with --data_package.</i>"
            )
            return
        fid, mid = self.pairs[self.current_pair_idx]
        state = self._current_state()
        scale = 2**self.level

        mode_label = {"xy": "XY", "xz": "XZ", "yz": "YZ"}.get(self._projection_mode, "XY")
        lines = [f"<b>Pair {self.current_pair_idx + 1}/{len(self.pairs)}: z{fid:02d} → z{mid:02d}  [{mode_label}]</b>"]
        lines.append(f"Working res (level {self.level}): tx={state.tx:.1f}  ty={state.ty:.1f}  rot={state.rotation:.2f}°")
        lines.append(f"Full res (level 0): tx={state.tx * scale:.1f}  ty={state.ty * scale:.1f}  rot={state.rotation:.2f}°")

        offsets = self._current_offsets.get(mid, (0, 0))
        if offsets != (0, 0):
            lines.append(f"Z offsets: fixed={offsets[0]}  moving={offsets[1]}  (Δ={offsets[0] - offsets[1]:+d})")

        hint = (
            "<i style='color: grey;'>"
            "Arrow: 1px · Alt+Arrow: 10px · Ctrl+Arrow: 50px"
            " · [/]: 0.1° · Alt+[/]: 1° · Ctrl+[/]: 5°"
            "</i>"
        )
        lines.append(hint)

        # Show automated metrics if available
        if mid in self.existing_transforms:
            metrics_path = self.existing_transforms[mid] / "pairwise_registration_metrics.json"
            metrics = load_pairwise_metrics(metrics_path)
            auto_tx = get_metric(metrics, "translation_x")
            auto_ty = get_metric(metrics, "translation_y")
            auto_rot = get_metric(metrics, "rotation")
            auto_mag = get_metric(metrics, "translation_magnitude")
            auto_conf = get_metric(metrics, "registration_confidence")
            auto_zcorr = get_metric(metrics, "z_correlation")

            auto_parts = []
            if auto_tx is not None:
                auto_parts.append(f"tx={auto_tx:.1f}")
            if auto_ty is not None:
                auto_parts.append(f"ty={auto_ty:.1f}")
            if auto_rot is not None:
                auto_parts.append(f"rot={auto_rot:.2f}°")
            if auto_mag is not None:
                auto_parts.append(f"mag={auto_mag:.0f}px")
            if auto_parts:
                lines.append(f"<i>Automated: {', '.join(auto_parts)}</i>")

            quality_parts = []
            if auto_conf is not None:
                quality_parts.append(f"conf={auto_conf:.3f}")
            if auto_zcorr is not None:
                quality_parts.append(f"zcorr={auto_zcorr:.3f}")
            if quality_parts:
                lines.append(f"<i>Quality: {', '.join(quality_parts)}</i>")

        if mid in self.saved_pairs:
            lines.append("<b style='color: green;'>✓ SAVED</b>")

        self.status_label.setText("<br>".join(lines))

    # ----- Keybindings -----

    def _install_keybindings(self) -> None:
        for key, method_name, args in _KEYBINDINGS:
            method = getattr(self, method_name)
            self.viewer.bind_key(key, lambda _v, m=method, a=args: m(*a), overwrite=True)

        # Undo/redo use dedicated macOS-aware labels, so keep them explicit
        self.viewer.bind_key("Control-z", lambda _v: self._undo(), overwrite=True)
        self.viewer.bind_key("Control-Shift-z", lambda _v: self._redo(), overwrite=True)

    def _on_overlay_mode_changed(self, _: int = 0) -> None:
        self._overlay_mode = [OVERLAY_COLOR, OVERLAY_DIFF, OVERLAY_CHECKER][self.combo_overlay.currentIndex()]
        is_checker = self._overlay_mode == OVERLAY_CHECKER
        self._tile_row_label.setVisible(is_checker)
        self.spin_tile.setVisible(is_checker)
        self._rebuild_layer_visibility()
        if self._overlay_mode != OVERLAY_COLOR:
            state = self._current_state()
            self._refresh_composite(state)

    def _on_tile_size_changed(self, _: int = 0) -> None:
        if self._overlay_mode == OVERLAY_CHECKER:
            state = self._current_state()
            self._refresh_composite(state)

    def _on_enhance_changed(self, _: int = 0) -> None:
        """Re-enhance both AIPs from the stored raw data and refresh the display."""
        modes = [ENHANCE_NONE, ENHANCE_EDGES, ENHANCE_CLAHE, ENHANCE_SHARPEN]
        self._enhance_mode = modes[self.combo_enhance.currentIndex()]

        if self._raw_fixed_aip is None or self._raw_moving_aip is None:
            return

        self._original_fixed_aip = enhance_aip(self._raw_fixed_aip, self._enhance_mode)
        self._original_moving_aip = enhance_aip(self._raw_moving_aip, self._enhance_mode)

        if self.fixed_layer is not None:
            self.fixed_layer.data = self._original_fixed_aip

        if self.moving_layer is not None:
            if self._projection_mode == "xy":
                # XY: bake rotation + translation into pixel data
                state = self._current_state()
                self._apply_state(state, push=False)
            else:
                # XZ/YZ: _apply_state only sets translate/rotate and never touches
                # .data, so update it explicitly here.
                self.moving_layer.data = self._original_moving_aip

        # Refresh composite overlay (Difference / Checkerboard) if active
        if self._overlay_mode != OVERLAY_COLOR and self._composite_layer is not None:
            state = self._current_state()
            self._refresh_composite(state)

    def _nudge_translate(self, dx: float, dy: float) -> None:
        if self._projection_mode != "xy" and dy != 0:
            # In Z alignment mode Up/Down adjusts the moving Z-overlap voxel index.
            # Pressing Up (dy=-1) decreases moving_z so the moving layer shifts up,
            # which reveals more of the fixed slice below — intuitive for finding overlap.
            self._nudge_z_offset(int(-dy))
            return
        state = self._current_state()
        if self._projection_mode == "yz":
            # In YZ mode the horizontal axis maps to ty (Y), not tx (X).
            state.ty += dx
        else:
            state.tx += dx
            state.ty += dy
        self._apply_state(state, push=True)
        self._update_status()

    def _nudge_z_offset(self, delta: int) -> None:
        """Adjust the moving Z-overlap voxel index by *delta* (Z alignment mode only).

        When moving_z hits its bound the overflow is absorbed by fixed_z in the
        opposite direction — both changes shift the visual overlap by the same
        amount because dz_display = (moving_z - fixed_z) / scale.
        """
        if not self.pairs:
            return
        new_moving = self.spin_moving_z.value() + delta
        lo, hi = self.spin_moving_z.minimum(), self.spin_moving_z.maximum()
        if new_moving < lo:
            overflow = lo - new_moving  # how many steps past the lower bound
            self.spin_moving_z.setValue(lo)
            new_fixed = min(self.spin_fixed_z.maximum(), self.spin_fixed_z.value() + overflow)
            self.spin_fixed_z.setValue(new_fixed)
        elif new_moving > hi:
            overflow = new_moving - hi  # how many steps past the upper bound
            self.spin_moving_z.setValue(hi)
            new_fixed = max(self.spin_fixed_z.minimum(), self.spin_fixed_z.value() - overflow)
            self.spin_fixed_z.setValue(new_fixed)
        else:
            self.spin_moving_z.setValue(new_moving)

    def _restore_camera(self, zoom: float, center: tuple) -> None:
        """Re-apply a previously snapshotted camera state."""
        self.viewer.camera.zoom = zoom
        self.viewer.camera.center = center

    def _load_pair_preserve_camera(self, idx: int) -> None:
        """Load a pair while keeping the current zoom level and position."""
        zoom = self.viewer.camera.zoom
        center = tuple(self.viewer.camera.center)
        self._load_pair(idx, preserve_camera=True)
        QTimer.singleShot(0, lambda: self._restore_camera(zoom, center))

    def _toggle_z_proj(self) -> None:
        """Switch between XZ and YZ views (Z alignment mode only)."""
        if self._projection_mode == "xy":
            return
        # idClicked only fires on real mouse clicks; block signals and call
        # the handler directly so the display is updated on keyboard use too.
        if self._projection_mode == "xz":
            self._proj_btn_group.blockSignals(True)
            self._btn_proj_yz.setChecked(True)
            self._proj_btn_group.blockSignals(False)
            self._on_z_proj_changed(1)
        else:
            self._proj_btn_group.blockSignals(True)
            self._btn_proj_xz.setChecked(True)
            self._proj_btn_group.blockSignals(False)
            self._on_z_proj_changed(0)

    def _toggle_alignment_mode(self) -> None:
        """Toggle between XY Alignment and Z Alignment modes."""
        # Block signals on both buttons to prevent re-entrant toggled calls,
        # set the visual state, then invoke the handler once directly.
        if self._projection_mode == "xy":
            if not self._btn_mode_z.isEnabled():
                return
            self._btn_mode_xy.blockSignals(True)
            self._btn_mode_z.blockSignals(True)
            self._btn_mode_xy.setChecked(False)
            self._btn_mode_z.setChecked(True)
            self._btn_mode_xy.blockSignals(False)
            self._btn_mode_z.blockSignals(False)
            self._on_mode_btn_toggled("z", True)
        else:
            self._btn_mode_xy.blockSignals(True)
            self._btn_mode_z.blockSignals(True)
            self._btn_mode_xy.setChecked(True)
            self._btn_mode_z.setChecked(False)
            self._btn_mode_xy.blockSignals(False)
            self._btn_mode_z.blockSignals(False)
            self._on_mode_btn_toggled("xy", True)

    def _nudge_rotate(self, delta_deg: float) -> None:
        state = self._current_state()
        state.rotation += delta_deg
        self._apply_state(state, push=True)
        self._update_status()

    def closeEvent(self, event: object) -> None:
        """Terminate all persistent SSH readers on widget close."""
        self._cs_mgr.close_all()
        super().closeEvent(event)  # type: ignore[arg-type]

    # ----- Close guard -----

    def _install_close_guard(self) -> None:
        """Install event filter to warn about unsaved changes on window close."""
        try:
            main_window = self.viewer.window._qt_window
            main_window.installEventFilter(self)
        except AttributeError:
            pass  # napari API changed; skip gracefully

    def eventFilter(self, obj: object, event: object) -> bool:
        """Intercept close events to warn about unsaved changes."""
        from qtpy.QtCore import QEvent as _QEvent

        if isinstance(event, _QEvent) and event.type() == _QEvent.Close and not self._confirm_close():
            event.ignore()
            return True
        return super().eventFilter(obj, event)

    def _confirm_close(self) -> bool:
        """Return True if it's OK to close (no unsaved changes or user confirmed)."""
        if self._close_confirmed:
            return True
        if not self.unsaved_changes:
            return True
        n = len(self.unsaved_changes)
        msg = QMessageBox(self)
        msg.setWindowTitle("Unsaved Changes")
        msg.setText(f"{n} modified pair(s) have unsaved changes.")
        msg.setInformativeText("Save all before closing?")
        msg.setIcon(QMessageBox.Warning)
        save_btn = msg.addButton("Save All && Close", QMessageBox.AcceptRole)
        msg.addButton("Discard && Close", QMessageBox.DestructiveRole)
        cancel_btn = msg.addButton("Cancel", QMessageBox.RejectRole)
        msg.setDefaultButton(save_btn)
        msg.exec_()
        clicked = msg.clickedButton()
        if clicked is cancel_btn:
            return False
        if clicked is save_btn:
            self._save_all_and_exit()
            return False  # _save_all_and_exit handles close itself
        return True  # Discard

    # ----- Server config GUI -----

    def _browse_server_config(self) -> None:
        """Open file dialog to select a nextflow.config for server access."""
        start_dir = str(Path.home() / "Downloads")
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select nextflow.config",
            start_dir,
            "Config files (*.config);;All files (*)",
        )
        if not path:
            return
        self._on_config_selected(Path(path))

    def _on_config_selected(self, config_path: Path) -> None:
        """Parse a nextflow.config and enable server features."""
        from linumpy_manual_align.server_transfer import parse_server_config

        host = self.host_edit.text().strip() or "132.207.157.41"
        cfg = parse_server_config(config_path, host=host)
        if cfg is None:
            self.server_status_label.setText("<b style='color: red;'>Failed to parse config</b>")
            return

        self.server_config = cfg
        self.config_path_edit.setText(str(config_path))
        self.host_edit.setText(cfg.host)
        self.btn_download.setEnabled(True)
        self.btn_upload.setEnabled(True)

        # Re-anchor output_dir to the config's parent so paths resolve correctly
        self.output_dir = config_path.parent / "manual_transforms"

        # Check if a package was already downloaded for this config
        existing = self._find_existing_package()
        if existing is not None:
            self.server_status_label.setText(
                f"Configured: {cfg.subject_id} @ {cfg.host} — existing package found at {existing}, loading…"
            )
            self._load_existing_package(existing)
        else:
            self.server_status_label.setText(f"Configured: {cfg.subject_id} @ {cfg.host}")

    def _find_existing_package(self) -> Path | None:
        """Return the aips/ dir of an already-downloaded package, or None."""
        candidates = [
            self.output_dir.parent / "server_package" / "manual_align_package" / "aips",
            self.output_dir.parent / "server_package" / "aips",
        ]
        for path in candidates:
            if path.exists() and any(path.glob("*.npz")):
                return path
        return None

    def _load_existing_package(self, aips_dir: Path) -> None:
        """Load a previously downloaded package without hitting the server."""
        self.aips_dir = aips_dir
        self.slice_paths = discover_aips(aips_dir)
        self.slice_ids = list(self.slice_paths.keys())

        # Discover axis-specific AIPs
        self._discover_axis_aip_dirs(aips_dir.parent)

        # Load transforms alongside the aips if present
        for tfm_candidate in (aips_dir.parent / "transforms", aips_dir.parent.parent / "transforms"):
            if tfm_candidate.exists():
                self.transforms_dir = tfm_candidate
                self.existing_transforms = discover_transforms(tfm_candidate)
                break

        # Load remote zarr metadata for interactive cross-section mode
        self._load_remote_cs_metadata(aips_dir.parent)

        self._refresh_saved_pairs()
        self._rebuild_pairs()

    def _load_remote_cs_metadata(self, pkg_root: Path) -> None:
        """Delegate metadata loading to the CrossSectionManager."""
        self._cs_mgr.load_metadata(pkg_root)

    def _on_host_changed(self, text: str) -> None:
        """Update server config host when the user edits the host field."""
        if self.server_config is not None:
            self.server_config.host = text.strip()
