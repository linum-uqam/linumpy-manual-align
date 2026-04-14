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
import sys
from pathlib import Path

import napari
import numpy as np
from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSlider,
    QSpinBox,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

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
    enhance_aip,
    normalize_aip,
)
from linumpy_manual_align.omezarr_io import load_aip_from_ome_zarr
from linumpy_manual_align.server_transfer import ScpWorker
from linumpy_manual_align.state import AlignmentState, UndoStack
from linumpy_manual_align.transform_io import (
    discover_slices,
    discover_transforms,
    load_offsets,
    load_pairwise_metrics,
    load_transform,
    save_transform,
)

logger = logging.getLogger(__name__)

_IS_MACOS = sys.platform == "darwin"
_UNDO_LABEL = "Undo (⌘Z)" if _IS_MACOS else "Undo (Ctrl+Z)"
_REDO_LABEL = "Redo (⌘⇧Z)" if _IS_MACOS else "Redo (Ctrl+Shift+Z)"

# ---------------------------------------------------------------------------
# Widget
# ---------------------------------------------------------------------------


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
            self.slice_paths = self._discover_aips(self.aips_dir)
            self._use_precomputed_aips = True
        elif self.input_dir is not None:
            self.slice_paths = discover_slices(self.input_dir)
            self._use_precomputed_aips = False
        else:
            # Empty startup — will be populated after server download
            self.slice_paths = {}
            self._use_precomputed_aips = True

        # Discover axis-specific AIPs (XZ, YZ projections)
        self.slice_paths_xz = self._discover_aips(self.aips_xz_dir) if self.aips_xz_dir else {}
        self.slice_paths_yz = self._discover_aips(self.aips_yz_dir) if self.aips_yz_dir else {}

        self.slice_ids = list(self.slice_paths.keys())
        self.existing_transforms = discover_transforms(self.transforms_dir) if self.transforms_dir else {}

        # Build slice pair list: [(fixed_id, moving_id), ...]
        self.pairs = []
        for i in range(len(self.slice_ids) - 1):
            fid, mid = self.slice_ids[i], self.slice_ids[i + 1]
            if filter_slices is None or mid in filter_slices:
                self.pairs.append((fid, mid))

        if not self.pairs:
            logger.info("Starting in empty state — no slice pairs found. Download data from server.")

        # Per-pair state
        self.undo_stacks: dict[int, UndoStack] = {}  # keyed by moving_id
        self.saved_pairs: set[int] = set()  # moving_ids that have been saved
        self.unsaved_changes: set[int] = set()  # moving_ids with unsaved modifications
        self.pair_centers: dict[int, tuple[float, float]] = {}  # (cx, cy) per moving_id

        # Layers (set during pair loading)
        self.fixed_layer: napari.layers.Image | None = None
        self.moving_layer: napari.layers.Image | None = None
        self._composite_layer: napari.layers.Image | None = None
        # Raw (normalized, pre-enhancement) AIPs kept so we can re-enhance without disk I/O
        self._raw_fixed_aip: np.ndarray | None = None
        self._raw_moving_aip: np.ndarray | None = None
        # Enhanced AIPs — what is actually displayed and used for compositing
        self._original_fixed_aip: np.ndarray | None = None
        self._original_moving_aip: np.ndarray | None = None  # before rotation
        self._suppress_translate_event = False
        self._suppress_spinbox_event = False
        self._suppress_z_offset_event = False
        self._worker: ScpWorker | None = None  # prevent GC of background thread
        self._close_confirmed = False

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

    @staticmethod
    def _form(spacing: int = 4) -> QFormLayout:
        """Return a consistently styled QFormLayout."""
        f = QFormLayout()
        f.setLabelAlignment(Qt.AlignRight | Qt.AlignVCenter)
        f.setFieldGrowthPolicy(QFormLayout.ExpandingFieldsGrow)
        f.setSpacing(spacing)
        return f

    def _build_ui(self) -> None:
        # Outer layout holds a scroll area so the panel never overflows
        outer = QVBoxLayout()
        outer.setContentsMargins(0, 0, 0, 0)
        self.setLayout(outer)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.NoFrame)
        outer.addWidget(scroll)

        content = QWidget()
        layout = QVBoxLayout()
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(6)
        content.setLayout(layout)
        scroll.setWidget(content)

        # ── Pair navigation ───────────────────────────────────────────────────
        nav_row = QHBoxLayout()
        nav_row.setSpacing(4)

        self.btn_prev = QPushButton("◀ Prev (P)")
        self.btn_prev.clicked.connect(self._prev_pair)
        nav_row.addWidget(self.btn_prev)

        self.pair_combo = QComboBox()
        for fid, mid in self.pairs:
            label = f"z{fid:02d} → z{mid:02d}"
            if mid in self.existing_transforms:
                metrics_path = self.existing_transforms[mid] / "pairwise_registration_metrics.json"
                metrics = load_pairwise_metrics(metrics_path)
                mag = self._get_metric(metrics, "translation_magnitude")
                if mag is not None:
                    label += f"  ({mag:.0f}px)"
            self.pair_combo.addItem(label)
        self.pair_combo.currentIndexChanged.connect(self._on_pair_changed)
        nav_row.addWidget(self.pair_combo, stretch=1)

        self.btn_next = QPushButton("Next (N) ▶")
        self.btn_next.clicked.connect(self._next_pair)
        nav_row.addWidget(self.btn_next)

        layout.addLayout(nav_row)

        # ── Mode toggle buttons ───────────────────────────────────────────────
        mode_row = QHBoxLayout()
        mode_row.setSpacing(0)

        self._btn_mode_xy = QPushButton("XY Alignment")
        self._btn_mode_xy.setCheckable(True)
        self._btn_mode_xy.setChecked(True)
        self._btn_mode_xy.setToolTip("Lateral alignment: adjust TX, TY, and Rotation")
        self._btn_mode_xy.toggled.connect(lambda checked: self._on_mode_btn_toggled("xy", checked))
        mode_row.addWidget(self._btn_mode_xy)

        self._btn_mode_z = QPushButton("Z Alignment")
        self._btn_mode_z.setCheckable(True)
        self._btn_mode_z.setChecked(False)
        self._btn_mode_z.setToolTip("Depth alignment: adjust Z-overlap offsets, view XZ/YZ cross-sections")
        self._btn_mode_z.toggled.connect(lambda checked: self._on_mode_btn_toggled("z", checked))
        mode_row.addWidget(self._btn_mode_z)

        has_axis_aips = bool(self.slice_paths_xz or self.slice_paths_yz)
        if not has_axis_aips:
            self._btn_mode_z.setEnabled(False)
            self._btn_mode_z.setToolTip("XZ/YZ projections not available — regenerate the data package to enable")

        layout.addLayout(mode_row)

        # ── Stacked content (sizes to active page only) ───────────────────────
        self._mode_stack = QStackedWidget()
        layout.addWidget(self._mode_stack)

        # Page 0 - XY Alignment ───────────────────────────────────────────────
        xy_page = QWidget()
        xy_layout = QVBoxLayout()
        xy_layout.setContentsMargins(0, 4, 0, 0)
        xy_layout.setSpacing(4)
        xy_page.setLayout(xy_layout)

        xy_form = self._form()
        self.spin_tx = QDoubleSpinBox()
        self.spin_tx.setRange(-2000, 2000)
        self.spin_tx.setDecimals(1)
        self.spin_tx.setSingleStep(1.0)
        self.spin_tx.setSuffix(" px")
        self.spin_tx.valueChanged.connect(self._on_spinbox_changed)
        xy_form.addRow("TX:", self.spin_tx)

        self.spin_ty = QDoubleSpinBox()
        self.spin_ty.setRange(-2000, 2000)
        self.spin_ty.setDecimals(1)
        self.spin_ty.setSingleStep(1.0)
        self.spin_ty.setSuffix(" px")
        self.spin_ty.valueChanged.connect(self._on_spinbox_changed)
        xy_form.addRow("TY:", self.spin_ty)

        self.spin_rot = QDoubleSpinBox()
        self.spin_rot.setRange(-180, 180)
        self.spin_rot.setDecimals(2)
        self.spin_rot.setSingleStep(0.1)
        self.spin_rot.setSuffix("°")
        self.spin_rot.valueChanged.connect(self._on_rotation_changed)
        xy_form.addRow("Rotation:", self.spin_rot)

        self.rot_slider = QSlider(Qt.Horizontal)
        self.rot_slider.setRange(-1800, 1800)
        self.rot_slider.setValue(0)
        self.rot_slider.valueChanged.connect(self._on_rotation_slider_changed)
        xy_form.addRow("", self.rot_slider)

        xy_layout.addLayout(xy_form)

        xy_hint = QLabel(
            "<i style='color: grey;'>Arrow: 1px · Shift: 10px · Ctrl: 50px · [/]: 0.1° · Shift[/]: 1° · Ctrl[/]: 5°</i>"
        )
        xy_hint.setWordWrap(True)
        xy_layout.addWidget(xy_hint)

        row1 = QHBoxLayout()
        self.btn_load_auto = QPushButton("Load Automated")
        self.btn_load_auto.clicked.connect(self._load_automated_transform)
        row1.addWidget(self.btn_load_auto)
        self.btn_reset = QPushButton("Reset")
        self.btn_reset.clicked.connect(self._reset_transform)
        row1.addWidget(self.btn_reset)
        xy_layout.addLayout(row1)

        row2 = QHBoxLayout()
        self.btn_undo = QPushButton(_UNDO_LABEL)
        self.btn_undo.clicked.connect(self._undo)
        row2.addWidget(self.btn_undo)
        self.btn_redo = QPushButton(_REDO_LABEL)
        self.btn_redo.clicked.connect(self._redo)
        row2.addWidget(self.btn_redo)
        xy_layout.addLayout(row2)

        self._mode_stack.addWidget(xy_page)

        # Page 1 - Z Alignment ────────────────────────────────────────────────
        z_page = QWidget()
        z_layout = QVBoxLayout()
        z_layout.setContentsMargins(0, 4, 0, 0)
        z_layout.setSpacing(4)
        z_page.setLayout(z_layout)

        z_form = self._form()

        self.proj_combo = QComboBox()
        self.proj_combo.addItem("XZ — front cross-section")
        self.proj_combo.addItem("YZ — side cross-section")
        self.proj_combo.setToolTip("Switch between front (XZ) and side (YZ) cross-section views")
        self.proj_combo.currentIndexChanged.connect(self._on_z_proj_changed)
        z_form.addRow("View:", self.proj_combo)

        self.spin_fixed_z = QSpinBox()
        self.spin_fixed_z.setRange(0, 200)
        self.spin_fixed_z.setSuffix(" voxels")
        self.spin_fixed_z.setToolTip("Z-index in the fixed (bottom/green) slice where overlap begins")
        self.spin_fixed_z.valueChanged.connect(self._on_z_offset_changed)
        z_form.addRow("Fixed Z start:", self.spin_fixed_z)

        self.spin_moving_z = QSpinBox()
        self.spin_moving_z.setRange(0, 200)
        self.spin_moving_z.setSuffix(" voxels")
        self.spin_moving_z.setToolTip("Z-index in the moving (top/red) slice where overlap begins")
        self.spin_moving_z.valueChanged.connect(self._on_z_offset_changed)
        z_form.addRow("Moving Z start:", self.spin_moving_z)

        self.z_relative_label = QLabel("Relative shift: 0 voxels")
        z_form.addRow("", self.z_relative_label)

        z_layout.addLayout(z_form)

        z_hint = QLabel("<i style='color: grey;'>Align tissue boundaries visible in the cross-section overlay.</i>")
        z_hint.setWordWrap(True)
        z_layout.addWidget(z_hint)

        self._mode_stack.addWidget(z_page)

        # ── Display ───────────────────────────────────────────────────────────
        disp_group = QGroupBox("Display")
        disp_form = self._form()
        disp_group.setLayout(disp_form)

        self.combo_overlay = QComboBox()
        self.combo_overlay.addItems(["Color (R/G)", "Difference", "Checkerboard"])
        self.combo_overlay.setToolTip(
            "Color: additive red/green overlay\n"
            "Difference: |fixed - moving| grayscale (misalignment = bright)\n"
            "Checkerboard: alternating tiles of fixed/moving"
        )
        self.combo_overlay.currentIndexChanged.connect(self._on_overlay_mode_changed)
        disp_form.addRow("Overlay:", self.combo_overlay)

        self.combo_enhance = QComboBox()
        self.combo_enhance.addItems(["None", "Edges (Sobel)", "CLAHE", "Sharpen"])
        self.combo_enhance.setToolTip(
            "None: display normalized AIP as-is\n"
            "Edges: Sobel gradient magnitude - highlights tissue boundaries\n"
            "  Best for oblique/angled cuts where edges are the key landmark\n"
            "CLAHE: adaptive histogram equalization - equalises local contrast\n"
            "  Best for sagittal cuts where projection blur hides boundaries\n"
            "Sharpen: unsharp mask - mild crispening for blurry projections"
        )
        self.combo_enhance.currentIndexChanged.connect(self._on_enhance_changed)
        disp_form.addRow("Enhance:", self.combo_enhance)

        self.spin_tile = QSpinBox()
        self.spin_tile.setRange(2, 512)
        self.spin_tile.setValue(16)
        self.spin_tile.setSingleStep(4)
        self.spin_tile.setToolTip("Checkerboard tile size (pixels at current pyramid level)")
        self.spin_tile.valueChanged.connect(self._on_tile_size_changed)
        self._tile_row_label = QLabel("Tile size:")
        disp_form.addRow(self._tile_row_label, self.spin_tile)
        self._tile_row_label.setVisible(False)
        self.spin_tile.setVisible(False)

        self.spin_gamma = QDoubleSpinBox()
        self.spin_gamma.setRange(0.1, 2.0)
        self.spin_gamma.setDecimals(2)
        self.spin_gamma.setSingleStep(0.05)
        self.spin_gamma.setValue(0.6)
        self.spin_gamma.setToolTip("Gamma correction (< 1 boosts midtones)")
        self.spin_gamma.valueChanged.connect(self._on_gamma_changed)
        disp_form.addRow("Gamma:", self.spin_gamma)

        self.spin_opacity_moving = QDoubleSpinBox()
        self.spin_opacity_moving.setRange(0.1, 1.0)
        self.spin_opacity_moving.setDecimals(2)
        self.spin_opacity_moving.setSingleStep(0.05)
        self.spin_opacity_moving.setValue(1.0)
        self.spin_opacity_moving.setToolTip("Opacity of the moving (red) layer")
        self.spin_opacity_moving.valueChanged.connect(self._on_opacity_moving_changed)
        disp_form.addRow("Moving opacity:", self.spin_opacity_moving)

        self.spin_opacity_fixed = QDoubleSpinBox()
        self.spin_opacity_fixed.setRange(0.1, 1.0)
        self.spin_opacity_fixed.setDecimals(2)
        self.spin_opacity_fixed.setSingleStep(0.05)
        self.spin_opacity_fixed.setValue(1.0)
        self.spin_opacity_fixed.setToolTip("Opacity of the fixed (green) layer")
        self.spin_opacity_fixed.valueChanged.connect(self._on_opacity_fixed_changed)
        disp_form.addRow("Fixed opacity:", self.spin_opacity_fixed)

        layout.addWidget(disp_group)

        # ── Save ──────────────────────────────────────────────────────────────
        save_row = QHBoxLayout()
        self.btn_save = QPushButton("Save Current (S)")
        self.btn_save.clicked.connect(self._save_current)
        save_row.addWidget(self.btn_save)
        self.btn_save_all = QPushButton("Save All && Exit")
        self.btn_save_all.clicked.connect(self._save_all_and_exit)
        save_row.addWidget(self.btn_save_all)
        layout.addLayout(save_row)

        # ── Server ────────────────────────────────────────────────────────────
        server_group = QGroupBox("Server")
        server_layout = QVBoxLayout()
        server_layout.setSpacing(4)
        server_group.setLayout(server_layout)

        cfg_row = QHBoxLayout()
        cfg_row.addWidget(QLabel("Config:"))
        self.config_path_edit = QLineEdit()
        self.config_path_edit.setPlaceholderText("nextflow.config path…")
        self.config_path_edit.setReadOnly(True)
        cfg_row.addWidget(self.config_path_edit, stretch=1)
        self.btn_browse_config = QPushButton("Browse…")
        self.btn_browse_config.clicked.connect(self._browse_server_config)
        cfg_row.addWidget(self.btn_browse_config)
        server_layout.addLayout(cfg_row)

        host_row = QHBoxLayout()
        host_row.addWidget(QLabel("Host:"))
        self.host_edit = QLineEdit("132.207.157.41")
        self.host_edit.setPlaceholderText("server hostname or IP")
        self.host_edit.textChanged.connect(self._on_host_changed)
        host_row.addWidget(self.host_edit, stretch=1)
        server_layout.addLayout(host_row)

        srv_btn_row = QHBoxLayout()
        self.btn_download = QPushButton("⬇ Download Data")
        self.btn_download.setToolTip("Download AIPs and transforms from the server.")
        self.btn_download.clicked.connect(self._download_from_server)
        srv_btn_row.addWidget(self.btn_download)
        self.btn_upload = QPushButton("⬆ Upload Transforms")
        self.btn_upload.setToolTip("Upload saved manual transforms to the server.")
        self.btn_upload.clicked.connect(self._upload_to_server)
        srv_btn_row.addWidget(self.btn_upload)
        server_layout.addLayout(srv_btn_row)

        self.server_progress = QProgressBar()
        self.server_progress.setRange(0, 0)
        self.server_progress.setTextVisible(False)
        self.server_progress.setFixedHeight(6)
        self.server_progress.hide()
        server_layout.addWidget(self.server_progress)

        self.server_status_label = QLabel("")
        self.server_status_label.setWordWrap(True)
        server_layout.addWidget(self.server_status_label)

        if self.server_config is not None:
            if hasattr(self.server_config, "config_path") and self.server_config.config_path:
                self.config_path_edit.setText(str(self.server_config.config_path))
            self.host_edit.setText(self.server_config.host)
        else:
            self.btn_download.setEnabled(False)
            self.btn_upload.setEnabled(False)
            self.server_status_label.setText("<i>Browse for a nextflow.config to enable server features</i>")

        layout.addWidget(server_group)

        # ── Status ────────────────────────────────────────────────────────────
        self.status_label = QLabel("")
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)

    # ----- AIP discovery -----

    @staticmethod
    def _discover_aips(aips_dir: Path) -> dict[int, Path]:
        """Discover pre-computed AIP .npz files."""
        import re

        pattern = re.compile(r"slice_z(\d+)")
        aips = {}
        for p in sorted(aips_dir.iterdir()):
            m = pattern.search(p.name)
            if m and p.name.endswith(".npz"):
                aips[int(m.group(1))] = p
        return dict(sorted(aips.items()))

    def _discover_axis_aip_dirs(self, pkg_root: Path) -> None:
        """Discover aips_xz/ and aips_yz/ directories in the package root."""
        for name, attr in [("aips_xz", "aips_xz_dir"), ("aips_yz", "aips_yz_dir")]:
            candidate = pkg_root / name
            if candidate.exists():
                setattr(self, attr, candidate)
                paths = self._discover_aips(candidate)
                if name == "aips_xz":
                    self.slice_paths_xz = paths
                else:
                    self.slice_paths_yz = paths

        # Enable the Z Alignment button if axis AIPs are now available
        has_axis_aips = bool(self.slice_paths_xz or self.slice_paths_yz)
        self._btn_mode_z.setEnabled(has_axis_aips)
        if has_axis_aips:
            self._btn_mode_z.setToolTip("Depth alignment: adjust Z-overlap offsets, view XZ/YZ cross-sections")

    # ----- Pair loading -----

    def _get_automated_state(self, mid: int) -> AlignmentState:
        """Get the automated transform for a slice as an AlignmentState, or zero state if unavailable."""
        if mid not in self.existing_transforms:
            return AlignmentState()
        tfm_dir = self.existing_transforms[mid]
        tfm_files = list(tfm_dir.glob("*.tfm"))
        if not tfm_files:
            return AlignmentState()
        tx, ty, rot, tfm_center = load_transform(tfm_files[0])
        scale = 2**self.level
        # Adjust for rotation center difference between automated transform and our image center
        img_center = self.pair_centers.get(mid)
        if img_center is not None and abs(rot) > 0.01:
            img_cx = img_center[0] * scale
            img_cy = img_center[1] * scale
            dcx = tfm_center[0] - img_cx
            dcy = tfm_center[1] - img_cy
            rad = np.radians(rot)
            cos_r, sin_r = np.cos(rad), np.sin(rad)
            tx += (1 - cos_r) * dcx + sin_r * dcy
            ty += -sin_r * dcx + (1 - cos_r) * dcy
        return AlignmentState(tx=tx / scale, ty=ty / scale, rotation=rot)

    def _load_pair(self, idx: int) -> None:
        """Load a slice pair and display as red/green AIP overlay."""
        self.current_pair_idx = idx
        fid, mid = self.pairs[idx]

        self.viewer.status = f"Loading z{fid:02d} / z{mid:02d}..."

        # Select AIP paths based on projection mode
        if self._projection_mode == "xz" and self.slice_paths_xz:
            aip_paths = self.slice_paths_xz
        elif self._projection_mode == "yz" and self.slice_paths_yz:
            aip_paths = self.slice_paths_yz
        else:
            aip_paths = self.slice_paths

        if fid not in aip_paths or mid not in aip_paths:
            logger.warning(f"No {self._projection_mode.upper()} AIPs for pair z{fid:02d}→z{mid:02d}")
            aip_paths = self.slice_paths  # fall back to XY

        if self._use_precomputed_aips:
            fixed_aip, fixed_scale_yx = self._load_aip_from_npz(aip_paths[fid])
            moving_aip, moving_scale_yx = self._load_aip_from_npz(aip_paths[mid])
        else:
            fixed_aip, fixed_scale_yx = self._load_aip_from_zarr(self.slice_paths[fid])
            moving_aip, moving_scale_yx = self._load_aip_from_zarr(self.slice_paths[mid])

        # Normalize to [0, 1] for display
        fixed_aip = self._normalize(fixed_aip)
        moving_aip = self._normalize(moving_aip)

        # Store raw (normalized) AIPs so enhancement can be changed without reloading from disk
        self._raw_fixed_aip = fixed_aip.copy()
        self._raw_moving_aip = moving_aip.copy()

        # Apply current enhancement to derive the display AIPs
        fixed_aip = enhance_aip(fixed_aip, self._enhance_mode)
        moving_aip = enhance_aip(moving_aip, self._enhance_mode)

        self._original_fixed_aip = fixed_aip.copy()
        self._original_moving_aip = moving_aip.copy()

        # Store image center for this pair (only from XY view, needed for rotation)
        if self._projection_mode == "xy":
            self.pair_centers[mid] = (moving_aip.shape[1] / 2.0, moving_aip.shape[0] / 2.0)  # (cx, cy)

        # Remove existing layers (including stale composite)
        self._composite_layer = None
        while len(self.viewer.layers) > 0:
            self.viewer.layers.pop(0)

        # Compute shared contrast limits from both AIPs
        clim = (0.0, 1.0)
        gamma = self.spin_gamma.value()
        opacity_fixed = self.spin_opacity_fixed.value()
        opacity_moving = self.spin_opacity_moving.value()

        view_suffix = {"xy": "", "xz": " (XZ)", "yz": " (YZ)"}.get(self._projection_mode, "")

        # Always add both individual layers (hidden in non-color modes)
        self.fixed_layer = self.viewer.add_image(
            fixed_aip,
            name=f"Fixed z{fid:02d}{view_suffix}",
            colormap="green",
            blending="additive",
            contrast_limits=clim,
            gamma=gamma,
            opacity=opacity_fixed,
            scale=fixed_scale_yx,
        )
        self.moving_layer = self.viewer.add_image(
            moving_aip,
            name=f"Moving z{mid:02d}{view_suffix}",
            colormap="red",
            blending="additive",
            contrast_limits=clim,
            gamma=gamma,
            opacity=opacity_moving,
            scale=moving_scale_yx,
        )

        # Adjust visibility / add composite layer based on current overlay mode
        self._rebuild_layer_visibility()

        # Connect layer translate events for bidirectional sync (XY mode only)
        self.moving_layer.events.translate.connect(self._on_layer_translated)

        # Initialize or restore undo stack
        if mid not in self.undo_stacks:
            initial = self._get_automated_state(mid)
            self.undo_stacks[mid] = UndoStack(initial)

        # Load Z-offsets for this pair
        if mid not in self._current_offsets:
            offsets = (0, 0)
            if mid in self.existing_transforms:
                offsets = load_offsets(self.existing_transforms[mid] / "offsets.txt")
            self._current_offsets[mid] = offsets

        # Update Z-offset spinboxes
        self._suppress_z_offset_event = True
        self.spin_fixed_z.setValue(self._current_offsets[mid][0])
        self.spin_moving_z.setValue(self._current_offsets[mid][1])
        self._suppress_z_offset_event = False
        self._update_z_relative_label()

        state = self.undo_stacks[mid].current
        self._apply_state(state, push=False)

        # Update UI
        self._suppress_spinbox_event = True
        self.pair_combo.setCurrentIndex(idx)
        self._suppress_spinbox_event = False
        self._update_status()

        self.viewer.reset_view()
        self.viewer.status = f"Pair z{fid:02d} → z{mid:02d} loaded ({self._projection_mode.upper()} view)"

    def _load_aip_from_npz(self, npz_path: Path) -> tuple[np.ndarray, list[float]]:
        """Load pre-computed AIP from an .npz file."""
        data = np.load(str(npz_path))
        aip = data["aip"].astype(np.float32)
        scale = data["scale"]
        # Scale is (Z, Y, X) from OME-Zarr; return YX only
        scale_yx = list(scale[1:]) if len(scale) == 3 else list(scale)
        return aip, scale_yx

    def _load_aip_from_zarr(self, zarr_path: Path) -> tuple[np.ndarray, list[float]]:
        """Load a volume from OME-Zarr and compute the AIP."""
        return load_aip_from_ome_zarr(zarr_path, level=self.level)

    def _normalize(self, img: np.ndarray) -> np.ndarray:
        return normalize_aip(img)

    # ----- Overlay helpers -----

    def _make_shifted_moving(self, state: AlignmentState) -> np.ndarray:
        """Return the moving AIP with rotation + pixel-level shift baked in."""
        moving = self._original_moving_aip
        if moving is None:
            return np.zeros((1, 1), dtype=np.float32)
        return apply_transform(moving, rotation=state.rotation, tx=state.tx, ty=state.ty)

    def _compute_composite(self, fixed: np.ndarray, shifted_moving: np.ndarray) -> np.ndarray:
        """Build a composite image for Difference or Checkerboard overlay modes."""
        return build_overlay(fixed, shifted_moving, mode=self._overlay_mode, tile_size=self.spin_tile.value())

    def _rebuild_layer_visibility(self) -> None:
        """Show/hide individual layers and manage composite layer for current overlay mode."""
        is_color = self._overlay_mode == OVERLAY_COLOR

        if self.fixed_layer is not None:
            self.fixed_layer.visible = is_color
        if self.moving_layer is not None:
            self.moving_layer.visible = is_color

        # Remove stale composite layer
        if self._composite_layer is not None:
            with contextlib.suppress(ValueError):
                self.viewer.layers.remove(self._composite_layer)
            self._composite_layer = None

        if not is_color and self._original_fixed_aip is not None and self.fixed_layer is not None:
            colormap = "inferno" if self._overlay_mode == OVERLAY_DIFF else "gray"
            comp = np.zeros_like(self._original_fixed_aip)
            self._composite_layer = self.viewer.add_image(
                comp,
                name="Composite",
                colormap=colormap,
                blending="translucent",
                contrast_limits=(0.0, 1.0),
                gamma=self.spin_gamma.value(),
                scale=list(self.fixed_layer.scale),
            )

    def _refresh_composite(self, state: AlignmentState) -> None:
        """Recompute and push composite image data for non-color overlay modes."""
        if self._overlay_mode == OVERLAY_COLOR or self._composite_layer is None:
            return
        if self._original_fixed_aip is None or self._original_moving_aip is None:
            return
        shifted = self._make_shifted_moving(state)
        self._composite_layer.data = self._compute_composite(self._original_fixed_aip, shifted)

    # ----- State application -----

    def _apply_state(self, state: AlignmentState, push: bool = True) -> None:
        """Apply an alignment state to the moving layer."""
        if self.moving_layer is None or self._original_moving_aip is None or not self.pairs:
            return

        mid = self.pairs[self.current_pair_idx][1]

        if push:
            self.undo_stacks[mid].push(state)
            self.unsaved_changes.add(mid)

        if self._projection_mode == "xy":
            # Apply rotation to image data (translation handled via layer.translate)
            rotated = apply_transform(self._original_moving_aip, rotation=state.rotation)
            self.moving_layer.data = rotated

            # Apply translation via layer translate
            scale = self.moving_layer.scale
            self._suppress_translate_event = True
            self.moving_layer.translate = [state.ty * scale[0], state.tx * scale[1]]
            self._suppress_translate_event = False
        else:
            # XZ/YZ mode: no rotation, show Z-offset as vertical shift
            self.moving_layer.data = self._original_moving_aip.copy()

            offsets = self._current_offsets.get(mid, (0, 0))
            scale_factor = 2**self.level
            dz_display = (offsets[0] - offsets[1]) / scale_factor

            scale = self.moving_layer.scale
            self._suppress_translate_event = True
            if self._projection_mode == "xz":
                self.moving_layer.translate = [dz_display * scale[0], state.tx * scale[1]]
            else:  # yz
                self.moving_layer.translate = [dz_display * scale[0], state.ty * scale[1]]
            self._suppress_translate_event = False

        # Update composite overlay if active
        if self._overlay_mode != OVERLAY_COLOR and self._projection_mode == "xy":
            self._refresh_composite(state)

        # Sync spinboxes
        self._suppress_spinbox_event = True
        self.spin_tx.setValue(state.tx)
        self.spin_ty.setValue(state.ty)
        self.spin_rot.setValue(state.rotation)
        self.rot_slider.setValue(int(state.rotation * 10))
        self._suppress_spinbox_event = False

    def _current_state(self) -> AlignmentState:
        return AlignmentState(
            tx=self.spin_tx.value(),
            ty=self.spin_ty.value(),
            rotation=self.spin_rot.value(),
        )

    # ----- Event handlers -----

    def _on_pair_changed(self, idx: int) -> None:
        if idx >= 0 and idx != self.current_pair_idx:
            self._load_pair(idx)

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
        self._suppress_spinbox_event = True
        self.rot_slider.setValue(int(state.rotation * 10))
        self._suppress_spinbox_event = False
        self._apply_state(state, push=True)
        self._update_status()

    def _on_rotation_slider_changed(self, value: int) -> None:
        if self._suppress_spinbox_event:
            return
        rot = value / 10.0
        self._suppress_spinbox_event = True
        self.spin_rot.setValue(rot)
        self._suppress_spinbox_event = False
        state = AlignmentState(tx=self.spin_tx.value(), ty=self.spin_ty.value(), rotation=rot)
        self._apply_state(state, push=True)
        self._update_status()

    def _on_layer_translated(self, _event) -> None:
        """Sync spinboxes when the user drags the moving layer."""
        if self._suppress_translate_event or self.moving_layer is None:
            return

        # In XZ/YZ mode, ignore drag events (the vertical axis is controlled by Z-offsets)
        if self._projection_mode != "xy":
            state = self._current_state()
            self._apply_state(state, push=False)
            return

        translate = self.moving_layer.translate
        scale = self.moving_layer.scale
        ty = translate[0] / scale[0]
        tx = translate[1] / scale[1]

        self._suppress_spinbox_event = True
        self.spin_tx.setValue(tx)
        self.spin_ty.setValue(ty)
        self._suppress_spinbox_event = False

        mid = self.pairs[self.current_pair_idx][1]
        state = AlignmentState(tx=tx, ty=ty, rotation=self.spin_rot.value())
        self.undo_stacks[mid].push(state)
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
            z_proj_idx = self.proj_combo.currentIndex()
            self._projection_mode = "xz" if z_proj_idx == 0 else "yz"
            self._mode_stack.setCurrentIndex(1)
        if self.pairs:
            self._load_pair(self.current_pair_idx)

    def _on_z_proj_changed(self, idx: int) -> None:
        """Switch between XZ and YZ within the Z Alignment page."""
        if self._projection_mode == "xy":
            return  # ignore if XY mode is active
        self._projection_mode = "xz" if idx == 0 else "yz"
        if self.pairs:
            self._load_pair(self.current_pair_idx)

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

    # ----- Navigation -----

    def _prev_pair(self) -> None:
        if self.current_pair_idx > 0:
            self._load_pair(self.current_pair_idx - 1)

    def _next_pair(self) -> None:
        if self.current_pair_idx < len(self.pairs) - 1:
            self._load_pair(self.current_pair_idx + 1)

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
        # Adjust for rotation center difference
        img_center = self.pair_centers.get(mid)
        if img_center is not None and abs(rot) > 0.01:
            img_cx = img_center[0] * scale
            img_cy = img_center[1] * scale
            dcx = tfm_center[0] - img_cx
            dcy = tfm_center[1] - img_cy
            rad = np.radians(rot)
            cos_r, sin_r = np.cos(rad), np.sin(rad)
            tx += (1 - cos_r) * dcx + sin_r * dcy
            ty += -sin_r * dcx + (1 - cos_r) * dcy
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

        # Determine local destination
        local_dir = self.aips_dir.parent if self.aips_dir is not None else self.output_dir.parent / "server_package"

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
                self.slice_paths = self._discover_aips(pkg_aips)
                self._use_precomputed_aips = True
                self.slice_ids = list(self.slice_paths.keys())

                # Discover axis-specific AIPs
                self._discover_axis_aip_dirs(pkg_aips.parent)

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
        self.pairs = []
        for i in range(len(self.slice_ids) - 1):
            fid, mid = self.slice_ids[i], self.slice_ids[i + 1]
            self.pairs.append((fid, mid))

        if not self.pairs:
            self.viewer.status = "No slice pairs found after download"
            return

        # Rebuild combo box
        self.pair_combo.blockSignals(True)
        self.pair_combo.clear()
        for fid, mid in self.pairs:
            label = f"z{fid:02d} → z{mid:02d}"
            if mid in self.existing_transforms:
                metrics_path = self.existing_transforms[mid] / "pairwise_registration_metrics.json"
                metrics = load_pairwise_metrics(metrics_path)
                mag = self._get_metric(metrics, "translation_magnitude")
                if mag is not None:
                    label += f"  (mag={mag:.0f}px)"
            self.pair_combo.addItem(label)
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
            "Arrow: 1px · Shift+Arrow: 10px · Ctrl+Arrow: 50px"
            " · [/]: 0.1° · Shift+[/]: 1° · Ctrl+[/]: 5°"
            "</i>"
        )
        lines.append(hint)

        # Show automated metrics if available
        if mid in self.existing_transforms:
            metrics_path = self.existing_transforms[mid] / "pairwise_registration_metrics.json"
            metrics = load_pairwise_metrics(metrics_path)
            auto_tx = self._get_metric(metrics, "translation_x")
            auto_ty = self._get_metric(metrics, "translation_y")
            auto_rot = self._get_metric(metrics, "rotation")
            auto_mag = self._get_metric(metrics, "translation_magnitude")
            auto_conf = self._get_metric(metrics, "registration_confidence")
            auto_zcorr = self._get_metric(metrics, "z_correlation")

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

    @staticmethod
    def _get_metric(metrics: dict, key: str) -> float | None:
        try:
            return float(metrics["metrics"][key]["value"])
        except (KeyError, TypeError, ValueError):
            return None

    # ----- Keybindings -----

    def _install_keybindings(self) -> None:
        viewer = self.viewer

        @viewer.bind_key("n")
        def _next(viewer):
            self._next_pair()

        @viewer.bind_key("p")
        def _prev(viewer):
            self._prev_pair()

        @viewer.bind_key("s")
        def _save(viewer):
            self._save_current()

        # Fine translation (1px at working resolution)
        @viewer.bind_key("Right")
        def _right(viewer):
            self._nudge_translate(1, 0)

        @viewer.bind_key("Left")
        def _left(viewer):
            self._nudge_translate(-1, 0)

        @viewer.bind_key("Up")
        def _up(viewer):
            self._nudge_translate(0, -1)

        @viewer.bind_key("Down")
        def _down(viewer):
            self._nudge_translate(0, 1)

        # Coarse translation (10px)
        @viewer.bind_key("Shift-Right")
        def _right10(viewer):
            self._nudge_translate(10, 0)

        @viewer.bind_key("Shift-Left")
        def _left10(viewer):
            self._nudge_translate(-10, 0)

        @viewer.bind_key("Shift-Up")
        def _up10(viewer):
            self._nudge_translate(0, -10)

        @viewer.bind_key("Shift-Down")
        def _down10(viewer):
            self._nudge_translate(0, 10)

        # Large translation (50px)
        @viewer.bind_key("Control-Right")
        def _right50(viewer):
            self._nudge_translate(50, 0)

        @viewer.bind_key("Control-Left")
        def _left50(viewer):
            self._nudge_translate(-50, 0)

        @viewer.bind_key("Control-Up")
        def _up50(viewer):
            self._nudge_translate(0, -50)

        @viewer.bind_key("Control-Down")
        def _down50(viewer):
            self._nudge_translate(0, 50)

        # Rotation
        @viewer.bind_key("]")
        def _rot_cw(viewer):
            self._nudge_rotate(0.1)

        @viewer.bind_key("[")
        def _rot_ccw(viewer):
            self._nudge_rotate(-0.1)

        @viewer.bind_key("Shift-]")
        def _rot_cw_coarse(viewer):
            self._nudge_rotate(1.0)

        @viewer.bind_key("Shift-[")
        def _rot_ccw_coarse(viewer):
            self._nudge_rotate(-1.0)

        @viewer.bind_key("Control-]")
        def _rot_cw_large(viewer):
            self._nudge_rotate(5.0)

        @viewer.bind_key("Control-[")
        def _rot_ccw_large(viewer):
            self._nudge_rotate(-5.0)

        # Undo/redo (Ctrl+Z / Ctrl+Shift+Z)
        @viewer.bind_key("Control-z")
        def _undo(viewer):
            self._undo()

        @viewer.bind_key("Control-Shift-z")
        def _redo(viewer):
            self._redo()

    def _on_gamma_changed(self, value: float) -> None:
        if self._overlay_mode == OVERLAY_COLOR:
            if self.fixed_layer is not None:
                self.fixed_layer.gamma = value
            if self.moving_layer is not None:
                self.moving_layer.gamma = value
        elif self._composite_layer is not None:
            self._composite_layer.gamma = value

    def _on_opacity_moving_changed(self, value: float) -> None:
        if self._overlay_mode == OVERLAY_COLOR and self.moving_layer is not None:
            self.moving_layer.opacity = value

    def _on_opacity_fixed_changed(self, value: float) -> None:
        if self._overlay_mode == OVERLAY_COLOR and self.fixed_layer is not None:
            self.fixed_layer.opacity = value

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
            # Re-apply rotation + translation on top of the new enhanced base
            state = self._current_state()
            self._apply_state(state, push=False)

    def _nudge_translate(self, dx: float, dy: float) -> None:
        state = self._current_state()
        state.tx += dx
        state.ty += dy
        self._apply_state(state, push=True)
        self._update_status()

    def _nudge_rotate(self, delta_deg: float) -> None:
        state = self._current_state()
        state.rotation += delta_deg
        self._apply_state(state, push=True)
        self._update_status()

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
        self.slice_paths = self._discover_aips(aips_dir)
        self._use_precomputed_aips = True
        self.slice_ids = list(self.slice_paths.keys())

        # Discover axis-specific AIPs
        self._discover_axis_aip_dirs(aips_dir.parent)

        # Load transforms alongside the aips if present
        for tfm_candidate in (aips_dir.parent / "transforms", aips_dir.parent.parent / "transforms"):
            if tfm_candidate.exists():
                self.transforms_dir = tfm_candidate
                self.existing_transforms = discover_transforms(tfm_candidate)
                break

        self._rebuild_pairs()

    def _on_host_changed(self, text: str) -> None:
        """Update server config host when the user edits the host field."""
        if self.server_config is not None:
            self.server_config.host = text.strip()
