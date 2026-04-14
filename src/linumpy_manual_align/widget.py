"""Napari widget for interactive manual slice alignment.

Displays consecutive common-space slices as red/green AIP overlays.
The user adjusts translation and rotation of the moving slice until
it aligns with the fixed slice (yellow = aligned).

Saves corrected transforms as SimpleITK .tfm files compatible with
the linumpy stacking pipeline.
"""

from __future__ import annotations

import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path

import napari
import numpy as np
from qtpy.QtCore import Qt, QThread, Signal
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
    QSlider,
    QVBoxLayout,
    QWidget,
)
from scipy.ndimage import rotate as ndimage_rotate

from linumpy_manual_align.omezarr_io import load_aip_from_ome_zarr
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

_MAX_UNDO_HISTORY = 500

# ---------------------------------------------------------------------------
# Undo / redo
# ---------------------------------------------------------------------------


@dataclass
class AlignmentState:
    """Snapshot of alignment parameters for one slice pair."""

    tx: float = 0.0
    ty: float = 0.0
    rotation: float = 0.0


@dataclass
class UndoStack:
    """Simple linear undo/redo stack per slice pair."""

    _history: list[AlignmentState] = field(default_factory=lambda: [AlignmentState()])
    _index: int = 0

    def __init__(self, initial: AlignmentState | None = None):
        self._history = [initial if initial is not None else AlignmentState()]
        self._index = 0

    def push(self, state: AlignmentState) -> None:
        # Discard redo history when a new state is pushed
        self._history = self._history[: self._index + 1]
        self._history.append(AlignmentState(state.tx, state.ty, state.rotation))
        self._index = len(self._history) - 1
        # Enforce maximum history size
        if len(self._history) > _MAX_UNDO_HISTORY:
            trim = len(self._history) - _MAX_UNDO_HISTORY
            self._history = self._history[trim:]
            self._index -= trim

    def undo(self) -> AlignmentState | None:
        if self._index > 0:
            self._index -= 1
            s = self._history[self._index]
            return AlignmentState(s.tx, s.ty, s.rotation)
        return None

    def redo(self) -> AlignmentState | None:
        if self._index < len(self._history) - 1:
            self._index += 1
            s = self._history[self._index]
            return AlignmentState(s.tx, s.ty, s.rotation)
        return None

    @property
    def current(self) -> AlignmentState:
        return self._history[self._index]


# ---------------------------------------------------------------------------
# Background worker for server operations
# ---------------------------------------------------------------------------


class _ScpWorker(QThread):
    """Background QThread for SCP download/upload operations."""

    finished = Signal(bool, str)

    def __init__(self, func: object, args: tuple) -> None:
        super().__init__()
        self._func = func
        self._args = args

    def run(self) -> None:
        ok, msg = self._func(*self._args)
        self.finished.emit(ok, msg)


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
        server_config: object = None,
    ):
        super().__init__()
        self.viewer = viewer
        self.input_dir = Path(input_dir) if input_dir else None
        self.transforms_dir = Path(transforms_dir) if transforms_dir else None
        self.output_dir = Path(output_dir)
        self.level = level
        self.aips_dir = Path(aips_dir) if aips_dir else None
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
        self._original_moving_aip: np.ndarray | None = None  # before rotation
        self._suppress_translate_event = False
        self._suppress_spinbox_event = False
        self._worker: _ScpWorker | None = None  # prevent GC of background thread
        self._close_confirmed = False

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

    def _build_ui(self) -> None:
        layout = QVBoxLayout()
        self.setLayout(layout)

        # -- Pair selector --
        nav_group = QGroupBox("Slice Pair")
        nav_layout = QHBoxLayout()
        nav_group.setLayout(nav_layout)

        self.btn_prev = QPushButton("◀ Prev (P)")
        self.btn_prev.clicked.connect(self._prev_pair)
        nav_layout.addWidget(self.btn_prev)

        self.pair_combo = QComboBox()
        for fid, mid in self.pairs:
            label = f"z{fid:02d} → z{mid:02d}"
            if mid in self.existing_transforms:
                # Load metrics to show severity
                metrics_path = self.existing_transforms[mid] / "pairwise_registration_metrics.json"
                metrics = load_pairwise_metrics(metrics_path)
                mag = self._get_metric(metrics, "translation_magnitude")
                if mag is not None:
                    label += f"  (mag={mag:.0f}px)"
            self.pair_combo.addItem(label)
        self.pair_combo.currentIndexChanged.connect(self._on_pair_changed)
        nav_layout.addWidget(self.pair_combo, stretch=1)

        self.btn_next = QPushButton("Next (N) ▶")
        self.btn_next.clicked.connect(self._next_pair)
        nav_layout.addWidget(self.btn_next)

        layout.addWidget(nav_group)

        # -- Transform controls --
        ctrl_group = QGroupBox("Transform (working resolution)")
        ctrl_layout = QFormLayout()
        ctrl_group.setLayout(ctrl_layout)

        self.spin_tx = QDoubleSpinBox()
        self.spin_tx.setRange(-2000, 2000)
        self.spin_tx.setDecimals(1)
        self.spin_tx.setSingleStep(1.0)
        self.spin_tx.setSuffix(" px")
        self.spin_tx.valueChanged.connect(self._on_spinbox_changed)
        ctrl_layout.addRow("TX:", self.spin_tx)

        self.spin_ty = QDoubleSpinBox()
        self.spin_ty.setRange(-2000, 2000)
        self.spin_ty.setDecimals(1)
        self.spin_ty.setSingleStep(1.0)
        self.spin_ty.setSuffix(" px")
        self.spin_ty.valueChanged.connect(self._on_spinbox_changed)
        ctrl_layout.addRow("TY:", self.spin_ty)

        self.spin_rot = QDoubleSpinBox()
        self.spin_rot.setRange(-180, 180)
        self.spin_rot.setDecimals(2)
        self.spin_rot.setSingleStep(0.1)
        self.spin_rot.setSuffix("°")
        self.spin_rot.valueChanged.connect(self._on_rotation_changed)
        ctrl_layout.addRow("Rotation:", self.spin_rot)

        # Rotation slider for finer control
        self.rot_slider = QSlider(Qt.Horizontal)
        self.rot_slider.setRange(-1800, 1800)  # -180.0 to 180.0 in 0.1° steps
        self.rot_slider.setValue(0)
        self.rot_slider.valueChanged.connect(self._on_rotation_slider_changed)
        ctrl_layout.addRow("", self.rot_slider)

        layout.addWidget(ctrl_group)

        # -- Display controls --
        display_group = QGroupBox("Display")
        display_layout = QFormLayout()
        display_group.setLayout(display_layout)

        self.spin_gamma = QDoubleSpinBox()
        self.spin_gamma.setRange(0.1, 2.0)
        self.spin_gamma.setDecimals(2)
        self.spin_gamma.setSingleStep(0.05)
        self.spin_gamma.setValue(0.6)
        self.spin_gamma.setToolTip("Gamma correction for both layers (< 1 boosts midtones)")
        self.spin_gamma.valueChanged.connect(self._on_gamma_changed)
        display_layout.addRow("Gamma:", self.spin_gamma)

        self.spin_opacity_moving = QDoubleSpinBox()
        self.spin_opacity_moving.setRange(0.1, 1.0)
        self.spin_opacity_moving.setDecimals(2)
        self.spin_opacity_moving.setSingleStep(0.05)
        self.spin_opacity_moving.setValue(1.0)
        self.spin_opacity_moving.setToolTip("Opacity of the moving (red) layer")
        self.spin_opacity_moving.valueChanged.connect(self._on_opacity_moving_changed)
        display_layout.addRow("Moving opacity:", self.spin_opacity_moving)

        self.spin_opacity_fixed = QDoubleSpinBox()
        self.spin_opacity_fixed.setRange(0.1, 1.0)
        self.spin_opacity_fixed.setDecimals(2)
        self.spin_opacity_fixed.setSingleStep(0.05)
        self.spin_opacity_fixed.setValue(1.0)
        self.spin_opacity_fixed.setToolTip("Opacity of the fixed (green) layer")
        self.spin_opacity_fixed.valueChanged.connect(self._on_opacity_fixed_changed)
        display_layout.addRow("Fixed opacity:", self.spin_opacity_fixed)

        layout.addWidget(display_group)

        # -- Action buttons --
        action_group = QGroupBox("Actions")
        action_layout = QHBoxLayout()
        action_group.setLayout(action_layout)

        self.btn_load_auto = QPushButton("Load Automated")
        self.btn_load_auto.clicked.connect(self._load_automated_transform)
        action_layout.addWidget(self.btn_load_auto)

        self.btn_reset = QPushButton("Reset")
        self.btn_reset.clicked.connect(self._reset_transform)
        action_layout.addWidget(self.btn_reset)

        self.btn_undo = QPushButton(_UNDO_LABEL)
        self.btn_undo.clicked.connect(self._undo)
        action_layout.addWidget(self.btn_undo)

        self.btn_redo = QPushButton(_REDO_LABEL)
        self.btn_redo.clicked.connect(self._redo)
        action_layout.addWidget(self.btn_redo)

        layout.addWidget(action_group)

        # -- Save buttons --
        save_group = QGroupBox("Save")
        save_layout = QHBoxLayout()
        save_group.setLayout(save_layout)

        self.btn_save = QPushButton("Save Current (S)")
        self.btn_save.clicked.connect(self._save_current)
        save_layout.addWidget(self.btn_save)

        self.btn_save_all = QPushButton("Save All Modified && Exit")
        self.btn_save_all.clicked.connect(self._save_all_and_exit)
        save_layout.addWidget(self.btn_save_all)

        layout.addWidget(save_group)

        # -- Server transfer buttons --
        server_group = QGroupBox("Server")
        server_layout = QVBoxLayout()
        server_group.setLayout(server_layout)

        # Config file selector
        config_layout = QHBoxLayout()
        config_layout.addWidget(QLabel("Config:"))
        self.config_path_edit = QLineEdit()
        self.config_path_edit.setPlaceholderText("nextflow.config path...")
        self.config_path_edit.setReadOnly(True)
        config_layout.addWidget(self.config_path_edit, stretch=1)
        self.btn_browse_config = QPushButton("Browse...")
        self.btn_browse_config.clicked.connect(self._browse_server_config)
        config_layout.addWidget(self.btn_browse_config)
        server_layout.addLayout(config_layout)

        # Host field
        host_layout = QHBoxLayout()
        host_layout.addWidget(QLabel("Host:"))
        self.host_edit = QLineEdit("132.207.157.41")
        self.host_edit.setPlaceholderText("server hostname or IP")
        self.host_edit.textChanged.connect(self._on_host_changed)
        host_layout.addWidget(self.host_edit, stretch=1)
        server_layout.addLayout(host_layout)

        server_btn_layout = QHBoxLayout()
        self.btn_download = QPushButton("⬇ Download Data")
        self.btn_download.setToolTip("Download AIPs and transforms from the server.\nRequires a configured server.")
        self.btn_download.clicked.connect(self._download_from_server)
        server_btn_layout.addWidget(self.btn_download)

        self.btn_upload = QPushButton("⬆ Upload Transforms")
        self.btn_upload.setToolTip(
            "Upload saved manual transforms to the server.\nThey will be placed in output/manual_transforms/."
        )
        self.btn_upload.clicked.connect(self._upload_to_server)
        server_btn_layout.addWidget(self.btn_upload)
        server_layout.addLayout(server_btn_layout)

        self.server_progress = QProgressBar()
        self.server_progress.setRange(0, 0)  # indeterminate pulse
        self.server_progress.setTextVisible(False)
        self.server_progress.setFixedHeight(6)
        self.server_progress.hide()
        server_layout.addWidget(self.server_progress)

        self.server_status_label = QLabel("")
        self.server_status_label.setWordWrap(True)
        server_layout.addWidget(self.server_status_label)

        # Initialize server UI state
        if self.server_config is not None:
            if hasattr(self.server_config, "config_path") and self.server_config.config_path:
                self.config_path_edit.setText(str(self.server_config.config_path))
            self.host_edit.setText(self.server_config.host)
        else:
            self.btn_download.setEnabled(False)
            self.btn_upload.setEnabled(False)
            self.server_status_label.setText("<i>Browse for a nextflow.config to enable server features</i>")

        layout.addWidget(server_group)

        # -- Status display --
        self.status_label = QLabel("")
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)

        layout.addStretch()

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

        if self._use_precomputed_aips:
            fixed_aip, fixed_scale_yx = self._load_aip_from_npz(self.slice_paths[fid])
            moving_aip, moving_scale_yx = self._load_aip_from_npz(self.slice_paths[mid])
        else:
            fixed_aip, fixed_scale_yx = self._load_aip_from_zarr(self.slice_paths[fid])
            moving_aip, moving_scale_yx = self._load_aip_from_zarr(self.slice_paths[mid])

        # Normalize to [0, 1] for display
        fixed_aip = self._normalize(fixed_aip)
        moving_aip = self._normalize(moving_aip)

        self._original_moving_aip = moving_aip.copy()

        # Store image center for this pair
        self.pair_centers[mid] = (moving_aip.shape[1] / 2.0, moving_aip.shape[0] / 2.0)  # (cx, cy)

        # Remove existing layers
        while len(self.viewer.layers) > 0:
            self.viewer.layers.pop(0)

        # Compute shared contrast limits from both AIPs
        clim = (0.0, 1.0)
        gamma = self.spin_gamma.value()
        opacity_fixed = self.spin_opacity_fixed.value()
        opacity_moving = self.spin_opacity_moving.value()

        # Add layers: fixed (green) then moving (red) on top
        self.fixed_layer = self.viewer.add_image(
            fixed_aip,
            name=f"Fixed z{fid:02d}",
            colormap="green",
            blending="additive",
            contrast_limits=clim,
            gamma=gamma,
            opacity=opacity_fixed,
            scale=fixed_scale_yx,
        )
        self.moving_layer = self.viewer.add_image(
            moving_aip,
            name=f"Moving z{mid:02d}",
            colormap="red",
            blending="additive",
            contrast_limits=clim,
            gamma=gamma,
            opacity=opacity_moving,
            scale=moving_scale_yx,
        )

        # Connect layer translate events for bidirectional sync
        self.moving_layer.events.translate.connect(self._on_layer_translated)

        # Initialize or restore undo stack
        if mid not in self.undo_stacks:
            # Auto-load automated transform as initial state if available
            initial = self._get_automated_state(mid)
            self.undo_stacks[mid] = UndoStack(initial)

        state = self.undo_stacks[mid].current
        self._apply_state(state, push=False)

        # Update UI
        self._suppress_spinbox_event = True
        self.pair_combo.setCurrentIndex(idx)
        self._suppress_spinbox_event = False
        self._update_status()

        self.viewer.reset_view()
        self.viewer.status = f"Pair z{fid:02d} → z{mid:02d} loaded"

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
        p_low = np.percentile(img[img > 0], 1) if np.any(img > 0) else 0
        p_high = np.percentile(img, 99)
        if p_high <= p_low:
            return np.zeros_like(img)
        return np.clip((img - p_low) / (p_high - p_low), 0, 1).astype(np.float32)

    # ----- State application -----

    def _apply_state(self, state: AlignmentState, push: bool = True) -> None:
        """Apply an alignment state to the moving layer."""
        if self.moving_layer is None or self._original_moving_aip is None or not self.pairs:
            return

        mid = self.pairs[self.current_pair_idx][1]

        if push:
            self.undo_stacks[mid].push(state)
            self.unsaved_changes.add(mid)

        # Apply rotation to image data
        if abs(state.rotation) > 0.01:
            rotated = ndimage_rotate(
                self._original_moving_aip, -state.rotation, reshape=False, order=1, mode="constant", cval=0
            )
        else:
            rotated = self._original_moving_aip.copy()

        self.moving_layer.data = rotated

        # Apply translation via layer translate
        scale = self.moving_layer.scale
        self._suppress_translate_event = True
        self.moving_layer.translate = [state.ty * scale[0], state.tx * scale[1]]
        self._suppress_translate_event = False

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

        # Carry over automated offsets if available
        offsets = (0, 0)
        if mid in self.existing_transforms:
            offsets = load_offsets(self.existing_transforms[mid] / "offsets.txt")

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

            # Carry over automated offsets if available
            offsets = (0, 0)
            if mid in self.existing_transforms:
                offsets = load_offsets(self.existing_transforms[mid] / "offsets.txt")

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

        self._worker = _ScpWorker(download_manual_align_package, (self.server_config, local_dir, self.level))
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

        self._worker = _ScpWorker(upload_manual_transforms, (self.server_config, self.output_dir))
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

        lines = [f"<b>Pair {self.current_pair_idx + 1}/{len(self.pairs)}: z{fid:02d} → z{mid:02d}</b>"]
        lines.append(f"Working res (level {self.level}): tx={state.tx:.1f}  ty={state.ty:.1f}  rot={state.rotation:.2f}°")
        lines.append(f"Full res (level 0): tx={state.tx * scale:.1f}  ty={state.ty * scale:.1f}  rot={state.rotation:.2f}°")
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
        if self.fixed_layer is not None:
            self.fixed_layer.gamma = value
        if self.moving_layer is not None:
            self.moving_layer.gamma = value

    def _on_opacity_moving_changed(self, value: float) -> None:
        if self.moving_layer is not None:
            self.moving_layer.opacity = value

    def _on_opacity_fixed_changed(self, value: float) -> None:
        if self.fixed_layer is not None:
            self.fixed_layer.opacity = value

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
