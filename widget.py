"""Napari widget for interactive manual slice alignment.

Displays consecutive common-space slices as red/green AIP overlays.
The user adjusts translation and rotation of the moving slice until
it aligns with the fixed slice (yellow = aligned).

Saves corrected transforms as SimpleITK .tfm files compatible with
the linumpy stacking pipeline.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import napari
import numpy as np
from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSlider,
    QVBoxLayout,
    QWidget,
)
from scipy.ndimage import rotate as ndimage_rotate
from transform_io import (
    discover_slices,
    discover_transforms,
    load_pairwise_metrics,
    load_transform,
    save_transform,
)

logger = logging.getLogger(__name__)

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

    def push(self, state: AlignmentState) -> None:
        # Discard redo history when a new state is pushed
        self._history = self._history[: self._index + 1]
        self._history.append(AlignmentState(state.tx, state.ty, state.rotation))
        self._index = len(self._history) - 1

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
# Widget
# ---------------------------------------------------------------------------


class ManualAlignWidget(QWidget):
    """Napari dock widget for manual slice alignment."""

    def __init__(
        self,
        viewer: napari.Viewer,
        input_dir: Path,
        transforms_dir: Path | None,
        output_dir: Path,
        level: int = 1,
        filter_slices: list[int] | None = None,
    ):
        super().__init__()
        self.viewer = viewer
        self.input_dir = Path(input_dir)
        self.transforms_dir = Path(transforms_dir) if transforms_dir else None
        self.output_dir = Path(output_dir)
        self.level = level

        # Discover slices and transforms
        self.slice_paths = discover_slices(self.input_dir)
        self.slice_ids = list(self.slice_paths.keys())
        self.existing_transforms = discover_transforms(self.transforms_dir) if self.transforms_dir else {}

        # Build slice pair list: [(fixed_id, moving_id), ...]
        self.pairs = []
        for i in range(len(self.slice_ids) - 1):
            fid, mid = self.slice_ids[i], self.slice_ids[i + 1]
            if filter_slices is None or mid in filter_slices:
                self.pairs.append((fid, mid))

        if not self.pairs:
            raise ValueError("No slice pairs found to align.")

        # Per-pair state
        self.undo_stacks: dict[int, UndoStack] = {}  # keyed by moving_id
        self.saved_pairs: set[int] = set()  # moving_ids that have been saved
        self.pair_centers: dict[int, tuple[float, float]] = {}  # (cx, cy) per moving_id

        # Layers (set during pair loading)
        self.fixed_layer: napari.layers.Image | None = None
        self.moving_layer: napari.layers.Image | None = None
        self._original_moving_aip: np.ndarray | None = None  # before rotation
        self._suppress_translate_event = False
        self._suppress_spinbox_event = False

        # Current pair index
        self.current_pair_idx = 0

        self._build_ui()
        self._install_keybindings()
        self._load_pair(0)

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

        self.btn_undo = QPushButton("Undo (⌘Z)")
        self.btn_undo.clicked.connect(self._undo)
        action_layout.addWidget(self.btn_undo)

        self.btn_redo = QPushButton("Redo (⌘⇧Z)")
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

        # -- Status display --
        self.status_label = QLabel("")
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)

        layout.addStretch()

    # ----- Pair loading -----

    def _load_pair(self, idx: int) -> None:
        """Load a slice pair and display as red/green AIP overlay."""
        from linumpy.io.zarr import read_omezarr

        self.current_pair_idx = idx
        fid, mid = self.pairs[idx]

        self.viewer.status = f"Loading z{fid:02d} / z{mid:02d}..."

        # Read slices at working resolution
        fixed_vol, fixed_scale = read_omezarr(str(self.slice_paths[fid]), level=self.level)
        moving_vol, moving_scale = read_omezarr(str(self.slice_paths[mid]), level=self.level)

        # Compute AIPs
        fixed_aip = np.asarray(fixed_vol).mean(axis=0).astype(np.float32)
        moving_aip = np.asarray(moving_vol).mean(axis=0).astype(np.float32)

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

        # Add layers: fixed (green) then moving (red) on top
        self.fixed_layer = self.viewer.add_image(
            fixed_aip,
            name=f"Fixed z{fid:02d}",
            colormap="green",
            blending="additive",
            contrast_limits=clim,
            scale=fixed_scale[1:],  # YX only
        )
        self.moving_layer = self.viewer.add_image(
            moving_aip,
            name=f"Moving z{mid:02d}",
            colormap="red",
            blending="additive",
            contrast_limits=clim,
            scale=moving_scale[1:],
        )

        # Connect layer translate events for bidirectional sync
        self.moving_layer.events.translate.connect(self._on_layer_translated)

        # Initialize or restore undo stack
        if mid not in self.undo_stacks:
            self.undo_stacks[mid] = UndoStack()

        state = self.undo_stacks[mid].current
        self._apply_state(state, push=False)

        # Update UI
        self._suppress_spinbox_event = True
        self.pair_combo.setCurrentIndex(idx)
        self._suppress_spinbox_event = False
        self._update_status()

        self.viewer.reset_view()
        self.viewer.status = f"Pair z{fid:02d} → z{mid:02d} loaded"

    def _normalize(self, img: np.ndarray) -> np.ndarray:
        p_low = np.percentile(img[img > 0], 0.1) if np.any(img > 0) else 0
        p_high = np.percentile(img, 99.9)
        if p_high <= p_low:
            return np.zeros_like(img)
        return np.clip((img - p_low) / (p_high - p_low), 0, 1).astype(np.float32)

    # ----- State application -----

    def _apply_state(self, state: AlignmentState, push: bool = True) -> None:
        """Apply an alignment state to the moving layer."""
        if self.moving_layer is None or self._original_moving_aip is None:
            return

        mid = self.pairs[self.current_pair_idx][1]

        if push:
            self.undo_stacks[mid].push(state)

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
        mid = self.pairs[self.current_pair_idx][1]
        stack = self.undo_stacks.get(mid)
        if stack:
            state = stack.undo()
            if state:
                self._apply_state(state, push=False)
                self._update_status()

    def _redo(self) -> None:
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
        mid = self.pairs[self.current_pair_idx][1]
        if mid not in self.existing_transforms:
            self.viewer.status = f"No automated transform for z{mid:02d}"
            return

        tfm_dir = self.existing_transforms[mid]
        tfm_files = list(tfm_dir.glob("*.tfm"))
        if not tfm_files:
            self.viewer.status = f"No .tfm file in {tfm_dir}"
            return

        tx, ty, rot = load_transform(tfm_files[0])
        # Scale from full resolution to working resolution
        scale = 2**self.level
        state = AlignmentState(tx=tx / scale, ty=ty / scale, rotation=rot)
        self._apply_state(state, push=True)
        self.viewer.status = f"Loaded automated transform for z{mid:02d}: tx={tx:.1f} ty={ty:.1f} rot={rot:.2f}°"
        self._update_status()

    def _reset_transform(self) -> None:
        state = AlignmentState()
        self._apply_state(state, push=True)
        self._update_status()

    # ----- Save -----

    def _save_current(self) -> None:
        """Save the current transform for the current pair."""
        _fid, mid = self.pairs[self.current_pair_idx]
        state = self._current_state()

        cx, cy = self.pair_centers.get(mid, (0.0, 0.0))

        out_dir = self.output_dir / f"slice_z{mid:02d}"
        save_transform(out_dir, state.tx, state.ty, state.rotation, center=(cx, cy), level=self.level)
        self.saved_pairs.add(mid)
        self.viewer.status = f"Saved transform for z{mid:02d} → {out_dir}"
        self._update_status()

    def _save_all_and_exit(self) -> None:
        """Save all modified (non-zero) pairs and close."""
        count = 0
        for _fid, mid in self.pairs:
            stack = self.undo_stacks.get(mid)
            if stack is None:
                continue
            state = stack.current
            if state.tx == 0 and state.ty == 0 and state.rotation == 0:
                continue  # Skip untouched pairs

            cx, cy = self.pair_centers.get(mid, (0.0, 0.0))

            out_dir = self.output_dir / f"slice_z{mid:02d}"
            save_transform(out_dir, state.tx, state.ty, state.rotation, center=(cx, cy), level=self.level)
            self.saved_pairs.add(mid)
            count += 1

        self.viewer.status = f"Saved {count} transforms to {self.output_dir}"
        logger.info(f"Saved {count} manual transforms to {self.output_dir}")
        self.viewer.close()

    # ----- Status display -----

    def _update_status(self) -> None:
        fid, mid = self.pairs[self.current_pair_idx]
        state = self._current_state()
        scale = 2**self.level

        lines = [f"<b>Pair {self.current_pair_idx + 1}/{len(self.pairs)}: z{fid:02d} → z{mid:02d}</b>"]
        lines.append(f"Working res (level {self.level}): tx={state.tx:.1f}  ty={state.ty:.1f}  rot={state.rotation:.2f}°")
        lines.append(f"Full res (level 0): tx={state.tx * scale:.1f}  ty={state.ty * scale:.1f}  rot={state.rotation:.2f}°")

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

        # Undo/redo (Ctrl+Z / Ctrl+Shift+Z)
        @viewer.bind_key("Control-z")
        def _undo(viewer):
            self._undo()

        @viewer.bind_key("Control-Shift-z")
        def _redo(viewer):
            self._redo()

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
