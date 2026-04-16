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

Implementation is split across ``widget_*.py`` mixins and :mod:`widget_build`.
"""

from __future__ import annotations

import logging
from pathlib import Path

import napari
from qtpy.QtCore import QTimer
from qtpy.QtWidgets import QWidget

from linumpy_manual_align.io.image_utils import (
    ENHANCE_NONE,
    OVERLAY_COLOR,
)
from linumpy_manual_align.io.transform_io import (
    discover_aips,
    discover_pair_aips,
    discover_slices,
    discover_transforms,
)
from linumpy_manual_align.remote import ScpWorker
from linumpy_manual_align.remote.cross_section import CrossSectionManager
from linumpy_manual_align.settings import settings
from linumpy_manual_align.ui.widget_build import build_manual_align_ui
from linumpy_manual_align.ui.widget_close_guard import CloseGuardMixin
from linumpy_manual_align.ui.widget_cross_section import CrossSectionMixin
from linumpy_manual_align.ui.widget_interaction import InteractionMixin
from linumpy_manual_align.ui.widget_mixins import PairNavigationMixin
from linumpy_manual_align.ui.widget_overlay import OverlayStateMixin
from linumpy_manual_align.ui.widget_pair_loading import PairLoadingMixin
from linumpy_manual_align.ui.widget_projection import ProjectionEventMixin
from linumpy_manual_align.ui.widget_server import ServerMixin
from linumpy_manual_align.ui.widget_settings_ui import SettingsUiMixin
from linumpy_manual_align.ui.widget_status import StatusMixin
from linumpy_manual_align.ui.widget_ui import UiHelpersMixin
from linumpy_manual_align.ui.widget_undo_save import UndoSaveMixin

logger = logging.getLogger(__name__)


class ManualAlignWidget(
    PairNavigationMixin,
    UiHelpersMixin,
    PairLoadingMixin,
    OverlayStateMixin,
    ProjectionEventMixin,
    CrossSectionMixin,
    UndoSaveMixin,
    ServerMixin,
    StatusMixin,
    SettingsUiMixin,
    CloseGuardMixin,
    InteractionMixin,
    QWidget,
):
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

        if self.aips_dir is not None:
            self.slice_paths = discover_aips(self.aips_dir)
        elif self.input_dir is not None:
            self.slice_paths = discover_slices(self.input_dir)
        else:
            self.slice_paths = {}

        self.slice_paths_xz = discover_aips(self.aips_xz_dir) if self.aips_xz_dir else {}
        self.slice_paths_yz = discover_aips(self.aips_yz_dir) if self.aips_yz_dir else {}
        self.pair_paths_xz: dict[tuple[int, int], dict[str, Path]] = (
            discover_pair_aips(self.aips_xz_dir) if self.aips_xz_dir else {}
        )
        self.pair_paths_yz: dict[tuple[int, int], dict[str, Path]] = (
            discover_pair_aips(self.aips_yz_dir) if self.aips_yz_dir else {}
        )
        self.pair_paths_xy: dict[tuple[int, int], dict[str, Path]] = discover_pair_aips(self.aips_dir) if self.aips_dir else {}

        self.slice_ids = list(self.slice_paths.keys())
        self.existing_transforms = discover_transforms(self.transforms_dir) if self.transforms_dir else {}
        self._filter_slices = filter_slices

        self.pairs: list[tuple[int, int]] = []
        self._build_pairs()

        if not self.pairs:
            logger.info("Starting in empty state — no slice pairs found. Download data from server.")

        self.undo_stacks = {}
        self.saved_pairs: set[int] = set()
        self.unsaved_changes: set[int] = set()
        self.pair_centers: dict[int, tuple[float, float]] = {}

        self._refresh_saved_pairs()

        self.fixed_layer: napari.layers.Image | None = None
        self.moving_layer: napari.layers.Image | None = None
        self._composite_layer: napari.layers.Image | None = None
        self._raw_fixed_aip = None
        self._raw_moving_aip = None
        self._moving_scale_yx: list[float] = [1.0, 1.0]
        self._original_fixed_aip = None
        self._original_moving_aip = None
        self._suppress_spinbox_event = False
        self._suppress_z_offset_event = False
        self._worker: ScpWorker | None = None
        self._close_confirmed = False

        self._cs_mgr = CrossSectionManager(parent=self)
        self._cs_mgr.reader_ready.connect(self._on_reader_ready)
        self._cs_mgr.reader_failed.connect(self._on_reader_failed)
        self._cs_mgr.cross_section_ready.connect(self._on_cross_section_ready)
        self._cs_mgr.cross_section_failed.connect(self._on_cross_section_failed)
        self._cross_section_y: int = 0
        self._cross_section_x: int = 0
        self._cs_positions: dict[int, tuple[int, int]] = {}
        self._cs_debounce_timer: QTimer | None = None

        self._projection_mode = "xy"
        self._current_offsets: dict[int, tuple[int, int]] = {}
        self._content_bbox_wl: tuple[int, int, int, int] | None = None
        self._fixed_cs_pos: tuple[int, int] = (0, 0)

        self._overlay_mode = OVERLAY_COLOR
        self._enhance_mode = ENHANCE_NONE

        self.current_pair_idx = 0

        self._settings_dialog = None

        self._host_persist_timer = QTimer(self)
        self._host_persist_timer.setSingleShot(True)
        self._host_persist_timer.timeout.connect(self._persist_server_host)

        build_manual_align_ui(self)
        self._sync_server_config_host_from_ui()
        self._install_keybindings()
        self._install_close_guard()
        settings.changed.connect(self._on_settings_changed)
        if self.pairs:
            self._load_pair(0)
        else:
            if self.server_config is not None and not self.pairs:
                existing = self._find_existing_package()
                if existing is not None:
                    self._load_existing_package(existing)
                    self.server_status_label.setText(f"Existing package loaded from {existing.parent}")
            self._update_status()
