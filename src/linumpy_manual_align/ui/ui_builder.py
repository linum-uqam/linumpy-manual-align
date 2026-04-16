"""Qt widget construction helpers for the manual alignment dock widget.

Each ``build_*`` function creates one logical UI section, wires signals, and
returns the top-level layout/widget together with a :class:`types.SimpleNamespace`
holding every named child widget so the caller can assign them as instance
attributes.

All builder functions are pure Qt constructors with no napari or I/O
dependencies, which makes them easy to instantiate in tests without a display.
"""

from __future__ import annotations

import sys
import types
from collections.abc import Callable

from qtpy.QtCore import Qt
from qtpy.QtGui import QIcon
from qtpy.QtWidgets import (
    QButtonGroup,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSlider,
    QSpinBox,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from linumpy_manual_align.settings_runtime import (
    apply_cross_section_slider_steps,
    default_host_display,
    spin_step_rot,
    spin_step_tile,
    spin_step_tx_ty,
    xy_page_keyboard_hint_html,
)

_IS_MACOS = sys.platform == "darwin"
UNDO_LABEL = "Undo (⌘Z)" if _IS_MACOS else "Undo (Ctrl+Z)"
REDO_LABEL = "Redo (⌘⇧Z)" if _IS_MACOS else "Redo (Ctrl+Shift+Z)"


def make_form(spacing: int = 4) -> QFormLayout:
    """Return a consistently styled :class:`QFormLayout`."""
    f = QFormLayout()
    f.setLabelAlignment(Qt.AlignRight | Qt.AlignVCenter)
    f.setFieldGrowthPolicy(QFormLayout.ExpandingFieldsGrow)
    f.setSpacing(spacing)
    return f


def build_scroll_content() -> tuple[QScrollArea, QVBoxLayout]:
    """Build the outer scroll area and return ``(scroll, content_layout)``."""
    scroll = QScrollArea()
    scroll.setWidgetResizable(True)
    scroll.setFrameShape(QScrollArea.NoFrame)

    content = QWidget()
    layout = QVBoxLayout()
    layout.setContentsMargins(6, 6, 6, 6)
    layout.setSpacing(6)
    content.setLayout(layout)
    scroll.setWidget(content)
    return scroll, layout


def build_navigation_row(
    pairs: list[tuple[int, int]],
    pair_labels: list[str],
    on_prev: Callable,
    on_next: Callable,
    on_combo_changed: Callable,
) -> tuple[QHBoxLayout, types.SimpleNamespace]:
    """Build the slice-pair navigation row.

    Returns ``(layout, widgets)`` where *widgets* has attributes
    ``btn_prev``, ``btn_next``, ``pair_combo``, ``btn_settings``.
    """
    row = QHBoxLayout()
    row.setSpacing(4)

    btn_prev = QPushButton("◀ Prev (P)")
    btn_prev.clicked.connect(on_prev)
    row.addWidget(btn_prev)

    pair_combo = QComboBox()
    for label in pair_labels:
        pair_combo.addItem(label)
    pair_combo.currentIndexChanged.connect(on_combo_changed)
    row.addWidget(pair_combo, stretch=1)

    btn_next = QPushButton("Next (N) ▶")
    btn_next.clicked.connect(on_next)
    row.addWidget(btn_next)

    btn_settings = QToolButton()
    btn_settings.setToolTip("Settings…")
    btn_settings.setFixedSize(22, 22)
    # Prefer the freedesktop "preferences" icon (Linux); macOS/Windows often have no theme icon.
    _pref_icon = QIcon.fromTheme("preferences-system")
    if _pref_icon.isNull():
        _pref_icon = QIcon.fromTheme("preferences-desktop")
    if _pref_icon.isNull():
        btn_settings.setText("\u2699")  # gear
        btn_settings.setToolButtonStyle(Qt.ToolButtonTextOnly)
    else:
        btn_settings.setIcon(_pref_icon)
    row.addWidget(btn_settings)

    return row, types.SimpleNamespace(
        btn_prev=btn_prev,
        btn_next=btn_next,
        pair_combo=pair_combo,
        btn_settings=btn_settings,
    )


def build_mode_row(
    has_axis_aips: bool,
    on_xy_toggled: Callable,
    on_z_toggled: Callable,
) -> tuple[QHBoxLayout, types.SimpleNamespace]:
    """Build the XY / Z Alignment mode toggle row.

    Returns ``(layout, widgets)`` where *widgets* has ``btn_mode_xy``, ``btn_mode_z``.
    """
    row = QHBoxLayout()

    btn_xy = QPushButton("XY Alignment")
    btn_xy.setCheckable(True)
    btn_xy.setChecked(True)
    btn_xy.setToolTip("Lateral alignment: adjust TX, TY, and Rotation  [M to toggle]")
    btn_xy.toggled.connect(on_xy_toggled)
    row.addWidget(btn_xy)

    btn_z = QPushButton("Z Alignment")
    btn_z.setCheckable(True)
    btn_z.setChecked(False)
    btn_z.setToolTip("Depth alignment: adjust Z-overlap offsets, view XZ/YZ cross-sections  [M to toggle]")
    btn_z.toggled.connect(on_z_toggled)
    if not has_axis_aips:
        btn_z.setEnabled(False)
        btn_z.setToolTip("XZ/YZ projections not available — regenerate the data package to enable")
    row.addWidget(btn_z)

    return row, types.SimpleNamespace(btn_mode_xy=btn_xy, btn_mode_z=btn_z)


def build_xy_page(
    on_spinbox_changed: Callable,
    on_rotation_changed: Callable,
    on_rotation_slider_changed: Callable,
    on_load_auto: Callable,
    on_reset: Callable,
    on_undo: Callable,
    on_redo: Callable,
) -> tuple[QWidget, types.SimpleNamespace]:
    """Build the XY Alignment mode page (page 0 of the stacked widget).

    Returns ``(page_widget, widgets)`` where *widgets* has ``spin_tx``,
    ``spin_ty``, ``spin_rot``, ``rot_slider``, ``btn_load_auto``,
    ``btn_reset``, ``btn_undo``, ``btn_redo``.
    """
    page = QWidget()
    layout = QVBoxLayout()
    layout.setContentsMargins(0, 4, 0, 0)
    layout.setSpacing(4)
    page.setLayout(layout)

    form = make_form()

    spin_tx = QDoubleSpinBox()
    spin_tx.setRange(-2000, 2000)
    spin_tx.setDecimals(1)
    spin_tx.setSingleStep(spin_step_tx_ty())
    spin_tx.setSuffix(" px")
    spin_tx.valueChanged.connect(on_spinbox_changed)
    form.addRow("TX:", spin_tx)

    spin_ty = QDoubleSpinBox()
    spin_ty.setRange(-2000, 2000)
    spin_ty.setDecimals(1)
    spin_ty.setSingleStep(spin_step_tx_ty())
    spin_ty.setSuffix(" px")
    spin_ty.valueChanged.connect(on_spinbox_changed)
    form.addRow("TY:", spin_ty)

    spin_rot = QDoubleSpinBox()
    spin_rot.setRange(-180, 180)
    spin_rot.setDecimals(2)
    spin_rot.setSingleStep(spin_step_rot())
    spin_rot.setSuffix("°")
    spin_rot.valueChanged.connect(on_rotation_changed)
    form.addRow("Rotation:", spin_rot)

    rot_slider = QSlider(Qt.Horizontal)
    rot_slider.setRange(-1800, 1800)
    rot_slider.setValue(0)
    rot_slider.valueChanged.connect(on_rotation_slider_changed)
    form.addRow("", rot_slider)

    layout.addLayout(form)

    hint = QLabel(xy_page_keyboard_hint_html())
    hint.setWordWrap(True)
    layout.addWidget(hint)

    row1 = QHBoxLayout()
    btn_load_auto = QPushButton("Load Automated")
    btn_load_auto.clicked.connect(on_load_auto)
    row1.addWidget(btn_load_auto)
    btn_reset = QPushButton("Reset")
    btn_reset.clicked.connect(on_reset)
    row1.addWidget(btn_reset)
    layout.addLayout(row1)

    row2 = QHBoxLayout()
    btn_undo = QPushButton(UNDO_LABEL)
    btn_undo.clicked.connect(on_undo)
    row2.addWidget(btn_undo)
    btn_redo = QPushButton(REDO_LABEL)
    btn_redo.clicked.connect(on_redo)
    row2.addWidget(btn_redo)
    layout.addLayout(row2)

    return page, types.SimpleNamespace(
        spin_tx=spin_tx,
        spin_ty=spin_ty,
        spin_rot=spin_rot,
        rot_slider=rot_slider,
        btn_load_auto=btn_load_auto,
        btn_reset=btn_reset,
        btn_undo=btn_undo,
        btn_redo=btn_redo,
    )


def build_z_page(
    parent_widget: QWidget,
    on_proj_changed: Callable,
    on_fixed_z_changed: Callable,
    on_moving_z_changed: Callable,
) -> tuple[QWidget, types.SimpleNamespace]:
    """Build the Z Alignment mode page (page 1 of the stacked widget).

    Returns ``(page_widget, widgets)`` where *widgets* has
    ``btn_proj_xz``, ``btn_proj_yz``, ``proj_btn_group``,
    ``spin_fixed_z``, ``spin_moving_z``, ``z_relative_label``,
    ``slider_fixed_y``, ``lbl_fixed_y``, ``fixed_y_form_row_label``,
    ``slider_cs_y``, ``lbl_cs_y``, ``cs_y_form_row_label``,
    ``slider_fixed_x``, ``lbl_fixed_x``, ``fixed_x_form_row_label``,
    ``slider_cs_x``, ``lbl_cs_x``, ``cs_x_form_row_label``,
    ``cs_loading_label``.
    """
    page = QWidget()
    layout = QVBoxLayout()
    layout.setContentsMargins(0, 4, 0, 0)
    layout.setSpacing(4)
    page.setLayout(layout)

    form = make_form()

    btn_proj_xz = QPushButton("XZ")
    btn_proj_xz.setCheckable(True)
    btn_proj_xz.setChecked(True)
    btn_proj_xz.setToolTip("Front cross-section (Z-X plane)  [V to toggle]")

    btn_proj_yz = QPushButton("YZ")
    btn_proj_yz.setCheckable(True)
    btn_proj_yz.setToolTip("Side cross-section (Z-Y plane)  [V to toggle]")

    proj_btn_group = QButtonGroup(parent_widget)
    proj_btn_group.setExclusive(True)
    proj_btn_group.addButton(btn_proj_xz, 0)
    proj_btn_group.addButton(btn_proj_yz, 1)
    proj_btn_group.idClicked.connect(on_proj_changed)

    proj_row = QHBoxLayout()
    proj_row.setSpacing(4)
    proj_row.addWidget(btn_proj_xz)
    proj_row.addWidget(btn_proj_yz)
    proj_row.addStretch()
    form.addRow("View:", proj_row)

    spin_fixed_z = QSpinBox()
    spin_fixed_z.setRange(0, 200)
    spin_fixed_z.setSuffix(" voxels")
    spin_fixed_z.setToolTip("Z-index in the fixed (bottom/green) slice where overlap begins")
    spin_fixed_z.valueChanged.connect(on_fixed_z_changed)
    form.addRow("Fixed Z start:", spin_fixed_z)

    spin_moving_z = QSpinBox()
    spin_moving_z.setRange(0, 200)
    spin_moving_z.setSuffix(" voxels")
    spin_moving_z.setToolTip("Z-index in the moving (top/red) slice where overlap begins")
    spin_moving_z.valueChanged.connect(on_moving_z_changed)
    form.addRow("Moving Z start:", spin_moving_z)

    z_relative_label = QLabel("Relative shift: 0 voxels")
    form.addRow("", z_relative_label)

    # Interactive cross-section sliders (hidden until remote metadata is loaded)

    # Fixed reference slider — read-only, shows where the static NPZ cross-section was taken
    slider_fixed_y = QSlider(Qt.Horizontal)
    slider_fixed_y.setMinimum(0)
    slider_fixed_y.setMaximum(0)
    slider_fixed_y.setValue(0)
    slider_fixed_y.setEnabled(False)
    slider_fixed_y.setToolTip("Fixed slice Y position — where the reference cross-section was extracted")
    lbl_fixed_y = QLabel("—")
    lbl_fixed_y.setFixedWidth(40)
    fixed_y_row = QHBoxLayout()
    fixed_y_row.setSpacing(4)
    fixed_y_row.addWidget(slider_fixed_y)
    fixed_y_row.addWidget(lbl_fixed_y)
    fixed_y_form_row_label = QLabel("Fixed Y:")
    form.addRow(fixed_y_form_row_label, fixed_y_row)

    slider_cs_y = QSlider(Qt.Horizontal)
    slider_cs_y.setMinimum(0)
    slider_cs_y.setMaximum(0)
    slider_cs_y.setValue(0)
    slider_cs_y.setEnabled(False)
    lbl_cs_y = QLabel("0")
    lbl_cs_y.setFixedWidth(40)
    cs_y_row = QHBoxLayout()
    cs_y_row.setSpacing(4)
    cs_y_row.addWidget(slider_cs_y)
    cs_y_row.addWidget(lbl_cs_y)
    cs_y_form_row_label = QLabel("Moving Y:")
    form.addRow(cs_y_form_row_label, cs_y_row)

    # Fixed reference slider — X axis
    slider_fixed_x = QSlider(Qt.Horizontal)
    slider_fixed_x.setMinimum(0)
    slider_fixed_x.setMaximum(0)
    slider_fixed_x.setValue(0)
    slider_fixed_x.setEnabled(False)
    slider_fixed_x.setToolTip("Fixed slice X position — where the reference cross-section was extracted")
    lbl_fixed_x = QLabel("—")
    lbl_fixed_x.setFixedWidth(40)
    fixed_x_row = QHBoxLayout()
    fixed_x_row.setSpacing(4)
    fixed_x_row.addWidget(slider_fixed_x)
    fixed_x_row.addWidget(lbl_fixed_x)
    fixed_x_form_row_label = QLabel("Fixed X:")
    form.addRow(fixed_x_form_row_label, fixed_x_row)

    slider_cs_x = QSlider(Qt.Horizontal)
    slider_cs_x.setMinimum(0)
    slider_cs_x.setMaximum(0)
    slider_cs_x.setValue(0)
    slider_cs_x.setEnabled(False)
    lbl_cs_x = QLabel("0")
    lbl_cs_x.setFixedWidth(40)
    cs_x_row = QHBoxLayout()
    cs_x_row.setSpacing(4)
    cs_x_row.addWidget(slider_cs_x)
    cs_x_row.addWidget(lbl_cs_x)
    cs_x_form_row_label = QLabel("Moving X:")
    form.addRow(cs_x_form_row_label, cs_x_row)

    apply_cross_section_slider_steps(slider_cs_y, slider_cs_x)

    cs_loading_label = QLabel("")
    cs_loading_label.setStyleSheet("color: grey; font-style: italic;")
    form.addRow("", cs_loading_label)

    layout.addLayout(form)

    hint = QLabel("<i style='color: grey;'>Align tissue boundaries visible in the cross-section overlay.</i>")
    hint.setWordWrap(True)
    layout.addWidget(hint)

    return page, types.SimpleNamespace(
        btn_proj_xz=btn_proj_xz,
        btn_proj_yz=btn_proj_yz,
        proj_btn_group=proj_btn_group,
        spin_fixed_z=spin_fixed_z,
        spin_moving_z=spin_moving_z,
        z_relative_label=z_relative_label,
        slider_fixed_y=slider_fixed_y,
        lbl_fixed_y=lbl_fixed_y,
        fixed_y_form_row_label=fixed_y_form_row_label,
        slider_cs_y=slider_cs_y,
        lbl_cs_y=lbl_cs_y,
        cs_y_form_row_label=cs_y_form_row_label,
        slider_fixed_x=slider_fixed_x,
        lbl_fixed_x=lbl_fixed_x,
        fixed_x_form_row_label=fixed_x_form_row_label,
        slider_cs_x=slider_cs_x,
        lbl_cs_x=lbl_cs_x,
        cs_x_form_row_label=cs_x_form_row_label,
        cs_loading_label=cs_loading_label,
    )


def build_display_group(
    on_overlay_changed: Callable,
    on_enhance_changed: Callable,
    on_tile_size_changed: Callable,
) -> tuple[QGroupBox, types.SimpleNamespace]:
    """Build the Display group box (overlay mode, enhancement, tile size).

    Returns ``(group_widget, widgets)`` where *widgets* has
    ``combo_overlay``, ``combo_enhance``, ``spin_tile``, ``tile_row_label``.
    """
    group = QGroupBox("Display")
    form = make_form()
    group.setLayout(form)

    combo_overlay = QComboBox()
    combo_overlay.addItems(["Color (R/G)", "Difference", "Checkerboard"])
    combo_overlay.setToolTip(
        "Color: additive red/green overlay\n"
        "Difference: |fixed - moving| grayscale (misalignment = bright)\n"
        "Checkerboard: alternating tiles of fixed/moving"
    )
    combo_overlay.currentIndexChanged.connect(on_overlay_changed)
    form.addRow("Overlay:", combo_overlay)

    combo_enhance = QComboBox()
    combo_enhance.addItems(["None", "Edges (Sobel)", "CLAHE", "Sharpen"])
    combo_enhance.setToolTip(
        "None: display normalized AIP as-is\n"
        "Edges: Sobel gradient magnitude - highlights tissue boundaries\n"
        "  Best for oblique/angled cuts where edges are the key landmark\n"
        "CLAHE: adaptive histogram equalization - equalises local contrast\n"
        "  Best for sagittal cuts where projection blur hides boundaries\n"
        "Sharpen: unsharp mask - mild crispening for blurry projections"
    )
    combo_enhance.currentIndexChanged.connect(on_enhance_changed)
    form.addRow("Enhance:", combo_enhance)

    tile_row_label = QLabel("Tile size:")
    spin_tile = QSpinBox()
    spin_tile.setRange(2, 512)
    spin_tile.setValue(16)
    spin_tile.setSingleStep(spin_step_tile())
    spin_tile.setToolTip("Checkerboard tile size (pixels at current pyramid level)")
    spin_tile.valueChanged.connect(on_tile_size_changed)
    form.addRow(tile_row_label, spin_tile)
    tile_row_label.setVisible(False)
    spin_tile.setVisible(False)

    return group, types.SimpleNamespace(
        combo_overlay=combo_overlay,
        combo_enhance=combo_enhance,
        spin_tile=spin_tile,
        tile_row_label=tile_row_label,
    )


def build_save_row(on_save: Callable, on_save_all: Callable) -> tuple[QHBoxLayout, types.SimpleNamespace]:
    """Build the Save / Save All row.

    Returns ``(layout, widgets)`` where *widgets* has ``btn_save``, ``btn_save_all``.
    """
    row = QHBoxLayout()
    btn_save = QPushButton("Save Current (S)")
    btn_save.clicked.connect(on_save)
    row.addWidget(btn_save)
    btn_save_all = QPushButton("Save All && Exit")
    btn_save_all.clicked.connect(on_save_all)
    row.addWidget(btn_save_all)
    return row, types.SimpleNamespace(btn_save=btn_save, btn_save_all=btn_save_all)


def build_server_group(
    server_config: object,
    on_browse: Callable,
    on_host_changed: Callable,
    on_download: Callable,
    on_upload: Callable,
) -> tuple[QGroupBox, types.SimpleNamespace]:
    """Build the Server group box (config, host, download/upload buttons).

    Returns ``(group_widget, widgets)`` where *widgets* has
    ``config_path_edit``, ``btn_browse_config``, ``host_edit``,
    ``btn_download``, ``btn_upload``, ``server_progress``,
    ``server_status_label``.
    """
    group = QGroupBox("Server")
    layout = QVBoxLayout()
    layout.setSpacing(4)
    group.setLayout(layout)

    cfg_row = QHBoxLayout()
    cfg_row.addWidget(QLabel("Config:"))
    config_path_edit = QLineEdit()
    config_path_edit.setPlaceholderText("nextflow.config path…")
    config_path_edit.setReadOnly(True)
    cfg_row.addWidget(config_path_edit, stretch=1)
    btn_browse_config = QPushButton("Browse…")
    btn_browse_config.clicked.connect(on_browse)
    cfg_row.addWidget(btn_browse_config)
    layout.addLayout(cfg_row)

    host_row = QHBoxLayout()
    host_row.addWidget(QLabel("Host:"))
    host_edit = QLineEdit(default_host_display())
    host_edit.setPlaceholderText("server hostname or IP")
    host_edit.textChanged.connect(on_host_changed)
    host_row.addWidget(host_edit, stretch=1)
    layout.addLayout(host_row)

    srv_btn_row = QHBoxLayout()
    btn_download = QPushButton("⬇ Download Data")
    btn_download.setToolTip("Download AIPs and transforms from the server.")
    btn_download.clicked.connect(on_download)
    srv_btn_row.addWidget(btn_download)
    btn_upload = QPushButton("⬆ Upload Transforms")
    btn_upload.setToolTip("Upload saved manual transforms to the server.")
    btn_upload.clicked.connect(on_upload)
    srv_btn_row.addWidget(btn_upload)
    layout.addLayout(srv_btn_row)

    server_progress = QProgressBar()
    server_progress.setRange(0, 0)
    server_progress.setTextVisible(False)
    server_progress.setFixedHeight(6)
    server_progress.hide()
    layout.addWidget(server_progress)

    server_status_label = QLabel("")
    server_status_label.setWordWrap(True)
    layout.addWidget(server_status_label)

    # Pre-populate from server_config if available
    if server_config is not None:
        if hasattr(server_config, "config_path") and server_config.config_path:
            config_path_edit.setText(str(server_config.config_path))
        if hasattr(server_config, "host"):
            host_edit.setText(server_config.host)
    else:
        btn_download.setEnabled(False)
        btn_upload.setEnabled(False)
        server_status_label.setText("<i>Browse for a nextflow.config to enable server features</i>")

    return group, types.SimpleNamespace(
        config_path_edit=config_path_edit,
        btn_browse_config=btn_browse_config,
        host_edit=host_edit,
        btn_download=btn_download,
        btn_upload=btn_upload,
        server_progress=server_progress,
        server_status_label=server_status_label,
    )
