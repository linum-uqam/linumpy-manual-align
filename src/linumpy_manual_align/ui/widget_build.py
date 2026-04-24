"""Construct the dock widget layout (Qt widgets + signal wiring)."""

from __future__ import annotations

from qtpy.QtCore import QTimer
from qtpy.QtWidgets import QLabel, QStackedWidget, QVBoxLayout

from linumpy_manual_align.ui.ui_builder import (
    build_display_group,
    build_mode_row,
    build_navigation_row,
    build_save_row,
    build_scroll_content,
    build_server_group,
    build_xy_page,
    build_z_page,
)
from linumpy_manual_align.ui.widget_typing import ManualAlignWidget


def build_manual_align_ui(widget: ManualAlignWidget) -> None:
    # ── Outer scroll area ────────────────────────────────────────────────
    outer = QVBoxLayout()
    outer.setContentsMargins(0, 0, 0, 0)
    widget.setLayout(outer)
    scroll, layout = build_scroll_content()
    outer.addWidget(scroll)

    # ── Pair navigation ───────────────────────────────────────────────────
    nav_layout, nav = build_navigation_row(
        pairs=widget.pairs,
        pair_labels=[widget._pair_label(fid, mid) for fid, mid in widget.pairs],
        on_prev=widget._prev_pair,
        on_next=widget._next_pair,
        on_combo_changed=widget._on_pair_changed,
    )
    widget.btn_prev = nav.btn_prev
    widget.btn_next = nav.btn_next
    widget.pair_combo = nav.pair_combo
    nav.btn_settings.clicked.connect(widget._open_settings_dialog)
    layout.addLayout(nav_layout)

    # ── Mode toggle buttons ───────────────────────────────────────────────
    has_axis_aips = bool(widget.slice_paths_xz or widget.slice_paths_yz or widget.pair_paths_xz or widget.pair_paths_yz)
    mode_layout, mode = build_mode_row(
        has_axis_aips=has_axis_aips,
        on_xy_toggled=lambda checked: widget._on_mode_btn_toggled("xy", checked),
        on_z_toggled=lambda checked: widget._on_mode_btn_toggled("z", checked),
    )
    widget._btn_mode_xy = mode.btn_mode_xy
    widget._btn_mode_z = mode.btn_mode_z
    layout.addLayout(mode_layout)

    # ── Stacked content (sizes to active page only) ───────────────────────
    widget._mode_stack = QStackedWidget()
    layout.addWidget(widget._mode_stack)

    # Page 0 - XY Alignment
    xy_page, xy = build_xy_page(
        on_spinbox_changed=widget._on_spinbox_changed,
        on_rotation_changed=widget._on_rotation_changed,
        on_rotation_slider_changed=widget._on_rotation_slider_changed,
        on_load_auto=widget._load_automated_transform,
        on_reset=widget._reset_transform,
        on_undo=widget._undo,
        on_redo=widget._redo,
    )
    widget.spin_tx = xy.spin_tx
    widget.spin_ty = xy.spin_ty
    widget.spin_rot = xy.spin_rot
    widget.rot_slider = xy.rot_slider
    widget.btn_load_auto = xy.btn_load_auto
    widget.btn_reset = xy.btn_reset
    widget.btn_undo = xy.btn_undo
    widget.btn_redo = xy.btn_redo
    widget._mode_stack.addWidget(xy_page)

    # Page 1 - Z Alignment
    z_page, z = build_z_page(
        parent_widget=widget,
        on_proj_changed=widget._on_z_proj_changed,
        on_fixed_z_changed=widget._on_z_offset_changed,
        on_moving_z_changed=widget._on_z_offset_changed,
    )
    widget._btn_proj_xz = z.btn_proj_xz
    widget._btn_proj_yz = z.btn_proj_yz
    widget._proj_btn_group = z.proj_btn_group
    widget.spin_fixed_z = z.spin_fixed_z
    widget.spin_moving_z = z.spin_moving_z
    widget.z_relative_label = z.z_relative_label
    widget.slider_fixed_y = z.slider_fixed_y
    widget._lbl_fixed_y = z.lbl_fixed_y
    widget._fixed_y_form_row_label = z.fixed_y_form_row_label
    widget.slider_cs_y = z.slider_cs_y
    widget._lbl_cs_y = z.lbl_cs_y
    widget._cs_y_form_row_label = z.cs_y_form_row_label
    widget.slider_fixed_x = z.slider_fixed_x
    widget._lbl_fixed_x = z.lbl_fixed_x
    widget._fixed_x_form_row_label = z.fixed_x_form_row_label
    widget.slider_cs_x = z.slider_cs_x
    widget._lbl_cs_x = z.lbl_cs_x
    widget._cs_x_form_row_label = z.cs_x_form_row_label
    widget._cs_loading_label = z.cs_loading_label
    widget._set_cs_sliders_visible(False)
    widget._apply_cs_slider_step()
    widget._mode_stack.addWidget(z_page)

    # ── Display ───────────────────────────────────────────────────────────
    disp_group, disp = build_display_group(
        on_overlay_changed=widget._on_overlay_mode_changed,
        on_enhance_changed=widget._on_enhance_changed,
        on_tile_size_changed=widget._on_tile_size_changed,
    )
    widget.combo_overlay = disp.combo_overlay
    widget.combo_enhance = disp.combo_enhance
    widget.spin_tile = disp.spin_tile
    widget._tile_row_label = disp.tile_row_label
    layout.addWidget(disp_group)

    # ── Save ──────────────────────────────────────────────────────────────
    save_layout, save = build_save_row(
        on_save=widget._save_current,
        on_save_all=widget._save_all_and_exit,
    )
    widget.btn_save = save.btn_save
    widget.btn_save_all = save.btn_save_all
    layout.addLayout(save_layout)

    # ── Server ────────────────────────────────────────────────────────────
    server_group, srv = build_server_group(
        server_config=widget.server_config,
        on_browse=widget._browse_server_config,
        on_host_changed=widget._on_host_changed,
        on_remote_python_changed=widget._persist_remote_python,
        on_download=widget._download_from_server,
        on_upload=widget._upload_to_server,
    )
    widget.config_path_edit = srv.config_path_edit
    widget.btn_browse_config = srv.btn_browse_config
    widget.host_edit = srv.host_edit
    widget.host_edit.editingFinished.connect(widget._persist_server_host)
    widget.remote_python_edit = srv.remote_python_edit
    widget.btn_download = srv.btn_download
    widget.btn_upload = srv.btn_upload
    widget.server_progress = srv.server_progress
    widget.server_status_label = srv.server_status_label
    layout.addWidget(server_group)

    # ── Status ────────────────────────────────────────────────────────────
    widget.status_label = QLabel("")
    widget.status_label.setWordWrap(True)
    layout.addWidget(widget.status_label)

    widget.hints_label = QLabel()
    widget.hints_label.setWordWrap(True)
    widget.hints_label.setContentsMargins(0, 4, 0, 0)
    widget._refresh_shortcut_hints()
    layout.addWidget(widget.hints_label)

    # Timer for the ephemeral "✓ SAVED" flash — clears after 3 s
    widget._saved_flash_mid = None
    widget._saved_flash_timer = QTimer(widget)
    widget._saved_flash_timer.setSingleShot(True)
    widget._saved_flash_timer.timeout.connect(widget._on_saved_flash_timeout)

    # Wire cross-section slider signals (debounced via QTimer)
    widget._cs_debounce_timer = QTimer(widget)
    widget._cs_debounce_timer.setSingleShot(True)
    widget._cs_debounce_timer.timeout.connect(widget._on_cs_slider_settled)

    def _on_y_changed(v: int) -> None:
        widget._lbl_cs_y.setText(str(v))
        widget._cross_section_y = v
        if widget.pairs:
            _fid, _mid = widget.pairs[widget.current_pair_idx]
            widget._cs_positions[_mid] = (widget._cross_section_y, widget._cross_section_x)
        widget._cs_debounce_timer.start(150)

    def _on_x_changed(v: int) -> None:
        widget._lbl_cs_x.setText(str(v))
        widget._cross_section_x = v
        if widget.pairs:
            _fid, _mid = widget.pairs[widget.current_pair_idx]
            widget._cs_positions[_mid] = (widget._cross_section_y, widget._cross_section_x)
        widget._cs_debounce_timer.start(150)

    widget.slider_cs_y.valueChanged.connect(_on_y_changed)
    widget.slider_cs_x.valueChanged.connect(_on_x_changed)
