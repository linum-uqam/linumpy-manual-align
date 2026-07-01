# Mixin Responsibility Matrix

Inventory of the 13 behavioral mixins composing `ManualAlignWidget` in `ui/widget.py`. MRO order is authoritative — reordering bases breaks cooperative `super()` chains.

**Orchestrator:** `ManualAlignWidget` in `ui/widget.py` is not a matrix row; it combines mixins and runs `build_manual_align_ui()`.

**14th base:** `QWidget` is the final MRO entry but is not a behavioral mixin.

## Behavioral mixins (MRO order)

| Mixin | File | MRO # | Responsibility | Key methods | Upstream imports | Test coverage pointer |
|-------|------|------:|----------------|-------------|------------------|----------------------|
| PairNavigationMixin | `widget_mixins.py` | 1 | Pair list navigation, prev/next | `_on_prev_pair`, `_on_next_pair`, `_current_pair_index` | (minimal — protocol host) | `tests/test_widget.py` (`PairNavigationMixin` stacks) |
| UiHelpersMixin | `widget_ui.py` | 2 | Event suppression, CS slider visibility | `_suppress_events`, `_set_cs_sliders_visible` | Qt only | indirect / minimal direct |
| PairLoadingMixin | `widget_pair_loading.py` | 3 | AIP discovery and loading pairs into napari | `_load_pair`, `_discover_and_load` | `io.transform_io`, `io.image_utils`, `contracts` | `tests/test_widget.py` (Server+PairLoading stack) |
| OverlayStateMixin | `widget_overlay.py` | 4 | Overlay compositing, apply/restore state | `_apply_state`, `_current_state`, `_update_overlay` | `io.image_utils`, `state` | indirect via widget integration |
| ProjectionEventMixin | `widget_projection.py` | 5 | Pair combo, spinboxes, projection mode, Z-offset | `_on_pair_changed`, `_on_z_offset_changed` | `io.transform_io` | indirect |
| CrossSectionMixin | `widget_cross_section.py` | 6 | Remote XZ/YZ cross-sections and slider logic | `_init_cross_section`, `_on_cs_slider_moved` | `remote.cross_section`, `settings` | `tests/test_cross_section.py` (manager); mixin indirect |
| UndoSaveMixin | `widget_undo_save.py` | 7 | Undo/redo, automated load, reset, save | `_save_transform`, `_undo`, `_redo` | `io.transform_io`, `contracts`, `state` | `tests/test_widget.py` (extensive save/validation) |
| SessionMixin | `widget_session.py` | 8 | Session resume, upload readiness, metadata | `_check_upload_readiness`, `_resume_session` | `contracts`, `remote` | `tests/test_widget.py` (session/resume/upload) |
| ServerMixin | `widget_server.py` | 9 | SCP download/upload, server config, host sync | `_download_package`, `_upload_results`, `_parse_config` | `remote`, `io`, `contracts` | `tests/test_widget.py` (SCP/download/upload) |
| StatusMixin | `widget_status.py` | 10 | Status label and saved flash | `_set_status`, `_flash_saved` | Qt only | `tests/test_widget.py` (with SessionMixin) |
| SettingsUiMixin | `widget_settings_ui.py` | 11 | Settings dialog entry, hints, CS slider steps | `_open_settings`, `_apply_settings` | `settings`, `ui.settings_dialog` | `tests/test_settings.py` |
| CloseGuardMixin | `widget_close_guard.py` | 12 | Unsaved-changes guard on main window close | `_confirm_close`, `_has_unsaved_changes` | Qt only | `tests/test_widget.py` |
| InteractionMixin | `widget_interaction.py` | 13 | Keybindings, nudges, display toggles, closeEvent | `keyPressEvent`, `_nudge`, `closeEvent` | Qt, `settings` | indirect (keybindings) |

### Priority refactor targets (D-15)

| Mixin | File | Why |
|-------|------|-----|
| ServerMixin | `widget_server.py` | Largest server path; metadata gap on download (contracts completion target) |
| PairLoadingMixin | `widget_pair_loading.py` | Shared package-ingest precursor; overlaps with server load path |

See [REFACTOR-SEQUENCE.md](./REFACTOR-SEQUENCE.md) for the ordered remediation plan.

## Related UI modules (non-mixin)

| Module | Role |
|--------|------|
| `widget_build.py` | Layout + signal wiring entry (`build_manual_align_ui`) |
| `ui_builder.py` | Qt widget factories (returns named-widget namespaces) |
| `settings_dialog.py` | Modeless settings UI (`SettingsDialog`) |
| `napari_menus.py` | Napari menu registration |
| `widget_typing.py` | Re-exports `ManualAlignWidget` for mixin `self:` annotations only |

## MRO source of truth

```python
class ManualAlignWidget(
    PairNavigationMixin,
    UiHelpersMixin,
    PairLoadingMixin,
    OverlayStateMixin,
    ProjectionEventMixin,
    CrossSectionMixin,
    UndoSaveMixin,
    SessionMixin,
    ServerMixin,
    StatusMixin,
    SettingsUiMixin,
    CloseGuardMixin,
    InteractionMixin,
    QWidget,
):
```

File: `src/linumpy_manual_align/ui/widget.py`

## Partial mixin stack test pattern

```python
class _SessionWidget(SessionMixin, StatusMixin, UndoSaveMixin, PairNavigationMixin):
    def __init__(self):
        self.pairs = [(0, 1)]
        # ... minimal attrs for mixin under test
```

See `tests/test_widget.py` for existing partial-stack patterns.
