# Linumpy Manual Align

[![CI](https://github.com/linum-uqam/linumpy-manual-align/actions/workflows/ci.yml/badge.svg)](https://github.com/linum-uqam/linumpy-manual-align/actions/workflows/ci.yml)

**Command-line application** that launches [napari](https://napari.org) for manually correcting pairwise slice alignment
in the [linumpy](https://github.com/linum-uqam/linumpy) reconstruction pipeline. It is distributed as a normal Python package with a CLI entry point, not as an installable napari-plugin-manager plugin.

## Documentation

- [CLI + Nextflow integration guide](docs/cli-nextflow-guide.md)

## Installation

```bash
# From GitHub:
uv pip install git+https://github.com/linum-uqam/linumpy-manual-align.git

# With OME-Zarr support (ome-zarr extra):
uv pip install "linumpy-manual-align[zarr] @ git+https://github.com/linum-uqam/linumpy-manual-align.git"

# For development:
git clone https://github.com/linum-uqam/linumpy-manual-align.git
cd linumpy-manual-align
uv pip install -e ".[dev]"
```

## Usage

### With server download (recommended)

Launch the tool with no data arguments — open the Server panel, browse to a
local `nextflow.config`, and click **Download** to fetch the data package from
the reconstruction server via SCP:

```bash
linumpy-manual-align
```

Pre-populate the server config path from the CLI:

```bash
linumpy-manual-align --server_config ~/Downloads/sub-22/nextflow.config
```

After aligning, use **Upload** to push manual transforms back to the server.

### With a local data package

```bash
linumpy-manual-align --data_package /path/to/manual_align_package/
```

### Directly from OME-Zarr volumes (requires the `ome-zarr` extra)

```bash
linumpy-manual-align \
    --input_dir /path/to/bring_to_common_space/ \
    --transforms_dir /path/to/register_pairwise/ \
    --output_dir /path/to/manual_transforms/ \
    --level 1
```

## Workflow

1. **XY Alignment** — adjust lateral translation (TX/TY) and rotation for each
   consecutive slice pair using the red/green overlay.
2. **Z Alignment** — inspect and correct the Z-overlap (depth at which the two
   slices meet) using XZ and YZ cross-section views.
3. Save corrected transforms — outputs `.tfm` files compatible with the pipeline.
4. Upload transforms or copy to `output/manual_transforms/` on the server.
5. Re-run the pipeline from the `stack` step with `-resume`.

## Overlay Modes

Three overlay modes are available in the **Display** panel to help judge
lateral alignment:

| Mode | Description | Best for |
|------|-------------|----------|
| **Color (R/G)** | Additive red/green overlay; yellow = aligned | General use |
| **Difference** | \|fixed − moving\| in grayscale; bright = misaligned | Detecting subtle offsets |
| **Checkerboard** | Alternating tiles of fixed/moving | Seeing local discontinuities |

The checkerboard tile size is configurable (2–512 px, default 16 px).

## Enhancement Modes

An **Enhance** selector pre-processes each AIP before display:

| Mode | Description | Best for |
|------|-------------|----------|
| **None** | Normalized AIP as-is | General use |
| **Edges (Sobel)** | Gradient magnitude — tissue boundaries only | Oblique cuts (e.g. 45°) |
| **CLAHE** | Adaptive histogram equalization | Sagittal cuts with projection blur |
| **Sharpen** | Unsharp mask | Blurry projections |

Overlay and enhancement modes are independent and can be combined freely.

## Keyboard Shortcuts

| Key | Action |
|-----|--------|
| Arrow keys | Nudge translation ±1 px |
| Alt + Arrow | Nudge translation ±10 px |
| Ctrl + Arrow | Nudge translation ±50 px |
| `[` / `]` | Rotate ±0.1° |
| Alt + `[` / `]` | Rotate ±1.0° |
| Ctrl + `[` / `]` | Rotate ±5.0° |
| `N` / `P` | Next / previous slice pair |
| `M` | Toggle XY / Z Alignment mode |
| `V` | Toggle XZ / YZ projection (Z mode) |
| `S` | Save current pair |
| Ctrl+Z / Ctrl+Shift+Z | Undo / Redo |

## Settings

A **Settings** dialog lets you adjust behavioural parameters without editing code.

Open it via:
- The small **gear button** (⚙) at the right end of the slice-pair navigation row, or
- **Plugins → Manual Align Settings…** in napari’s menu bar (the built-in Plugins menu, not a separate plugin install).

The dialog is modeless — it stays open while you continue working. Changes take effect immediately after clicking **Apply** or **OK**.

### Configurable parameters

| Tab | Parameter | Default | Description |
|-----|-----------|---------|-------------|
| Shortcuts | Translation fine step | 1 px | Arrow key nudge |
| Shortcuts | Translation coarse step | 10 px | Alt+Arrow nudge |
| Shortcuts | Translation large step | 50 px | Shift+Arrow nudge |
| Shortcuts | Rotation fine step | 0.1° | `[`/`]` key nudge |
| Shortcuts | Rotation coarse step | 1.0° | Alt+`[`/`]` nudge |
| Shortcuts | Rotation large step | 5.0° | Ctrl+`[`/`]` nudge |
| Shortcuts | Cross-section nudge | 10 px | Alt+`,`/`.` , moving Y/X sliders, and prefetch spacing along the axis |
| Cross-section | Prefetch steps | 5 | Positions pre-fetched on each side of current |
| Cross-section | Evict radius multiplier | 15 | Evict radius = multiplier × cross-section nudge (px) |
| Spinboxes | TX / TY step | 1.0 px | Spinbox single-step increment |
| Spinboxes | Rotation step | 0.1° | Spinbox single-step increment |
| Spinboxes | Checkerboard tile step | 4 | Spinbox single-step increment |
| Server | Default host | *(empty)* | Set your SSH host here or in the dock; not stored in git |
| Server | Remote Python path | *(empty)* | Remote interpreter for the zarr reader; not stored in git |

### Where settings are stored

Settings are persisted using Qt's native preferences mechanism (on **your machine only** — nothing under this repository). They survive reinstalls of the package:

| OS | Location |
|----|----------|
| macOS | `~/Library/Preferences/com.linum-uqam.linumpy-manual-align.plist` |
| Linux | `~/.config/linum-uqam/linumpy-manual-align.conf` |
| Windows | Registry: `HKCU\Software\linum-uqam\linumpy-manual-align` |

Clicking **Reset All Defaults** in the dialog removes all stored overrides.

## Architecture

The package uses subpackages under `linumpy_manual_align/`. The napari dock UI is split into mixins under `ui/`; `ManualAlignWidget` in `ui/widget.py` subclasses them in a fixed MRO order and runs `build_manual_align_ui()` from `ui/widget_build.py`. Mixin methods annotate `self` via `ui/widget_typing.py` (one import) so type checkers see the full widget type without repeating `TYPE_CHECKING` blocks.

| Location | Responsibility |
|----------|----------------|
| `ui/widget.py` | `ManualAlignWidget` — thin class combining mixins and `__init__` |
| `ui/widget_typing.py` | Re-exports `ManualAlignWidget` for mixin `self:` annotations only |
| `ui/widget_build.py` | `build_manual_align_ui()` — dock layout and signal wiring |
| `ui/widget_mixins.py` | `PairNavigationMixin`, `_PairNavHost` protocol for pair list / prev-next |
| `ui/widget_pair_loading.py` | AIP discovery and loading pairs into napari |
| `ui/widget_overlay.py` | Overlay compositing, `_apply_state` / `_current_state` |
| `ui/widget_projection.py` | Pair combo, spinboxes, projection mode, Z-offset handlers |
| `ui/widget_cross_section.py` | Remote XZ/YZ cross-sections and slider logic |
| `ui/widget_server.py` | SCP download/upload, server config, host sync |
| `ui/widget_undo_save.py` | Undo/redo, automated load, reset, save |
| `ui/widget_status.py` | Status label and “saved” flash |
| `ui/widget_interaction.py` | Keybindings, nudges, display toggles, `closeEvent` |
| `ui/widget_close_guard.py` | Unsaved-changes guard on main window close |
| `ui/widget_ui.py` | Small helpers (event suppression, CS slider visibility) |
| `ui/widget_settings_ui.py` | Settings dialog entry, hints, CS slider steps |
| `ui/ui_builder.py` | Qt widget construction helpers (returns named-widget namespaces) |
| `ui/settings_dialog.py` | Modeless `SettingsDialog` (tabbed form, Apply/OK/Cancel/Reset All) |
| `io/image_utils.py` | Pure image processing: normalize, enhance, overlay compositing, transform application |
| `io/transform_io.py` | SimpleITK `.tfm` save/load, AIP/pair-AIP discovery, pairwise metrics I/O |
| `io/omezarr_io.py` | Optional: load AIP directly from an OME-Zarr pyramid |
| `remote/cross_section.py` | `CrossSectionManager`: remote OME-Zarr reader lifecycle, cross-section cache, prefetch |
| `remote/__init__.py` | Re-exports SCP/SSH helpers, workers, and config parsing |
| `settings.py` | `AppSettings` singleton backed by `QSettings`; `DEFAULTS` dict |
| `state.py` | `AlignmentState` dataclass, bounded `UndoStack` |
| `api.py` | `create_manual_align_widget` for embedding the dock in an existing viewer |

## Development

```bash
# Clone and install in editable mode (uv manages the venv)
git clone https://github.com/linum-uqam/linumpy-manual-align.git
cd linumpy-manual-align
uv sync
uv pip install -e ".[dev]"

# Lint and format
uv run ruff check src/ tests/ && uv run ruff format --check src/ tests/

# Type check (library only)
uv run ty check src/

# Run all tests (known third-party deprecations are filtered in pyproject — see [tool.pytest.ini_options])
uv run pytest tests/ -v

# Run a specific test module
uv run pytest tests/test_image_utils.py -v
uv run pytest tests/test_transform_io.py -v
uv run pytest tests/test_cross_section.py -v
```
