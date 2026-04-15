# Linumpy Manual Align

[![CI](https://github.com/linum-uqam/linumpy-manual-align/actions/workflows/ci.yml/badge.svg)](https://github.com/linum-uqam/linumpy-manual-align/actions/workflows/ci.yml)

Napari-based interactive tool for manually correcting pairwise slice alignment
in the [linumpy](https://github.com/linum-uqam/linumpy) reconstruction pipeline.

## Documentation

- [Napari plugin + Nextflow integration guide](docs/napari-plugin-nextflow-guide.md)

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

## Architecture

The package is split into focused modules to keep each one small and testable:

| Module | Responsibility |
|--------|---------------|
| `widget.py` | Main napari dock widget — coordinates all modules |
| `ui_builder.py` | Qt widget construction helpers (returns named-widget namespaces) |
| `cross_section.py` | `CrossSectionManager`: remote OME-Zarr reader lifecycle, cross-section cache, prefetch |
| `image_utils.py` | Pure image processing: normalize, enhance, overlay compositing, transform application |
| `transform_io.py` | SimpleITK `.tfm` save/load, AIP/pair-AIP discovery, pairwise metrics I/O |
| `server_transfer.py` | SCP/SSH download/upload workers, remote slice reader |
| `omezarr_io.py` | Optional: load AIP directly from an OME-Zarr pyramid |
| `state.py` | `AlignmentState` dataclass, bounded `UndoStack` |

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

# Run all tests
uv run pytest tests/ -v

# Run a specific test module
uv run pytest tests/test_image_utils.py -v
uv run pytest tests/test_transform_io.py -v
uv run pytest tests/test_cross_section.py -v
```
