# Linumpy Manual Align

[![CI](https://github.com/linum-uqam/linumpy-manual-align/actions/workflows/ci.yml/badge.svg)](https://github.com/linum-uqam/linumpy-manual-align/actions/workflows/ci.yml)

Napari-based interactive tool for manually correcting pairwise slice alignment
in the [linumpy](https://github.com/linum-uqam/linumpy) reconstruction pipeline.

## Installation

```bash
# From GitHub:
uv pip install git+https://github.com/linum-uqam/linumpy-manual-align.git

# With OME-Zarr support (requires linumpy):
uv pip install "linumpy-manual-align[zarr] @ git+https://github.com/linum-uqam/linumpy-manual-align.git"

# For development:
git clone https://github.com/linum-uqam/linumpy-manual-align.git
cd linumpy-manual-align
uv pip install -e ".[dev]"
```

## Usage

### With a data package (recommended)

The reconstruction pipeline exports a lightweight data package via
`linum_export_manual_align.py` (on the server). Download the package
locally, then align without needing linumpy installed:

```bash
linumpy-manual-align \
    --data_package /path/to/manual_align_package/ \
    --server_config ~/Downloads/sub-22/nextflow.config
```

### With server download/upload

When `--server_config` is provided, the UI shows Download/Upload buttons
for seamless transfer of data packages and manual transforms to/from the
reconstruction server.

### Directly from OME-Zarr volumes (requires linumpy)

```bash
linumpy-manual-align \
    --input_dir /path/to/bring_to_common_space/ \
    --transforms_dir /path/to/register_pairwise/ \
    --output_dir /path/to/manual_transforms/ \
    --level 1
```

## Workflow

1. Red/green overlay shows consecutive slice pair (AIP projections)
2. Adjust translation (drag or spinbox/arrows) and rotation (slider)
3. Save corrected transform — outputs `.tfm` compatible with pipeline
4. Upload transforms or copy to `register_pairwise/slice_z##/` on server
5. Re-run pipeline from `stack` step with `-resume`

## Keyboard Shortcuts

| Key | Action |
|-----|--------|
| Arrow keys | Nudge translation ±1px (Shift: ±10px) |
| `[` / `]` | Rotate ±0.1° (Shift: ±1.0°) |
| `N` / `P` | Next / Previous slice pair |
| `S` | Save current pair |
| Ctrl+Z / Ctrl+Shift+Z | Undo / Redo |

## Development

```bash
# Lint and format
uv run ruff check src/ && uv run ruff format --check src/

# Type check
uv run ty check src/

# Run tests
uv run pytest tests/ -v
```
