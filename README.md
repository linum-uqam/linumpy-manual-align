# Manual Slice Alignment Tool

Napari-based interactive tool for manually correcting pairwise slice alignment
in the linumpy reconstruction pipeline.

## Usage

Uses the linumpy venv (must have linumpy installed in editable mode):

```bash
cd /path/to/linumpy
.venv/bin/python tools/manual-align/manual_align.py \
    --input_dir /path/to/bring_to_common_space/ \
    --transforms_dir /path/to/register_pairwise/ \
    --output_dir /path/to/manual_transforms/ \
    --level 1
```

## Workflow

1. Red/green overlay shows consecutive slice pair (AIP projections)
2. Adjust translation (drag or spinbox/arrows) and rotation (slider)
3. Save corrected transform — outputs `.tfm` compatible with pipeline
4. Copy saved `.tfm` files into `register_pairwise/slice_z##/` on server
5. Re-run pipeline from `stack` step with `-resume`

## Keyboard Shortcuts

- **Arrow keys**: Nudge translation ±1px (Shift: ±10px)
- **`[` / `]`**: Rotate ±0.1° (Shift: ±1.0°)
- **`N` / `P`**: Next / Previous slice pair
- **`S`**: Save current pair
- **Ctrl+Z / Ctrl+Shift+Z**: Undo / Redo
