# Napari Manual Align Plugin Guide

This project provides an interactive Napari tool that lets you manually correct
pairwise slice alignment before stacking in the linumpy reconstruction pipeline.

The plugin is launched by the CLI entrypoint:

linumpy-manual-align

It opens a Napari viewer with a docked control panel where you can inspect each
consecutive slice pair and correct translation and rotation.

## What The Plugin Does

For each pair of consecutive slices:

1. Loads fixed and moving slice AIP projections.
2. Shows them as a green (fixed) and red (moving) additive overlay.
3. Lets you adjust TX, TY, and rotation interactively.
4. Saves a pipeline-compatible transform package in `manual_transforms/slice_z##/`.

Saved outputs per slice include:

- `transform.tfm` (SimpleITK Euler3DTransform)
- `offsets.txt` (carried from automated registration if available)
- `pairwise_registration_metrics.json` (marked with `"source": "manual"`)

## Input Modes

The tool supports three workflows.

### 1) Server-first workflow (recommended)

Start with no data arguments or only a server config:

```bash
linumpy-manual-align
# or
linumpy-manual-align --server_config ~/Downloads/sub-22/nextflow.config
```

Then in the Server panel:

1. Browse to a local `nextflow.config`.
2. Click Download Data.
3. Align slices.
4. Save and Upload Transforms.

### 2) Local data package workflow

If a package was already exported/downloaded:

```bash
linumpy-manual-align --data_package /path/to/manual_align_package/
```

Expected package layout:

```text
manual_align_package/
  aips/
    slice_z00.npz
    slice_z01.npz
    ...
  transforms/
    slice_z01/transform.tfm
    ...
  manual_align_metadata.json  # optional
```

### 3) Direct OME-Zarr workflow

Requires the optional ome-zarr extra:

```bash
linumpy-manual-align \
  --input_dir /path/to/bring_to_common_space/ \
  --transforms_dir /path/to/register_pairwise/ \
  --output_dir /path/to/manual_transforms/ \
  --level 1
```

## UI Walkthrough

Main controls:

- Slice Pair selector: navigate pair-by-pair (`zAA -> zBB`).
- Transform controls: TX, TY (pixels at working level), Rotation (degrees).
- Display controls: gamma and fixed/moving opacity.
- Actions: Load Automated, Reset, Undo, Redo.
- Save: Save Current or Save All Modified and Exit.
- Server: choose config, host, download data, upload manual transforms.

Common shortcut keys:

- Arrow keys: nudge translation
- `[` and `]`: rotate
- `N` / `P`: next / previous pair
- `S`: save current pair
- Undo/redo: Cmd+Z and Cmd+Shift+Z on macOS, Ctrl variants elsewhere

## Transform Semantics And Pipeline Compatibility

The plugin writes transforms at full-resolution coordinates even when you align
at a downsampled pyramid level.

If you align at level L, translation and center values are scaled by:

$$
2^L
$$

before saving to `transform.tfm`.

This allows transforms to be consumed directly by the downstream stack step.

The generated `pairwise_registration_metrics.json` also stores manual metadata,
including the working-level values and the selected pyramid level.

## Nextflow Integration

The plugin is designed around the linumpy Nextflow output structure.

### Server config parsing

When you provide a local `nextflow.config`, the tool derives the subject ID from
the config parent folder name (for example `sub-22`) and builds the remote path:

- remote workspace: `/scratch/workspace/<subject_id>`
- remote output: `/scratch/workspace/<subject_id>/output`

### Download source on server

Download pulls from:

`output/make_manual_align_package/manual_align_package/`

under the remote output directory, recursively.

### Upload destination on server

Upload pushes local `manual_transforms/slice_z*/` folders to:

`output/manual_transforms/`

under the same remote output directory.

## Typical End-To-End Nextflow Flow

1. Run the Nextflow pipeline through pairwise registration and package export.
2. Launch `linumpy-manual-align`.
3. Download the manual alignment package (or open local package).
4. Review and correct problematic pairs.
5. Save transforms.
6. Upload transforms to server (or copy manually to `output/manual_transforms/`).
7. Re-run the pipeline from stack with resume.

## Practical Notes

- Use `--slices` to focus only on problematic moving slices.
- If automated transform exists, Load Automated gives a better starting point.
- Save frequently when iterating through many pairs.
- Download/upload use `scp` and remote directory creation uses `ssh`; ensure keys
  and host access are configured in your shell environment.
