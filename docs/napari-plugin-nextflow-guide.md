# Napari Manual Align Plugin Guide

This guide covers the full workflow for the `linumpy-manual-align` interactive
Napari tool: exporting a data package from the server, correcting slice
alignment, and uploading results back to the pipeline.

---

## What the Plugin Does

For each consecutive slice pair the plugin:

1. Loads pre-computed AIP projections (XY) and center cross-sections (XZ, YZ).
2. Displays them as interactive overlays for lateral (XY) and depth (Z) alignment.
3. Lets you adjust TX, TY, rotation, and Z-overlap interactively.
4. Saves pipeline-compatible transform packages to `manual_transforms/slice_z##/`.

Saved outputs per slice:

- `transform.tfm` — SimpleITK `Euler3DTransform`
- `offsets.txt` — Z-overlap indices (full-resolution voxels)
- `pairwise_registration_metrics.json` — marked `"source": "manual"`

---

## Data Package Layout

The data package produced by `linum_export_manual_align` (linumpy) has the
following layout:

```text
manual_align_package/
  aips/                       # XY AIP (mean over Z) per slice
    slice_z00.npz
    slice_z01.npz
    ...
  aips_xz/                    # XZ center cross-section per slice
    slice_z00.npz
    ...
  aips_yz/                    # YZ center cross-section per slice
    slice_z00.npz
    ...
  transforms/
    slice_z01/
      transform.tfm           # automated pairwise transform
      offsets.txt             # Z-overlap (fixed_z, moving_z voxel indices)
      pairwise_registration_metrics.json
    slice_z02/
      ...
  manual_align_metadata.json  # pyramid level, slice list, directory names
```

Each `.npz` file contains two arrays:

- `aip` — float32 image
- `scale` — pixel spacing in physical units (2 values for cross-sections,
  3 values for XY AIPs where the Z component is stripped on load)

The XZ/YZ cross-sections are taken at the Y/X position with the highest
integrated intensity, so they always cut through tissue even when the tissue
is not centred in the field.

---

## Input Modes

### 1. Server-first workflow (recommended)

Start with no data arguments or with a pre-filled server config:

```bash
linumpy-manual-align
# or
linumpy-manual-align --server_config ~/Downloads/sub-22/nextflow.config
```

Then in the **Server** panel:

1. Browse to or confirm the local `nextflow.config`.
2. Click **Download Data** — fetches the package from the server via SCP.
3. Correct alignment.
4. Click **Save All & Exit** then **Upload Transforms**.

### 2. Local data package workflow

```bash
linumpy-manual-align --data_package /path/to/manual_align_package/
```

### 3. Direct OME-Zarr workflow

Requires the optional `ome-zarr` extra.  AIPs are computed on the fly from
the full 3-D volumes; no `aips_xz/` or `aips_yz/` cross-sections are available
in this mode.

```bash
linumpy-manual-align \
  --input_dir /path/to/bring_to_common_space/ \
  --transforms_dir /path/to/register_pairwise/ \
  --output_dir /path/to/manual_transforms/ \
  --level 1
```

---

## UI Walkthrough

### Pair selector

The **Slice Pair** combo at the top shows `zAA → zBB`.  Use **N** / **P** or
the combo directly to navigate.  Pairs with unsaved changes are indicated in
the status bar.

### XY Alignment mode

Toggle the **XY** button to enter lateral alignment mode.

- **TX / TY** — horizontal and vertical translation in pixels at the working
  pyramid level.  The spinboxes, arrow keys, and layer dragging all stay in sync.
- **Rotation** — counter-clockwise degrees, controlled by the slider or spinbox.
- **Load Automated** — restores the translation and rotation from the automated
  pairwise registration (useful as a starting point).
- **Reset** — zeroes all transform parameters.
- **Undo / Redo** — per-pair history (up to 500 states).

### Z Alignment mode

Toggle the **Z** button to inspect and correct the depth overlap.  This mode
is only available when `aips_xz/` and `aips_yz/` exist in the package.

- **View** — switch between the XZ (coronal-like) and YZ (sagittal-like)
  cross-section overlays.
- **Fixed Z start** / **Moving Z start** — full-resolution Z voxel indices at
  which the fixed and moving slices begin their overlap region.  These are read
  from `offsets.txt` and saved back when the pair is saved.
- **Relative shift** label — shows `fixed_z − moving_z` for quick reference.
- The moving cross-section is displayed offset by `(fixed_z − moving_z) / 2^L`
  pixels (where L is the pyramid level) so you can visually verify the overlap.

### Display panel

#### Overlay

Three modes for comparing fixed and moving images side-by-side:

| Mode | Description | Best for |
|------|-------------|----------|
| **Color (R/G)** | Additive red/green blend; yellow = aligned | General alignment |
| **Difference** | \|fixed − moving\| grayscale; bright = misaligned | Detecting subtle offsets |
| **Checkerboard** | Alternating tiles of fixed/moving | Checking local continuity |

The **Tile size** spinbox (2–512 px, default 16 px) controls the checkerboard
granularity.  Smaller tiles reveal finer local misalignment; larger tiles give a
cleaner global picture.

#### Enhance

A pre-processing step applied to each AIP independently before display.
Switching mode does not require reloading from disk.

| Mode | Description | Best for |
|------|-------------|----------|
| **None** | Normalized AIP as-is | General use |
| **Edges (Sobel)** | Gradient magnitude; tissue boundaries only | Oblique/angled cuts where edges are the clearest landmark |
| **CLAHE** | Adaptive histogram equalization | Sagittal cuts where projection blur creates uneven contrast across the image |
| **Sharpen** | Unsharp mask (1.5× high-frequency boost) | Blurry projections that need mild crispening |

Overlay and enhancement modes are independent and can be combined freely.
For example, **Edges + Checkerboard** is very effective for angled cuts.

#### Gamma / Opacity

Adjust gamma and per-layer opacity.  In Difference/Checkerboard modes, gamma
and opacity apply to the composite layer.

### Save / Server panels

- **Save Current (S)** — saves the transform for the current pair.
- **Save All & Exit** — saves all modified pairs and closes the viewer.
- **Upload Transforms** — pushes `manual_transforms/slice_z*/` to the server
  via SCP.

---

## Transform Semantics and Pipeline Compatibility

Transforms are always saved at **full-resolution** coordinates regardless of the
working pyramid level.  If you align at level L, all translation and rotation
centre values are multiplied by `2^L` before writing `transform.tfm`.

The downstream `stack` step reads these transforms directly; no manual scaling
is needed.

---

## Nextflow Integration

### Server config parsing

When you provide a local `nextflow.config` the tool derives the subject ID from
the config parent folder (e.g. `sub-22`) and constructs remote paths:

- remote workspace: `/scratch/workspace/<subject_id>/`
- remote output:    `/scratch/workspace/<subject_id>/output/`

### Download source

```
{remote_output}/make_manual_align_package/manual_align_package/
```

Downloaded recursively; includes `aips/`, `aips_xz/`, `aips_yz/`, `transforms/`,
and `manual_align_metadata.json`.

### Upload destination

```
{remote_output}/manual_transforms/
```

One `slice_z##/` folder per corrected pair, each containing `transform.tfm`,
`offsets.txt`, and `pairwise_registration_metrics.json`.

---

## Typical End-to-End Flow

1. Run the Nextflow pipeline through `make_manual_align_package`.
2. Launch `linumpy-manual-align` (with server config or local package).
3. For each problematic pair:
   a. Use **XY Alignment** to correct lateral shift and rotation.
   b. Switch to **Z Alignment** to verify the overlap depth.
   c. Press **S** to save.
4. Click **Save All & Exit**.
5. Upload transforms to the server.
6. Re-run the pipeline from `stack` with `-resume`.

---

## Practical Notes

- Use `--slices` (export script) or `--filter_slices` (CLI) to focus on a
  subset of problematic pairs.
- **Load Automated** gives a better starting point than zero for pairs where
  automated registration partially succeeded.
- For oblique cuts (e.g. 45°): try **Edges + Checkerboard** — the gradient
  image makes tissue edges pop regardless of intensity, and the checkerboard
  reveals discontinuities at tile boundaries.
- For sagittal cuts with projection blur: try **CLAHE + Difference** — CLAHE
  equalises local contrast so dim periphery and bright core are comparable,
  and the difference image quantifies remaining misalignment precisely.
- Download and upload use `scp`; remote directory creation uses `ssh`.  Ensure
  SSH key authentication is configured for the server host.
