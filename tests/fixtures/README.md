# Contract test fixtures

Committed synthetic fixtures for headless workflow contract tests. They mirror
the `make_manual_align_package` export shape and the `manual_transforms/` upload
destination without requiring napari, SSH, or a pipeline run.

## Fixture roots

| Path | Purpose |
|------|---------|
| `manual_align_package/` | Canonical data package: 3 AIP `.npz` slices (z00–z02), automated `transforms/slice_z01` and `slice_z02`, and `manual_align_metadata.json`. |
| `manual_transforms/` | Golden manual output for moving IDs 1 and 2: `transform.tfm`, `offsets.txt`, and `pairwise_registration_metrics.json` with `source: "manual"`. |

## Tiny binary policy

AIP `.npz` files use deterministic 8×8 float32 arrays. Transform `.tfm`, `offsets.txt`,
and metrics JSON are short committed files kept small for fast CI.

Negative and edge-case variations are **not** committed here. Later tests generate
mutations at runtime via pytest factories and `copy_fixture_tree` in `conftest.py`,
which copies a fixture tree into `tmp_path` so tests never mutate these originals.

## Golden update policy

Intentional changes to golden numeric values (e.g. slice_z01 offsets or rotation)
require:

1. Regenerating files with `save_transform` from `linumpy_manual_align.io.transform_io`
   (no in-repo regeneration helper).
2. Explaining the change in the test or PR so reviewers know the contract shifted.

Golden slice_z01: `level=1`, `tx=8.0`, `ty=-5.0`, `rotation_deg=1.5`, `center=(120, 90)`, `offsets=(3, 7)`.

Golden slice_z02: `level=0`, `tx=-4.0`, `ty=6.0`, `rotation_deg=-2.0`, `center=(64, 64)`, `offsets=(2, 9)`.
