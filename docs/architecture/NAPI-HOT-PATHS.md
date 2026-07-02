# Napari Hot Paths — Baseline Audit (Phase 9)

**Status:** Baseline captured before Phase 9 lifecycle optimizations (D-18).  
**Scope:** Napari layer mutation paths in the manual-align viewer.  
**Related:** [REFACTOR-SEQUENCE.md Step 6](./REFACTOR-SEQUENCE.md), [BOUNDARIES.md](./BOUNDARIES.md), `src/linumpy_manual_align/ui/napari_layers.py`.

This document records **current (pre-optimization)** call graphs in the format:

`entry point → mixin method → napari API call`

Each section includes a **frequency** note and a **Planned changes (Phase 9)** pointer to the target state.

---

## 1. Pair switch (full teardown)

**Frequency:** Every pair switch and every projection-mode toggle (via `_load_pair_preserve_camera`).

```
pair_combo / keyboard nav
  → PairLoadingMixin._load_pair
      → viewer.layers.pop(0)  [loop until empty]     # widget_pair_loading.py ~L271-272
      → viewer.add_image(fixed_aip, ...)             # ~L279-288  ("Fixed z{fid:02d}{suffix}")
      → viewer.add_image(moving_aip, ...)            # ~L289-298  ("Moving z{mid:02d}{suffix}")
      → OverlayStateMixin._rebuild_layer_visibility  # composite add/remove per overlay mode
      → OverlayStateMixin._apply_state               # moving_layer.data / .translate
      → viewer.reset_view()                          # ~L362-363 unless preserve_camera
```

**Napari APIs touched:** `viewer.layers.pop`, `viewer.add_image`, `viewer.reset_view`, layer `.data`, `.translate`, `.visible`.

**Planned changes (Phase 9):** Wave 2 extracts teardown/create to `napari_layers.remove_all_layers` / `create_pair_layers`. Wave 3 adds incremental `update_pair_data` when shape, scale, and projection mode are unchanged (D-01–D-07); stable layer names `"Fixed"` / `"Moving"` (D-06).

---

## 2. Overlay mode toggle

**Frequency:** Overlay combo change (`Color` ↔ `Difference` ↔ `Checkerboard`).

```
combo_overlay.currentIndexChanged
  → InteractionMixin._on_overlay_mode_changed          # widget_interaction.py ~L35-43
      → OverlayStateMixin._rebuild_layer_visibility    # widget_overlay.py ~L29-57
          → fixed_layer.visible / moving_layer.visible
          → viewer.layers.remove(_composite_layer)     # switching to color
          → _composite_layer.colormap = ...              # diff ↔ checker in-place
          → viewer.add_image(comp, name="Composite")   # first diff/checker entry
      → OverlayStateMixin._refresh_composite           # ~L59-71 if non-color
          → _composite_layer.data = build_overlay(...)
```

**Napari APIs touched:** `viewer.layers.remove`, `viewer.add_image`, layer `.visible`, `.colormap`, `.data`.

**Planned changes (Phase 9):** Wave 2 moves logic to `napari_layers.set_overlay_mode` / `refresh_composite`. Wave 3 gates composite on `projection_mode == "xy"` — Z mode forces color and removes composite (D-10, D-13).

---

## 3. Enhancement change

**Frequency:** Enhancement combo change (`None` / `Edges` / `CLAHE` / `Sharpen`).

```
combo_enhance.currentIndexChanged
  → InteractionMixin._on_enhance_changed               # widget_interaction.py ~L50-85
      → fixed_layer.data = enhanced_fixed              # ~L61-62
      → OverlayStateMixin._apply_state (XY)            # ~L65-68  moving_layer.data baked rotation
      → moving_layer.data = enhanced_cs_or_static      # ~L70-80  XZ/YZ from cache or NPZ
      → OverlayStateMixin._refresh_composite           # ~L83-85  if diff/checker active
```

**Napari APIs touched:** layer `.data` in-place assignment on fixed and moving layers.

**Planned changes (Phase 9):** No extraction planned — enhancement remains in `InteractionMixin`. Incremental pair switch (Wave 3) re-pushes enhanced AIPs via `update_pair_data`.

---

## 4. Alignment apply (spinbox / slider drag)

**Frequency:** Every alignment nudge, spinbox change, or undo/redo that applies state.

```
spinbox / slider / keyboard nudge
  → OverlayStateMixin._apply_state                     # widget_overlay.py ~L75-121
      → moving_layer.data = baked_rotation (XY)        # ~L98-104
      → moving_layer.translate = [ty*sy, tx*sx]       # ~L106
      → moving_layer.translate (XZ/YZ pure shift)      # ~L107-117
      → OverlayStateMixin._refresh_composite           # ~L119-121 if XY diff/checker
          → _composite_layer.data = build_overlay(...)
```

**Napari APIs touched:** layer `.data`, `.rotate`, `.translate`, composite `.data`.

**Planned changes (Phase 9):** `_apply_state` stays in `OverlayStateMixin`. Wave 3 calls `refresh_composite` after incremental pair data push in diff/checker mode (D-05).

---

## 5. Projection mode toggle (XY ↔ Z)

**Frequency:** XY / Z mode button toggle; XZ ↔ YZ sub-toggle within Z mode.

```
mode button / Z-proj button
  → ProjectionEventMixin._on_mode_btn_toggled            # widget_projection.py ~L44-57
      → InteractionMixin._load_pair_preserve_camera    # widget_interaction.py ~L139-144
          → PairLoadingMixin._load_pair(preserve_camera=True)
              → [full teardown path — same as §1]
  → ProjectionEventMixin._on_z_proj_changed              # ~L59-65  (XZ ↔ YZ within Z)
      → _load_pair_preserve_camera
```

**Napari APIs touched:** Full layer teardown + recreate; camera preserved via snapshot/restore.

**Planned changes (Phase 9):** Projection mode change always triggers full teardown (D-09). Entering Z removes composite and forces color (D-10, D-13). Returning to XY rebuilds from AIP files; `_apply_state` restores alignment and overwrites stale cross-section pixels (D-12). Remote cross-section readers stay open across XY↔Z (D-11).

---

## 6. Cross-section refresh (OUT OF `napari_layers` SCOPE)

> **Audit-only.** Managed by `CrossSectionMixin`; excluded from `NapariLayerLifecycle` per Phase 8 D-21.

**Frequency:** Remote cross-section fetch completes while in XZ/YZ mode; cross-section slider drag.

```
CrossSectionManager signal
  → CrossSectionMixin._on_cross_section_ready          # widget_cross_section.py ~L191-216
      → moving_layer.data = enhance_aip(normalize_aip(img))  # ~L215-216
```

**Napari APIs touched:** `moving_layer.data` in-place write from remote slice.

**Planned changes (Phase 9):** No move into `napari_layers`. Documented here for audit completeness. XY return via `_load_pair` + `_apply_state` clears stale CS frames (D-12).

---

## Layer inventory guard (Phase 9 Wave 3)

**Planned:** `napari_layers.assert_layer_inventory` after every `_load_pair` — expected count 2 (color) or 3 (XY diff/checker); warns on unexpected layers (D-14).

---

## Memory benchmark (Phase 9 Wave 4)

**Planned:** Headless pytest `@pytest.mark.napari_benchmark` — 50+ XY pair switches, peak RSS + per-switch wall time (D-19–D-22). Git-based before/after comparison against this baseline doc.

---

*Baseline captured: Phase 9 Wave 1 (NAPI-03). Updated after Waves 2–4 with implementation notes.*
