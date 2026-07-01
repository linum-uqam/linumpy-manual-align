# Incremental Refactor Sequence

Ordered capability-based steps for reducing architectural debt without full mixin→controller decomposition. Each step is one mergeable change; the full test suite must stay green after every step.

**Standing test gate:** Run `uv run pytest tests/ -v` after every step. After boundary-affecting steps, also run `uv run lint-imports` (D-19).

**Out of scope:** Full mixin→controller decomposition (D-14) — deferred to a future requirement.

---

## Step 1 — Settings boundary decoupling

Remove remote→settings imports and neutralize `settings_runtime ↔ ServerConfig` coupling so the pyramid contract passes with **zero** grandfather ignores.

**Done when:** `uv run lint-imports` passes with no `ignore_imports` on the pyramid or remote-layer contracts; `uv run pytest tests/ -v` green.

**Risk:** Cross-section nudge behavior may regress if settings injection is incomplete; verify `tests/test_cross_section.py` and settings-related widget tests.

---

## Step 2 — Contracts completion (server download)

Wire `widget_server.py` download path through `load_manual_align_metadata` from contracts.

**Done when:** Existing server/download tests in `tests/test_widget.py` pass; metadata fields populated on server-first load.

**Risk:** Server path has the largest mixin surface; limit changes to `widget_server.py` only.

---

## Step 3 — Contract helper adoption (output paths)

Replace scattered path literals in ui mixins with contract helpers for slice directory naming.

**Done when:** Grep for ad-hoc `slice_z##` path construction in `ui/` is clean; pytest green.

**Risk:** Path mismatches break save/load round-trips; run save-validation tests.

---

## Step 4 — Shared package ingest helper

Extract a single entry point for package discovery/loading shared by server download and pair-loading paths.

**Done when:** New or updated tests cover both server and local pair-loading ingest; pytest green.

**Risk:** Touches two priority mixins (`widget_server.py`, `widget_pair_loading.py`); keep ≤2 mixins per PR.

---

## Step 5 — `ui/napari_layers.py` lifecycle interface

Document the create/update/remove API for napari layer lifecycle (pair-loading + overlay delegation). Stub module optional; interface doc is the deliverable.

**Done when:** Interface section committed in `ui/napari_layers.py` module docstring or companion doc; pytest green (no behavior change required).

**Risk:** Premature extraction before interface is agreed can force rework; document only in this step.

---

## Step 6 — Napari layer extraction (partial)

Move lifecycle calls from pair-loading and overlay mixins into `ui/napari_layers.py`. Touch at most two mixins.

**Done when:** `uv run pytest tests/ -v` green; layer create/update/remove behavior unchanged.

**Risk:** MRO `super()` chain — verify partial-stack widget tests after each mixin edit.

---

## Step 7 — Panel IA readiness checklist

Produce a signal map and section boundary checklist for future panel regrouping. **No widget regrouping in this step.**

**Done when:** Checklist committed under `docs/architecture/`; pytest green.

**Risk:** Scope creep into actual panel redesign — checklist only (D-18).

---

## Summary table

| Step | Capability | Mixins touched (max) | lint-imports gate |
|------|------------|---------------------|-------------------|
| 1 | Settings boundary decoupling | remote + settings modules | yes |
| 2 | Contracts completion (server) | `widget_server.py` | no |
| 3 | Contract helper adoption | ui mixins (paths) | no |
| 4 | Shared package ingest | server + pair-loading | no |
| 5 | napari_layers interface | none (doc/stub) | no |
| 6 | Napari layer extraction | ≤2 mixins | no |
| 7 | Panel IA checklist | none | no |

See also: [BOUNDARIES.md](./BOUNDARIES.md), [MIXIN-MATRIX.md](./MIXIN-MATRIX.md), [COVERAGE-BASELINE.md](./COVERAGE-BASELINE.md).
