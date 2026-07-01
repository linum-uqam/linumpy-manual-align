# Coverage Baseline — contracts/ and io/

**Generated:** 2026-07-01  
**Scope:** `linumpy_manual_align.contracts` and `linumpy_manual_align.io` (headless layers)

This document is the pre-refactor coverage baseline for the two pure layers. Use it to compare coverage before and after structural refactors. No fail-under threshold is enforced yet (D-09).

## Regenerate

```bash
uv run pytest tests/ \
  --cov=linumpy_manual_align.contracts \
  --cov=linumpy_manual_align.io \
  --cov-report=term-missing \
  --cov-report=markdown:docs/architecture/coverage-raw.md
```

The same command is documented in the README Development section.

## Summary

| Metric | Value |
|--------|-------|
| Combined statements | 631 |
| Combined missing | 97 |
| **Combined coverage** | **85%** |

## Per-module breakdown

| Module | Stmts | Missing | Coverage |
|--------|------:|--------:|---------:|
| `contracts/__init__.py` | 7 | 0 | 100% |
| `contracts/layout.py` | 116 | 6 | 95% |
| `contracts/metadata.py` | 111 | 23 | 79% |
| `contracts/models.py` | 22 | 0 | 100% |
| `contracts/session_state.py` | 40 | 0 | 100% |
| `contracts/upload_readiness.py` | 106 | 7 | 93% |
| `io/__init__.py` | 1 | 0 | 100% |
| `io/image_utils.py` | 66 | 0 | 100% |
| `io/omezarr_io.py` | 48 | 44 | 8% |
| `io/transform_io.py` | 114 | 17 | 85% |

## Raw report

Full pytest-cov markdown output: [coverage-raw.md](./coverage-raw.md)

## Notes

- `io/omezarr_io.py` is low because it requires the optional `zarr` extra and is not exercised by the default test suite.
- Baseline generation depends on lazy package `__init__.py` (plan 06-01) so coverage instrumentation on Python 3.14 does not eagerly import scipy/napari.
