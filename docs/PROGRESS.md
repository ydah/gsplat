# Implementation Progress

This file is the restart point for implementation sessions. Read it after `README.md`.

| Task | Status | Completed | Notes |
|---|---|---|---|
| P0-1 | Complete | 2026-07-23 | Gem foundation; 4 tests, 7 assertions |
| P0-2 | Complete | 2026-07-23 | Backend registry, selection, environment override, one-time fallback warning |
| P0-3 | Complete | 2026-07-23 | NPY v1.0 and stored/deflated NPZ; NumPy 2.3.1 interoperability verified |
| P0-4 | Complete (alternate DoD) | 2026-07-23 | 34-case generator and dependency-free dry run; CUDA goldens pending CUDA host |
| P0-5 | Complete | 2026-07-23 | allclose/golden/backend test helpers and Ruby 3.2–4.0 CI matrix |
| P1-1 | Complete | 2026-07-23 | Autograd Variable/Function/Context, branching, multi-output, no_grad, graph release |
| P1-2 | Complete | 2026-07-23 | Batched quaternion VJPs and closed-form 2x2/3x3 matrix operations |
| P1-3 | Complete (golden pending) | 2026-07-23 | Covariance/precision fwd+bwd, triu output, float64 gradcheck |
| P2-1 | Complete (golden pending) | 2026-07-23 | SH degrees 0–4, masks, arbitrary channels, analytic direction/coefficient VJPs |
| P3-1 | Complete (golden pending) | 2026-07-23 | Pinhole forward, world/camera primitives, scalar radii, culling and compensation |
| P3-2 | Complete (golden pending) | 2026-07-23 | Analytic projection VJPs for means, covariance, quaternion and scale |
| P3-3 | Complete (golden pending) | 2026-07-23 | Orthographic forward/backward and camera-model dispatch |
| P4-1 | Complete (golden pending) | 2026-07-23 | Tile AABBs, 64-bit intersection keys, sorting and prefix offsets |
| P5-1 | Complete | 2026-07-23 | Brute-force alpha compositor with arbitrary channels and backgrounds |
| P5-2 | Complete (golden pending) | 2026-07-23 | Vectorized tile compositor, masks, partial edge tiles and retained last ids |

## Phase gates

### P0 — Complete (alternate golden-data gate)

- Full suite: 25 tests, 91 assertions, no failures or skips.
- RuboCop: no offenses.
- Golden status: generator dry-run and syntax checks pass; complete CUDA output remains pending per
  `tools/README.md` because this host has no CUDA and the temporary PyTorch install exceeded disk capacity.
- Design differences: recorded in `docs/DECISIONS.md`.

### P1 — Complete (golden-data gate pending)

- L1/L2: analytic properties and float64 central differences pass.
- L3: `quat_covar_full.npz` test is present and skipped until `tools/README.md` CUDA generation is run.
- Full suite and lint evidence are recorded in the P1-3 commit.

### P2 — Complete (golden-data gate pending)

- L1/L2: known degree-zero/degree-one values and float64 direction/coefficient central differences pass.
- Masks suppress forward values and their corresponding VJPs; channels are not restricted to RGB.
- L3: `sh_deg3.npz` test is present and skipped until `tools/README.md` golden generation is run.
- Full suite: 52 tests, 130 assertions, no failures, 2 documented golden-data skips.
- RuboCop: no offenses.

### P3 — Complete (golden-data gate pending)

- L1/L2: hand-calculated pinhole/orthographic cases and float64 central differences pass.
- Covariance-direct and quaternion/scale paths both propagate analytic gradients.
- L3: pinhole and orthographic tests are present and skip until golden generation is run.
- Scalar radii follow the v1 design; golden visibility compares against upstream elliptical radii.
- Full suite: 63 tests, 155 assertions, no failures, 5 documented golden-data skips.
- RuboCop: no offenses.

### P4 — Complete (golden-data gate pending)

- L1: exact handcrafted tile counts, 64-bit keys, flattened ids and empty-tile offsets pass.
- Scalar v1 radii and upstream elliptical radii are both accepted.
- L3: `isect_c3_n1000.npz` coverage is present and skips until golden generation is run.
- Full suite: 68 tests, 163 assertions, no failures, 6 documented golden-data skips.
- RuboCop: no offenses.

### P5 — Complete (golden-data gate pending)

- L1/L4: single-pixel analytical cases and tile-vs-brute-force image comparisons pass.
- Partial edge tiles, arbitrary channels, backgrounds and tile masks are covered.
- L3: `raster_rgb.npz` forward coverage is present and skips until CUDA golden generation is run.
- Full suite: 75 tests, 175 assertions, no failures, 7 documented golden-data skips.
- RuboCop: no offenses.
