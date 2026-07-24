# Implementation Progress

This file is the restart point for implementation sessions. Read it after `README.md`.

| Task | Status | Completed | Notes |
|---|---|---|---|
| P0-1 | Complete | 2026-07-23 | Gem foundation; 4 tests, 7 assertions |
| P0-2 | Complete | 2026-07-23 | Backend registry, selection, environment override, one-time fallback warning |
| P0-3 | Complete | 2026-07-23 | NPY v1.0 and stored/deflated NPZ; NumPy 2.3.1 interoperability verified |
| P0-4 | Complete (alternate DoD) | 2026-07-23 | 47-case generator and dependency-free dry run; CUDA goldens pending CUDA host |
| P0-5 | Complete | 2026-07-23 | allclose/golden/backend test helpers and Ruby 3.2–4.0 CI matrix |
| P1-1 | Complete | 2026-07-23 | Autograd Variable/Function/Context, branching, multi-output, no_grad, graph release |
| P1-2 | Complete | 2026-07-23 | Batched quaternion VJPs and closed-form 2x2/3x3 matrix operations |
| P1-3 | Complete (golden pending) | 2026-07-23 | Covariance/precision fwd+bwd, triu output, float64 gradcheck |
| P2-1 | Complete (golden pending) | 2026-07-23 | SH degrees 0–4, masks, arbitrary channels, analytic direction/coefficient VJPs |
| P3-1 | Complete (golden pending) | 2026-07-23 | Pinhole forward, world/camera primitives, culling and compensation |
| P3-2 | Complete (golden pending) | 2026-07-23 | Analytic projection VJPs for means, covariance, quaternion and scale |
| P3-3 | Complete (golden pending) | 2026-07-23 | Orthographic forward/backward and camera-model dispatch |
| P4-1 | Complete (golden pending) | 2026-07-23 | Tile AABBs, 64-bit intersection keys, sorting and prefix offsets |
| P5-1 | Complete | 2026-07-23 | Brute-force alpha compositor with arbitrary channels and backgrounds |
| P5-2 | Complete (golden pending) | 2026-07-23 | Vectorized tile compositor, masks, partial edge tiles and retained last ids |
| P6-1 | Complete (golden pending) | 2026-07-23 | Reverse tile scan VJPs and brute-force gradient cross-check |
| P6-2 | Complete (golden pending) | 2026-07-23 | Optional per-pixel absolute mean-gradient accumulation |
| P7-1 | Complete (golden pending) | 2026-07-23 | High-level render modes, SH, antialiasing, depth normalization and metadata |
| P7-2 | Complete | 2026-07-23 | Self-consistency image fitter, 128px fixture and monotonic PSNR E2E |
| P8-1 | Complete | 2026-07-23 | Adam groups, bias correction, editable state and exponential scheduling |
| P8-2 | Complete | 2026-07-23 | Strategy lifecycle and synchronized duplicate/split/remove/reset operations |
| P8-3 | Complete (golden pending) | 2026-07-23 | Default densification statistics, growth, pruning and opacity reset |
| P8-4 | Complete (golden pending) | 2026-07-23 | Closed-form relocation, weighted sampling, covariance noise and MCMC strategy |
| P9-1 | Complete | 2026-07-23 | Inria PLY writer and arbitrary-order ASCII/binary little-endian reader |
| P9-2 | Complete | 2026-07-23 | COLMAP bin/txt cameras, images and 100-point fixture with pose conversion |
| P9-3 | Complete (golden pending) | 2026-07-23 | k-NN initialization, SH color helpers, scene scale and differentiable SSIM |
| P9-4 | Complete | 2026-07-23 | NPZ training checkpoints and vips/chunky_png RGB image IO |
| P9-5 | Complete | 2026-07-23 | Multi-view Trainer, staged SH, strategy hooks and training/render CLIs |
| P10-1 | Complete | 2026-07-23 | Optional C extension, Numo bridge, GVL release and build fallback |
| P10-2 | Complete (hybrid) | 2026-07-23 | C float32 projection/SH forward; analytic backward and float64 fallback |
| P10-3 | Complete | 2026-07-23 | C tile enumeration, stable 64-bit radix sort and offset encoding |
| P10-4 | Complete | 2026-07-23 | GVL-free OpenMP raster forward/backward with atomic scatter-add |
| P11-1 | Complete (golden pending) | 2026-07-23 | N-D features, differentiable channel chunking and radius clipping |
| P11-2 | Complete (golden pending) | 2026-07-23 | Full and packed direct covariance forward/backward across the renderer |
| P11-3 | Complete (golden pending) | 2026-07-23 | Axis-aligned elliptical radii in Ruby/native projection and metadata |
| P11-4 | Complete | 2026-07-23 | Morton grid sort, PNG quantization, SH K-means and compatible metadata layout |
| P11-5 | Complete (golden pending) | 2026-07-23 | Fisheye and OpenCV radial/tangential/thin-prism projection with VJPs |
| P11-6 | Complete (golden pending) | 2026-07-24 | Euclidean center sorting and per-pixel anisotropic ray-hit distance modes |
| P11-7 | Complete | 2026-07-24 | First-axis visibility-masked Adam parameter and moment updates |
| P11-8 | Complete (golden pending) | 2026-07-24 | 2DGS API, auxiliary geometry buffers and Trainer mode |
| P11-9 | Complete (golden pending) | 2026-07-24 | World-space color evaluation, ray-facing accumulated normals and numerical VJPs |
| P11-10 | Complete (golden pending) | 2026-07-24 | Iterative per-pixel Gaussian contribution-index enumeration |
| P12-1 | Complete | 2026-07-24 | README/Migration guide, executable quick start, example smoke tests and 100% public YARD coverage |
| P12-2 | Complete (external gates recorded) | 2026-07-24 | Design workloads, benchmark results, enabled CI matrix and G1–G6 acceptance report |
| P12-3 | Complete | 2026-07-24 | Version 1.0.0, Apache-2.0, release metadata, gem build and clean native install/render verified |

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
- Projection radii use the upstream `[C,N,2]` axis-aligned elliptical representation.
- Full suite: 63 tests, 155 assertions, no failures, 5 documented golden-data skips.
- RuboCop: no offenses.

### P4 — Complete (golden-data gate pending)

- L1: exact handcrafted tile counts, 64-bit keys, flattened ids and empty-tile offsets pass.
- Elliptical radii are primary; legacy scalar radii remain accepted by the low-level intersection API.
- L3: `isect_c3_n1000.npz` coverage is present and skips until golden generation is run.
- Full suite: 68 tests, 163 assertions, no failures, 6 documented golden-data skips.
- RuboCop: no offenses.

### P5 — Complete (golden-data gate pending)

- L1/L4: single-pixel analytical cases and tile-vs-brute-force image comparisons pass.
- Partial edge tiles, arbitrary channels, backgrounds and tile masks are covered.
- L3: `raster_rgb.npz` forward coverage is present and skips until CUDA golden generation is run.
- Full suite: 75 tests, 175 assertions, no failures, 7 documented golden-data skips.
- RuboCop: no offenses.

### P6 — Complete (golden-data gate pending)

- L2: float64 central differences pass for means, conics, colors, opacities and backgrounds.
- L4: tile and brute-force backward gradients match exactly on the reference scene.
- `absgrad` accumulates absolute per-pixel mean contributions and allocates no buffer when disabled.
- L3: raster gradients and absolute gradients skip until CUDA golden generation is run.
- Full suite: 81 tests, 190 assertions, no failures, 9 documented golden-data skips.
- RuboCop: no offenses.

### P7 — Complete (golden-data gate pending)

- High-level RGB/depth modes, SH, antialiasing, metadata and end-to-end gradients pass.
- The reduced image-fit E2E improves monotonically from 11.77 dB to 28.89 dB in 12 steps.
- A 128×128 PPM fixture and a configurable 2,000-Gaussian/300-step example are included.
- L3 render fixtures remain pending CUDA golden generation.
- Full suite: 88 tests, 248 assertions, no failures, 9 documented golden-data skips.
- RuboCop: no offenses.

### P8 — Complete (golden-data gate pending)

- Adam bias correction, exponential scheduling and synchronized structural state edits pass.
- Default strategy growth/pruning and MCMC relocation/growth/noise paths are covered.
- MCMC equation 9 is verified analytically; noise mean/variance and a reduced image fit pass.
- L3 strategy and relocation fixtures remain pending CUDA golden generation.
- Full suite: 109 tests, 323 assertions, no failures, 11 documented golden-data skips.
- RuboCop: no offenses.

### P9 — Complete (golden-data gate pending)

- Inria PLY and COLMAP bin/txt parsing pass round-trip and paired-fixture tests.
- k-NN initialization, scene scale, L1/SSIM/PSNR, checkpoint restart, and optional image backends pass.
- The eight-view synthetic Trainer E2E exceeds 28 dB in 80 steps; the production default remains 30,000 steps.
- `simple_trainer.rb` and `render_path.rb` provide the documented real-COLMAP workflow.
- L3 SSIM remains pending golden generation; optional image tests skip under Bundler when neither backend is declared.
- Full suite: 131 tests, 817 assertions, no failures, 14 documented skips.
- RuboCop: no offenses.

### P10-3 — Complete

- Native float32 intersections exactly match the Ruby keys, flattened ids, counts and offsets.
- The stable eight-pass radix sort preserves key/id pairing and runs without the Ruby GVL.
- The 1,000-Gaussian profile improves from 2.505 ms to 0.045 ms (55.7×).
- Full suite: 136 tests, 836 assertions, no failures, 14 documented skips.
- RuboCop: no offenses.

### P10 — Complete

- Contiguous float32 projection, SH, intersections, raster forward and raster backward use C kernels.
- Raster forward owns disjoint tiles; backward uses bounded-memory atomic scatter-add and supports
  arbitrary channels, backgrounds, masks and absolute mean gradients.
- At 100k Gaussians and 800×800, 8-thread forward is 29.643 ms and combined forward/backward is
  108.895 ms, within the 150/400 ms design targets.
- The explicit `:native` suite passes with one and eight OpenMP threads.
- Full suite: 137 tests, 845 assertions, no failures, 14 documented skips.
- RuboCop: no offenses.

### P11 — Complete (golden-data gate pending)

- Extended rendering covers feature chunking, direct covariance, elliptical radii, distorted/fisheye
  cameras, distance modes, eval3d normals, 2DGS, SelectiveAdam, and contribution enumeration.
- Each extension has analytic/property coverage and a pinned Python generator case where upstream
  execution is required. CUDA fixtures remain the documented external gate.
- Both backend selections pass the same 181-test, 977-assertion suite with 22 documented skips.
- RuboCop: no offenses.

### P12-1 — Complete

- README includes installation, executable quick start, conventions, API mapping, feature status,
  limitations, and the three production examples. `docs/MIGRATION.md` covers Python migration.
- `test/readme_test.rb` executes the exact quick-start fence. Example smoke tests run a real image-fit
  step, parse all COLMAP fixture records, and validate a PLY orbit setup without optional image gems.
- `yard stats`: 40 files, 129 methods, 124 attributes, 22 constants, undocumented 0 (100%).
- Ruby and native selections: 186 tests, 1,005 assertions, no failures, 22 documented skips.
- RuboCop: 130 files, no offenses.

### P12-2 — Complete (external validation pending)

- The 100k/800×800 workload passes all timing targets: Ruby 3,527.990/8,047.024 ms and
  native 30.419/112.573 ms for forward/combined forward+backward.
- The full 50k/512²/2,000-step native image fit completes in 157.457 seconds versus the
  30-minute target. The COLMAP benchmark path completes a one-step fixture smoke run.
- `docs/BENCHMARKS.md` records commands and limitations; `docs/ACCEPTANCE.md` maps G1–G6 to evidence.
  CUDA golden parity (G1/G2) and real-capture 30k quality (G4) remain explicit external gates.
- CI now runs Ruby 3.2–4.0, native build/full suite, documentation, and focused E2E jobs.
- Ruby and native selections: 188 tests, 1,041 assertions, no failures, 22 documented skips.
- YARD: undocumented 0 (100%). RuboCop: 132 files, no offenses. Workflow YAML parses successfully.

### P12-3 — Complete

- Version 1.0.0, release metadata, a bounded runtime dependency, and the release file manifest are
  covered by `test/gemspec_test.rb`.
- `gem build gsplat.gemspec` produces `gsplat-1.0.0.gem`. Installing that artifact into an empty
  temporary gem home compiles and loads the native extension, then completes a 1×1 render without
  loading files from the working tree.
- Ruby and native selections: 194 tests, 1,163 assertions, no failures, 22 documented skips.
- YARD: undocumented 0 (100%). RuboCop: 134 files, no offenses.
- Following explicit confirmation, the source license and gem metadata use Apache-2.0. No gem has
  been published.
