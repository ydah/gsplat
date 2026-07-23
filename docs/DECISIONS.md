# Design Decisions

Record only decisions that differ from, clarify, or resolve ambiguity in the design documents.

## Entries

### 2026-07-23: Golden reference version and CPU scope (P0-4)

- Pin Python reference data to the latest official release, `gsplat==1.5.3` (tag commit
  `937e29912570c372bed6747a5c9bf85fed877bae`), instead of a moving `main` branch.
- Upstream 1.5.3 returns elliptical radii shaped `[C,N,2]`; projection and golden files use that
  public shape. The low-level intersection operation still accepts legacy scalar radii.
- Upstream 1.5.3 does not expose rasterizer `last_ids`. Raster goldens therefore store public outputs,
  all public input gradients, and `means2d.absgrad`, but no `last_ids`.
- This development host has no PyTorch/gsplat installation or CUDA device. Per P0-4's alternate DoD,
  the dependency-free dry run is the local acceptance gate. `tools/README.md` documents CPU partial
  generation and the required CUDA command for committing the complete golden set.

### 2026-07-23: Native paths retain exact Ruby fallbacks (P10)

- Native kernels target contiguous `Numo::SFloat`, the production training dtype.
- Float64 gradchecks, masked SH, and analytic projection/SH backward continue through the Ruby
  implementation. The `:native` backend remains functionally complete while avoiding a second,
  divergent copy of derivative formulas.
- Performance reports label these paths as hybrid and measure each accelerated operation explicitly.
- Raster backward uses OpenMP atomic scatter-add instead of per-worker full gradient buffers. The
  100k/800×800 benchmark scales from 279.790 ms at one thread to 79.252 ms at eight threads while
  avoiding `O(workers × N × channels)` temporary memory.

### 2026-07-23: Distorted cameras use the shared Ruby projection (P11-5)

- Equidistant fisheye and OpenCV radial/tangential/thin-prism projection use one float32/float64
  implementation for both backends. Selecting `:native` delegates these camera models to the Ruby
  path so camera semantics and numerical derivatives cannot diverge.
- Pinhole radial coefficients follow gsplat 1.5.3's rational order: `k1..k3` form the numerator and
  `k4..k6` form the denominator. Fisheye coefficients multiply `theta^3..theta^9`.
- Distortion coefficients are calibration constants. Gradients are provided for Gaussian geometry,
  while coefficient and intrinsic optimization remain outside this phase.

### 2026-07-24: World-space extension paths prioritize one exact reference (P11-6)

- Hit-distance modes evaluate the anisotropic Gaussian in world space and replace the final feature
  channel with the per-pixel closest-approach distance, matching the upstream eval3d definition.
- The same Ruby implementation serves both backends. Its geometry VJP uses float64/float32 central
  differences so the extended path remains differentiable without maintaining a second derivative
  implementation; the core 2D raster path retains its analytic/native backward.
- `global_z_order: false` changes projection depths to camera-space Euclidean center distance. Near
  and far culling continue to use camera z, and the Euclidean norm VJP is propagated to means.
