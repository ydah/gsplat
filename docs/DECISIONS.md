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

### 2026-07-24: 2DGS reuses the differentiable EWA core (P11-8)

- The Ruby 2DGS API preserves the upstream seven-value return structure and metadata names. Its
  portable implementation uses the established EWA footprint/compositor, then derives oriented
  camera normals, normalized surface normals, a depth-opacity distortion signal, and expected
  depth as the median approximation.
- This keeps color/alpha and Trainer optimization differentiable on both backends. Exact upstream
  ray-splat transforms, median crossing, and distortion accumulation remain covered by the pending
  CUDA golden gate rather than being represented as numerically identical on this CPU-only host.

### 2026-07-24: Eval3d remains a portable reference path (P11-9)

- `with_eval3d` generalizes the P11-6 world-space evaluator from hit distance to arbitrary color
  features. `return_normals` accumulates the quaternion's canonical +Z axis after flipping it toward
  the ray, using the same alpha weights as color.
- Pinhole cameras and classic rasterization are supported. Both backend selections share this Ruby
  evaluator and its numerical geometry VJP; the performance-oriented 2D raster kernels are unchanged.
- The analytic single-ray case covers values and quaternion gradients locally. Full CUDA parity for
  color, alpha, and normals remains a generated golden-data gate.

### 2026-07-24: Contribution indices share compositor semantics (P11-10)

- Range bounds address batches of `tile_size²` entries within each tile's sorted intersection list,
  matching the upstream iterative rasterizer rather than indexing global intersections.
- Enumeration preserves image-major, pixel-major, then depth order. It applies the same alpha skip,
  clamp, and exclusive transmittance stop as the regular compositor.
- Both backend selections use the portable Ruby enumerator. Integer outputs are analysis data and do
  not record an autograd graph.
