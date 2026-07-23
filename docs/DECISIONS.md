# Design Decisions

Record only decisions that differ from, clarify, or resolve ambiguity in the design documents.

## Entries

### 2026-07-23: Golden reference version and CPU scope (P0-4)

- Pin Python reference data to the latest official release, `gsplat==1.5.3` (tag commit
  `937e29912570c372bed6747a5c9bf85fed877bae`), instead of a moving `main` branch.
- Upstream 1.5.3 returns elliptical radii shaped `[C,N,2]`; golden files retain that upstream shape.
  The scalar-radius v1 compatibility decision remains isolated to the Ruby projection implementation.
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
