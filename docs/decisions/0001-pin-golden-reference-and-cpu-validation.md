# 0001: Pin the golden reference and define CPU-only validation

- Status: Accepted
- Date: 2026-07-23

## Context

Golden data must remain reproducible while the upstream `main` branch and its public tensor shapes
continue to evolve. The development host also has no CUDA device, so it cannot execute the complete
upstream rasterizer locally.

## Decision

- Pin Python reference data to the latest official release, `gsplat==1.5.3` (tag commit
  `937e29912570c372bed6747a5c9bf85fed877bae`), instead of a moving `main` branch.
- Use the upstream 1.5.3 elliptical radii shape `[C,N,2]` in projection and golden files. The
  low-level intersection operation continues to accept legacy scalar radii.
- Do not create golden `last_ids`, because upstream 1.5.3 does not expose that internal rasterizer
  value. Raster goldens store public outputs, all public input gradients, and `means2d.absgrad`.
- Use the dependency-free generator dry run as the local P0-4 gate. Keep CPU partial-generation and
  the required CUDA command documented in `tools/README.md`.

## Consequences

Golden inputs and public output contracts are reproducible against an immutable upstream version.
Local development can validate the generator and all CPU-owned behavior, while complete raster
parity remains an explicit external CUDA acceptance gate.
