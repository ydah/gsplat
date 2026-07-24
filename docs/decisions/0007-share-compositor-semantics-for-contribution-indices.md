# 0007: Share compositor semantics for contribution indices

- Status: Accepted
- Date: 2026-07-24

## Context

`rasterize_to_indices_in_range` exposes iterative per-pixel contribution analysis. Its range
coordinates, ordering, and termination rules must match the regular compositor or analysis results
will disagree with rendered pixels.

## Decision

- Interpret range bounds as batches of `tile_size²` entries within each tile's sorted intersection
  list, matching the upstream iterative rasterizer rather than indexing global intersections.
- Preserve image-major, pixel-major, then depth order.
- Apply the compositor's alpha skip, clamp, and exclusive transmittance stop.
- Use the portable Ruby enumerator for both backend selections and do not record an autograd graph
  for integer analysis outputs.

## Consequences

Contribution enumeration is deterministic and agrees with regular rasterization semantics. It runs
at reference speed under either backend and intentionally produces non-differentiable integer
indices.
