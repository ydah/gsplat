# 0005: Reuse the differentiable EWA core for 2DGS

- Status: Accepted
- Date: 2026-07-24

## Context

The Ruby API needs a portable 2DGS-compatible surface and Trainer mode, but an exact upstream
ray-splat transform and median-crossing compositor cannot be validated on this CPU-only host.

## Decision

- Preserve the upstream seven-value return structure and metadata names.
- Reuse the established differentiable EWA footprint and compositor for color and alpha.
- Derive oriented camera normals, normalized surface normals, a depth-opacity distortion signal,
  and expected depth as the median approximation.
- Treat exact ray-splat transforms, median crossing, and distortion accumulation as part of the
  pending CUDA golden gate rather than claiming numerical identity.

## Consequences

2DGS color/alpha rendering and Trainer optimization remain differentiable on both backends with a
stable public shape. Auxiliary geometry buffers are approximate and must not be presented as an
exact substitute for upstream 2DGS until the external parity gate can be completed.
