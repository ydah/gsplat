# 0006: Keep eval3d as a portable reference path

- Status: Accepted
- Date: 2026-07-24

## Context

`with_eval3d` generalizes the world-space evaluator from
[ADR 0004](0004-use-portable-world-space-reference-paths.md) to arbitrary color features, and
`return_normals` adds view-oriented Gaussian normals. These semantics are more important than
duplicating them in the native raster kernels before CUDA parity is available.

## Decision

- Accumulate arbitrary features with the world-space evaluator.
- For normals, use the quaternion's canonical +Z axis, flip it toward the ray, and accumulate it
  with the same alpha weights as color.
- Support pinhole cameras and classic rasterization.
- Share the Ruby evaluator and numerical geometry VJP between backend selections.

## Consequences

The performance-oriented 2D raster kernels remain unchanged, while eval3d values and geometry stay
differentiable and portable. The analytic single-ray case validates values and quaternion
gradients locally; full CUDA color, alpha, and normal parity remains a generated-golden gate.
