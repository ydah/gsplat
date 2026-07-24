# 0004: Use one portable reference for world-space extension paths

- Status: Accepted
- Date: 2026-07-24

## Context

Hit-distance rendering and local depth ordering require anisotropic Gaussian evaluation in world
space. They extend the renderer beyond the performance-oriented 2D EWA core and introduce geometry
derivatives that would be costly to duplicate across backends.

## Decision

- Evaluate hit-distance modes against the anisotropic Gaussian in world space and replace the final
  feature channel with the per-pixel closest-approach distance, matching the upstream eval3d
  definition.
- Use the same Ruby implementation for both backend selections and a dtype-aware central-difference
  geometry VJP. Keep the core 2D raster path analytic and native-accelerated.
- Make `global_z_order: false` sort projected centers by camera-space Euclidean distance. Continue
  near/far culling by camera z and propagate the Euclidean-norm VJP to means.

## Consequences

The extension remains differentiable and behaviorally identical under `:ruby` and `:native`
without maintaining two world-space evaluators. It prioritizes semantic coverage over large-scene
speed and leaves the established fast raster path untouched.
