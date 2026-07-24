# 0002: Retain exact Ruby fallbacks for the native backend

- Status: Accepted
- Date: 2026-07-23

## Context

The native backend must accelerate production float32 training without creating a second,
divergent implementation of every derivative and compatibility path. Raster backward must also
remain within a practical memory bound at the 100k-Gaussian benchmark scale.

## Decision

- Target contiguous `Numo::SFloat` arrays in native kernels.
- Keep float64 gradchecks, masked spherical harmonics, and analytic projection/SH backward on the
  Ruby implementation. Selecting `:native` remains functionally complete through these fallbacks.
- Label performance results for these paths as hybrid and measure accelerated operations
  explicitly.
- Use OpenMP atomic scatter-add in raster backward instead of per-worker full gradient buffers.

## Consequences

The native backend accelerates the high-volume path while preserving one authoritative copy of
complex derivative formulas. Atomic scatter-add avoids `O(workers × N × channels)` temporary
memory. Its measured 100k/800×800 raster backward improves from 279.790 ms with one worker to
79.252 ms with eight workers, at the cost of atomic-update contention.
