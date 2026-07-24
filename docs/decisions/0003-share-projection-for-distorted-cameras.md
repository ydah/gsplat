# 0003: Share the Ruby projection for distorted cameras

- Status: Accepted
- Date: 2026-07-23

## Context

Equidistant fisheye and OpenCV distortion add calibration-sensitive formulas and numerical
derivatives. Independent Ruby and native implementations would increase the risk of camera-model
drift without improving the established pinhole performance path.

## Decision

- Use one float32/float64 Ruby implementation of equidistant fisheye and OpenCV
  radial/tangential/thin-prism projection for both backend selections.
- Delegate these camera models to the Ruby path when `:native` is selected.
- Follow gsplat 1.5.3's rational pinhole coefficient order: `k1..k3` form the numerator and
  `k4..k6` form the denominator. Fisheye coefficients multiply `theta^3..theta^9`.
- Treat distortion coefficients as calibration constants. Propagate gradients to Gaussian
  geometry, but do not optimize coefficients or intrinsics in this phase.

## Consequences

Both backend selections have identical distorted-camera semantics and numerical derivatives. These
models run at reference speed; the optimized pinhole path and its analytic/native backward remain
unchanged.
