# Performance Profile

## Ruby baseline — 2026-07-23

Command:

```bash
bundle exec ruby -Ilib benchmarks/profile_ruby.rb
```

Environment: CRuby 4.0.0, arm64-darwin24. Deterministic scene: 1,000 Gaussians, one 64×48
camera, SH degree 3, five measured iterations after one warm-up.

| Operation | Mean time |
|---|---:|
| Projection | 1.010 ms |
| Spherical harmonics | 0.393 ms |
| Tile intersections and sort | 2.505 ms |
| Rasterization forward | 58.350 ms |

Rasterization accounts for about 93.7% of the summed operation time. Tile intersection generation
is the second-largest component. This satisfies the P10 profiling prerequisite and fixes the native
implementation order: bridge first, projection/SH for coverage, intersection sort, then the
dominant raster forward/backward path. These numbers are a small-scene attribution profile, not
the final P12 throughput benchmark.

## Native projection and SH — 2026-07-23

With the P10-2 extension loaded (`GSPLAT_BACKEND=auto`) on the same workload:

| Operation | Ruby | Native/auto | Speedup |
|---|---:|---:|---:|
| Projection | 1.010 ms | 0.257 ms | 3.9× |
| Spherical harmonics | 0.393 ms | 0.043 ms | 9.1× |

The C paths cover contiguous float32 forward calls. Float64, masked SH, and analytic backward use
the Ruby implementation to preserve the existing numerical behavior.

## Native intersections — 2026-07-23

The P10-3 kernel adds float32 tile-bound enumeration, 64-bit key generation, stable LSD radix sort,
and tile-offset encoding. It retains the same key layout and ordering as the Ruby implementation.

| Operation | Ruby | Native/auto | Speedup |
|---|---:|---:|---:|
| Tile intersections and sort | 2.505 ms | 0.045 ms | 55.7× |

The measurement uses the same 1,000-Gaussian workload as the baseline. Float64 inputs retain the
Ruby path so numerical gradient tests continue to exercise the reference implementation.
