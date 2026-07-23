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

## Native rasterization — 2026-07-23

The float32 forward and backward tile kernels release the GVL. Forward owns disjoint pixel tiles.
Backward uses atomic scatter-add under OpenMP; this keeps memory at `O(N × D)` rather than allocating
one full gradient buffer per worker. On the 1,000-Gaussian, 64×48 profile:

| Operation | Ruby | Native (1 thread) | Speedup |
|---|---:|---:|---:|
| Raster forward | 55.821 ms | 2.284 ms | 24.4× |
| Raster backward + absgrad | 136.854 ms | 2.700 ms | 50.7× |

The required 100,000-Gaussian, 800×800 float32 workload gives:

| OpenMP threads | Forward | Backward + absgrad | Combined |
|---:|---:|---:|---:|
| 1 | 114.012 ms | 279.790 ms | 393.802 ms |
| 2 | 57.751 ms | 148.111 ms | 205.862 ms |
| 4 | 30.446 ms | 83.138 ms | 113.584 ms |
| 8 | 29.643 ms | 79.252 ms | 108.895 ms |

This meets the 150 ms forward and 400 ms combined design targets even at one thread. The measured
3.5× backward scaling from one to eight workers supports the bounded-memory atomic strategy. Scaling
flattens after four workers on this Apple Silicon host, so higher thread counts are not assumed to
improve throughput. Homebrew `libomp` is auto-detected on Apple Clang; other builds remain functional
without OpenMP.
