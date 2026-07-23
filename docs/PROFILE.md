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
