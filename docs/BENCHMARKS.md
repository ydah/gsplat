# Benchmark results

## Environment

- Date: 2026-07-24
- Ruby: CRuby 4.0.0
- Platform: arm64-darwin24 (Apple Silicon)
- Native workers: `OMP_NUM_THREADS=8`
- Dtype: `Numo::SFloat`
- Scene seed: 42

`benchmarks/bench_rasterize.rb` contains all four workloads from design section 9.1. Use `--list`
to inspect them and `--quick` for CI-sized smoke runs.

## Raster forward and backward

Commands:

```bash
bundle exec ruby -Ilib benchmarks/bench_rasterize.rb \
  --scenario raster --backend ruby --iterations 1

OMP_NUM_THREADS=8 bundle exec ruby -Ilib benchmarks/bench_rasterize.rb \
  --scenario raster --backend native --iterations 3
```

Workload: 100,000 Gaussians, one 800×800 camera, RGB, float32. Each measurement follows one warm-up.
Forward+backward includes a fresh forward, both color/alpha VJPs, and `absgrad`.

| Backend | Forward | Target | Forward + backward | Target | Result |
|---|---:|---:|---:|---:|---|
| Ruby | 3,527.990 ms | ≤ 10,000 ms | 8,047.024 ms | ≤ 30,000 ms | Pass |
| Native, 8 workers | 30.419 ms | ≤ 150 ms | 112.573 ms | ≤ 400 ms | Pass |

Native samples over three measured calls were 28.901–32.861 ms forward and
109.907–114.240 ms combined. The speedup over Ruby was approximately 116× forward and 71× combined
for this scene.

## Image fitting

Command:

```bash
OMP_NUM_THREADS=8 bundle exec ruby -Ilib benchmarks/bench_rasterize.rb \
  --scenario fit_image --backend native
```

Workload: 50,000 Gaussians, 512×512 image, 2,000 optimization steps, float32. The timed run completed
in 157.457 seconds (2 minutes 37.5 seconds), passing the ≤30 minute design target.

This timing workload optimizes only colors in a deterministic self-consistency scene. Its final
PSNR was 16.799 dB; that value is not the convergence acceptance case. The reduced convergence test
uses 64 Gaussians at 8×8 and reaches 30.878 dB in 20 monotonic steps.

## COLMAP training

The benchmark scenario loads a real COLMAP directory and runs the production `Training::Trainer`:

```bash
ruby -Ilib benchmarks/bench_rasterize.rb \
  --scenario colmap --backend native --data /path/to/capture --steps 30000
```

The complete 30,000-step, several-hundred-thousand-Gaussian benchmark was not run because no real
capture with reference images is present on this host. The execution path was smoke-tested with the
three-camera/100-point COLMAP fixture plus temporary 640×480 PNGs: one native step, including full
evaluation before and after training, completed in 2.584 seconds.

This smoke result demonstrates that the scenario and image/point pipeline execute; it is not a
substitute for the design's overnight wall-time or real-capture quality gate.

## Reproducing and extending

Override workload sizes with `--gaussians`, `--width`, `--height`, `--steps`, and `--iterations`.
Use `--json PATH` to save the complete environment/result object. The benchmark rejects zero or
negative dimensions and never changes the mathematical rasterization thresholds.

The older operation-attribution and OpenMP scaling measurements remain in
[`PROFILE.md`](PROFILE.md).
