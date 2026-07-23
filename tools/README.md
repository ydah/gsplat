# Golden data tools

The reference stack is pinned to gsplat 1.5.3. Generate data only from the Python implementation;
never regenerate golden files from Ruby output.

## Inspect the case matrix without dependencies

```bash
python3 tools/generate_golden.py --dry-run
```

## CPU setup

The upstream source distribution normally attempts to build its CUDA extension. Disable that step
for a CPU-only environment:

```bash
python3 -m venv .venv-golden
BUILD_NO_CUDA=1 .venv-golden/bin/pip install -r tools/requirements.txt
.venv-golden/bin/python tools/generate_golden.py --device cpu
```

CPU generation covers covariance, SH, projection, intersections, DefaultStrategy masks, and SSIM.
Rasterization, end-to-end rendering, AbsGrad, and MCMC relocation use upstream CUDA kernels and are
recorded as skipped in `test/golden/manifest.json`.

The Euclidean-order and single-ray hit-distance fixtures are analytic CPU cases. Their equations
mirror current upstream eval3d semantics while keeping the primary package pin at 1.5.3. The full
world-space color/alpha/normal fixture requires CUDA.

## Complete CUDA generation

On a machine with a CUDA toolkit compatible with the pinned PyTorch:

```bash
python3 -m venv .venv-golden
.venv-golden/bin/pip install -r tools/requirements.txt
.venv-golden/bin/python tools/generate_golden.py --device cuda --require-all
```

The script uses seed 42, writes one compressed NPZ per case, records provenance in `manifest.json`,
and fails if the output exceeds 50 MiB. Commit the resulting `test/golden/` directory unchanged.
