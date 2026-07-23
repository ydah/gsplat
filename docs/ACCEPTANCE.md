# Acceptance status

This report maps the design goals G1–G6 to reproducible evidence as of 2026-07-24. “Partial” means
the implementation and local tests exist but the exact external dataset/hardware gate has not run.

| Goal | Status | Evidence |
|---|---|---|
| G1: Python forward parity | Partial | 47-case pinned generator; analytic forward cases; reference compositor and backend-equivalence tests. CUDA-generated NPZ files are unavailable on this host, so their tests are documented skips. |
| G2: backward parity | Partial | Float64 central differences for covariance, SH, projection, rasterization, distortion, losses, and eval3d; tile-vs-reference VJPs. Python/CUDA gradient fixtures remain pending with G1. |
| G3: single-image convergence | Pass | `ImageFitterTest#test_self_consistency_fit_improves_psnr`: 11.772 → 30.878 dB in 20 monotonic reduced steps. The 50k/512²/2,000-step native timing also completes in 157.457 s. |
| G4: COLMAP training quality | Partial | Binary/text parser parity, 100-point fixture, actual fixture-based Trainer smoke path, and an eight-view synthetic Trainer reaching ≥28 dB in 80 steps. No Mip-NeRF 360/real-capture 30k quality run is available. |
| G5: compatible API surface | Pass within v1 scope | README mapping, migration guide, executable quick start, YARD 100%, low-level wrappers, strategies, optimizer, Trainer, IO, compression, 2DGS and eval3d surfaces. Unsupported upstream options are explicitly listed. |
| G6: Ruby-only completeness | Pass | Full suite passes with `GSPLAT_BACKEND=ruby`; the same full suite passes with explicit native selection. CI now builds the extension and runs both paths, while auto mode retains the Ruby fallback. |

## Local verification

```text
Ruby selection:   188 tests, 1,041 assertions, 0 failures, 22 documented skips
Native selection: 188 tests, 1,041 assertions, 0 failures, 22 documented skips
YARD:             129 methods, 125 attributes, 22 constants, undocumented 0
RuboCop:          132 files, no offenses
```

## Golden-data gate

`tools/generate_golden.py` is pinned to Python gsplat 1.5.3, uses seed 42, records provenance, and
caps compressed output at 50 MiB. The local host has no CUDA and a prior temporary PyTorch install
exceeded available disk. Consequently, tests calling `golden(...)` skip with a pointer to
`tools/README.md`; no Ruby-generated fixture is accepted as a replacement.

Run this on a compatible CUDA host to close G1/G2:

```bash
python3 -m venv .venv-golden
.venv-golden/bin/pip install -r tools/requirements.txt
.venv-golden/bin/python tools/generate_golden.py --device cuda --require-all
bundle exec rake test
```

## CI matrix

`.github/workflows/main.yml` runs:

- RuboCop and the 100% YARD gate on Ruby 4.0.
- The full Ruby backend suite on CRuby 3.2, 3.3, 3.4, and 4.0.
- A native extension build and the full native suite on CRuby 4.0 with OpenMP workers.
- Focused image-fit, multi-view Trainer, and all-example smoke tests.

The same workflow runs on pushes, pull requests, manual dispatch, and a weekly scheduled trigger.
Local equivalents pass; hosted status can only be confirmed after pushing the commit.

## Release interpretation

The package is functionally releasable for the documented CPU/dense scope. Strict completion of the
original cross-implementation and real-capture goals still requires:

1. Generating and committing the CUDA golden set, then resolving any mismatches.
2. Running a representative real COLMAP capture for 30,000 steps and recording quality/time.

These are external validation gates, not silently treated as passed.
