# gsplat

`gsplat` is a CRuby implementation of differentiable 3D Gaussian Splatting. It provides a portable
Numo::NArray backend, a faster optional C/OpenMP backend, a small reverse-mode autograd engine, and
training/IO utilities with an API modeled after Python
[`gsplat`](https://github.com/nerfstudio-project/gsplat).

The package currently renders on CPU. It is useful for Ruby-native pipelines, reference testing,
small training jobs, and inspecting the equations without a PyTorch/CUDA runtime.

## Requirements

- CRuby 3.2 or newer
- A C compiler for `numo-narray`
- Optional: OpenMP for the native parallel kernels
- Optional image IO: `ruby-vips` (recommended) or `chunky_png`

JRuby and TruffleRuby are not release targets. The Ruby implementation itself avoids C-specific
state where practical, but the packaged native extension targets CRuby.

## Installation

From a checkout:

```bash
bundle install
bundle exec rake compile
```

In another Bundler project, point at the checkout until the gem is published:

```ruby
gem "gsplat", path: "/path/to/gsplat"
```

The native extension is optional at runtime. If it cannot be loaded, `Gsplat.backend = :auto`
selects the Ruby implementation and emits one warning. Select a backend explicitly with:

```ruby
Gsplat.backend = :ruby
Gsplat.backend = :native
```

or with `GSPLAT_BACKEND=ruby|native|auto`.

## Quick start

This complete one-Gaussian render is extracted and executed by `test/readme_test.rb`.

<!-- quickstart:start -->
```ruby
require "gsplat"

f = Numo::SFloat
colors = Gsplat::Autograd::Variable.new(f[[1.0, 0.2, 0.1]], requires_grad: true)

rendered, alphas, meta = Gsplat.rasterization(
  means: f[[0.0, 0.0, 2.0]],
  quats: f[[1.0, 0.0, 0.0, 0.0]], # wxyz
  scales: f[[0.25, 0.25, 0.25]],
  opacities: f[0.8],               # activated opacity
  colors: colors,
  viewmats: f.eye(4).reshape(1, 4, 4),
  ks: f[[[8.0, 0.0, 2.0], [0.0, 8.0, 2.0], [0.0, 0.0, 1.0]]],
  width: 4,
  height: 4
)

rendered.backward(f.ones(*rendered.data.shape))
p [rendered.data.shape, alphas.data.shape, meta.fetch(:radii).shape, colors.grad.shape]
```
<!-- quickstart:end -->

The output shapes are:

```text
[[1, 4, 4, 3], [1, 4, 4, 1], [1, 1, 2], [1, 3]]
```

Use `Numo::SFloat` for the native fast path. `Numo::DFloat` is supported and is useful for gradient
checks, but currently falls back to the Ruby formulas.

## Conventions

- Quaternions are `wxyz` and are normalized when converted to rotations.
- `viewmats [C,4,4]` transform world coordinates into camera coordinates.
- Camera coordinates look along positive Z. Pixel samples are evaluated at `(x + 0.5, y + 0.5)`.
- `scales [N,3]` and `opacities [N]` passed to rendering are activated, positive/0–1 values.
  `Training::Trainer` stores their logarithm/logit internally and activates them before rendering.
- Plain colors are `[N,D]` or `[C,N,D]`. With `sh_degree`, coefficients are `[N,K,D]` with
  `K >= (sh_degree + 1)^2`.
- Dense render outputs are `[C,H,W,D]`; alpha is `[C,H,W,1]`; elliptical radii are `[C,N,2]`.
- Supplying an `Autograd::Variable` records a graph. Call `backward` with an explicit gradient for
  non-scalar outputs, and `zero_grad!` before reusing a leaf.

## Python API mapping

| Python gsplat | Ruby |
|---|---|
| `gsplat.rasterization(...)` | `Gsplat.rasterization(...)` |
| `gsplat.rasterization_2dgs(...)` | `Gsplat.rasterization_2dgs(...)` |
| `gsplat.spherical_harmonics(...)` | `Gsplat.spherical_harmonics(...)` |
| `gsplat.quat_scale_to_covar_preci(...)` | `Gsplat.quat_scale_to_covar_preci(...)` |
| `gsplat.fully_fused_projection(...)` | `Gsplat.fully_fused_projection(...)` |
| `gsplat.isect_tiles(...)` | `Gsplat.isect_tiles(...)` |
| `gsplat.isect_offset_encode(...)` | `Gsplat.isect_offset_encode(...)` |
| `gsplat.rasterize_to_pixels(...)` | `Gsplat.rasterize_to_pixels(...)` |
| `gsplat.rasterize_to_indices_in_range(...)` | `Gsplat.rasterize_to_indices_in_range(...)` |
| `gsplat.strategy.DefaultStrategy` | `Gsplat::Strategy::Default` |
| `gsplat.strategy.MCMCStrategy` | `Gsplat::Strategy::MCMC` |
| `gsplat.compression.PngCompression` | `Gsplat::Compression::Png` |
| `torch.optim.Adam` | `Gsplat::Optim::Adam` |
| `examples/simple_trainer.py` | `Gsplat::Training::Trainer` / `examples/simple_trainer.rb` |

See [docs/MIGRATION.md](docs/MIGRATION.md) for shape, autograd, activation, and training differences.

## Feature status

| Area | Status |
|---|---|
| Dense pinhole projection/rasterization and analytic gradients | Supported |
| Orthographic and equidistant fisheye cameras | Supported |
| OpenCV radial, tangential, and thin-prism distortion | Supported |
| RGB, arbitrary features, SH degree 0–4, D/ED and combined modes | Supported |
| Hit distance modes and `with_eval3d` accumulated normals | Supported; pinhole/classic, reference-speed VJP |
| Antialiased rendering, `radius_clip`, direct covariance input | Supported |
| `absgrad`, Default/MCMC densification, Adam/SelectiveAdam | Supported |
| COLMAP bin/txt, Inria PLY, NPZ checkpoints, PNG compression | Supported |
| 2DGS API and Trainer mode | Available; see limitation below |
| Native float32 projection, SH, intersections, raster fwd/bwd | Supported with Ruby fallback |
| `packed` / sparse gradients | Not implemented; dense mode is used |
| Distributed rendering | Not implemented |
| F-theta, lidar, rolling shutter, custom rays | Not implemented |
| Unscented-transform projection and renderer configs | Not implemented |
| Extra-signal API and LPIPS | Not implemented |
| CUDA/GPU backend | Not implemented |

The portable 2DGS implementation preserves the seven-output API and differentiable color/alpha path,
but currently derives normal/distortion/median buffers from the established EWA footprint. It is not
an exact replacement for the upstream ray-splat transform.

The eval3d path shares one Ruby implementation between backend selections and uses numerical geometry
VJPs. It prioritizes semantic coverage over large-scene speed.

## Image fitting example

Run the deterministic self-consistency fit:

```bash
bundle exec ruby examples/fit_image.rb --gaussians 2000 --steps 300
```

## Training a COLMAP capture

Prepare `sparse/0/{cameras,images,points3D}.bin` plus `images/`. For a downsample factor greater than
one, place resized images in `images_N/`; dimensions and intrinsics are scaled together.

```bash
gem install ruby-vips # or: gem install chunky_png
bundle exec ruby examples/simple_trainer.rb \
  --data /path/to/capture \
  --output results/capture \
  --steps 30000 \
  --data-factor 1
```

Use `--strategy mcmc` for relocation-based densification. The trainer writes NPZ checkpoints and the
example exports an Inria-compatible `splats.ply`.

Render an orbit from that PLY:

```bash
bundle exec ruby examples/render_path.rb \
  --ply results/capture/splats.ply \
  --output results/capture/path \
  --frames 120
```

`ruby-vips` is selected before `chunky_png`. The latter is a portable PNG-only fallback.

## Validation and development

```bash
bundle exec rake test
OMP_NUM_THREADS=8 GSPLAT_BACKEND=native bundle exec rake test
bundle exec rubocop --no-server --cache false
bundle exec yard stats
```

Golden-data generation is pinned to Python gsplat 1.5.3. This repository's CPU-only development host
cannot generate CUDA raster fixtures; the generator, exact commands, and documented skips are in
[tools/README.md](tools/README.md). Implementation differences are recorded in
[docs/DECISIONS.md](docs/DECISIONS.md), and phase evidence is tracked in
[docs/PROGRESS.md](docs/PROGRESS.md). See [docs/BENCHMARKS.md](docs/BENCHMARKS.md) for measured
performance and [docs/ACCEPTANCE.md](docs/ACCEPTANCE.md) for the G1–G6 status.

## License

Gsplat is available under the [Apache License 2.0](LICENSE.txt).
