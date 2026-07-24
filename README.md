# Gsplat

[![CI](https://github.com/ydah/gsplat/actions/workflows/main.yml/badge.svg)](https://github.com/ydah/gsplat/actions/workflows/main.yml)
[![License: Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE.txt)

Gsplat is a differentiable 3D Gaussian Splatting renderer and trainer for CRuby. It provides a
portable `Numo::NArray` implementation, an optional C/OpenMP fast path, reverse-mode automatic
differentiation, and training and IO utilities modeled after Python
[`gsplat`](https://github.com/nerfstudio-project/gsplat).

Gsplat currently runs on CPU. It is intended for Ruby-native graphics pipelines, reference
implementations, inspection of the rendering equations, and small training jobs.

## Features

- Differentiable dense 3D Gaussian projection and tiled alpha compositing.
- 3DGS and 2DGS APIs with RGB, arbitrary features, spherical harmonics, depth, and normals.
- Pinhole, orthographic, equidistant fisheye, and OpenCV-distorted cameras.
- Adam, SelectiveAdam, Default densification, MCMC strategy, and multi-view training.
- COLMAP, Inria PLY, NPY/NPZ checkpoints, image IO, and PNG parameter compression.
- Portable Ruby backend and native float32/OpenMP kernels with automatic fallback.

## Installation

Add Gsplat to your application's `Gemfile`:

```ruby
gem "gsplat"
```

Then install dependencies:

```bash
bundle install
```

Or install the gem directly:

```bash
gem install gsplat
```

### Requirements

- CRuby 3.2 or newer.
- A C compiler for `numo-narray` and the packaged native extension.
- OpenMP is optional and enables parallel native raster kernels.
- Image loading and writing require either `ruby-vips` (recommended) or `chunky_png`.

The source checkout includes `chunky_png` for its runnable examples. JRuby and TruffleRuby are not
release targets.

### Backend selection

The default `:auto` backend uses the native extension when it is available and otherwise falls back
to Ruby with a one-time warning.

```ruby
Gsplat.backend = :auto
Gsplat.backend = :ruby
Gsplat.backend = :native
```

The same selection can be made with `GSPLAT_BACKEND=auto|ruby|native`. Use `Numo::SFloat` for native
kernels. `Numo::DFloat` is supported for numerical checks and uses the Ruby formulas where needed.

## Usage

### Quick start

The following complete example renders one Gaussian and differentiates the image with respect to
its color.

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

Output:

```text
[[1, 4, 4, 3], [1, 4, 4, 1], [1, 1, 2], [1, 3]]
```

### Data conventions

- Quaternions use `wxyz` order and are normalized when converted to rotations.
- `viewmats [C,4,4]` transform world coordinates into camera coordinates.
- Cameras look along positive Z; pixels are sampled at `(x + 0.5, y + 0.5)`.
- Rendering receives activated positive scales and 0–1 opacities.
- Colors are `[N,D]` or `[C,N,D]`. Spherical harmonic coefficients are `[N,K,D]`.
- Dense color outputs are `[C,H,W,D]`, alpha is `[C,H,W,1]`, and radii are `[C,N,2]`.
- An `Autograd::Variable` records a graph. Non-scalar outputs require an explicit backward gradient.

### Image fitting

From a source checkout, run the deterministic image-fitting example:

```bash
bundle exec ruby examples/fit_image.rb --gaussians 2000 --steps 300
```

### Runnable sample data

The repository includes a tiny synthetic COLMAP dataset and its matching Inria PLY, so the
multi-view examples run without downloading external data or supplying arguments:

```bash
bundle exec ruby examples/simple_trainer.rb
bundle exec ruby examples/render_path.rb
```

The trainer uses 10 steps and writes `results/sample/splats.ply`. The renderer creates 12 images in
`renders/sample/` from the bundled PLY. To render the newly trained PLY instead, pass
`--ply results/sample/splats.ply`. Rebuild the checked-in data at any time with:

```bash
bundle exec ruby examples/generate_sample_data.rb
```

### Training a COLMAP capture

Prepare `sparse/0/{cameras,images,points3D}.bin` and an `images/` directory, then run:

```bash
bundle exec ruby examples/simple_trainer.rb \
  --data /path/to/capture \
  --output results/capture \
  --steps 30000 \
  --data-factor 1
```

Use `--strategy mcmc` for relocation-based densification. The trainer writes NPZ checkpoints and an
Inria-compatible `splats.ply`. Render an orbit from that PLY with:

```bash
bundle exec ruby examples/render_path.rb \
  --ply results/capture/splats.ply \
  --output results/capture/path \
  --frames 120
```

## API compatibility

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

See [the migration guide](docs/MIGRATION.md) for differences in shapes, activation, autograd, and
training.

## Limitations

- Rendering and training are CPU-only; CUDA/GPU execution is not implemented.
- Rendering is dense. Packed/sparse gradients and distributed rendering are not implemented.
- 2DGS auxiliary geometry is API-compatible but approximates the upstream ray-splat transform.
- Eval3d uses a shared Ruby reference implementation and a numerical geometry VJP.
- Complete upstream raster parity still requires the documented external CUDA golden-data run.

## Documentation

- [Python migration guide](docs/MIGRATION.md)
- [Architecture Decision Records](docs/DECISIONS.md)
- [Benchmarks](docs/BENCHMARKS.md)
- [Acceptance status](docs/ACCEPTANCE.md)
- [Implementation progress](docs/PROGRESS.md)
- [Golden-data tooling](tools/README.md)

Generate the API reference locally with `bundle exec yard doc`.

## Development

Clone the repository and install the development dependencies:

```bash
git clone https://github.com/ydah/gsplat.git
cd gsplat
bundle install
bundle exec rake compile
```

Run the validation suite:

```bash
GSPLAT_BACKEND=ruby bundle exec rake test
OMP_NUM_THREADS=8 GSPLAT_BACKEND=native bundle exec rake test
bundle exec rubocop --no-server --cache false
bundle exec yard stats
gem build gsplat.gemspec
```

Golden-data generation is pinned to Python gsplat 1.5.3. See
[`tools/README.md`](tools/README.md) for CPU and CUDA commands.

## Contributing

Bug reports and pull requests are welcome on
[GitHub](https://github.com/ydah/gsplat). Keep behavior changes covered by tests. Changes that
establish or replace a long-lived architectural contract should include an
[ADR](docs/DECISIONS.md) created from the repository template.

## License

Gsplat is available under the [Apache License 2.0](LICENSE.txt).
