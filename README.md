# gsplat

`gsplat` is a Ruby implementation of a differentiable 3D Gaussian Splatting rasterizer. It uses
Numo::NArray for a portable reference backend and can optionally use a native C extension for
performance.

The implementation follows the project design in
[`.idea/gsplat-ruby-設計書.md`](.idea/gsplat-ruby-設計書.md) and the phased acceptance criteria in
[`.idea/gsplat-ruby-作業指示書.md`](.idea/gsplat-ruby-作業指示書.md).

## Requirements

- CRuby 3.2 or newer
- `numo-narray`

## Supported rendering features

- Differentiable dense 3D Gaussian projection and tiled alpha compositing
- Arbitrary feature dimensions with configurable `channel_chunk`
- Pinhole, orthographic and equidistant fisheye cameras with OpenCV distortion
- 2D Gaussian surfel rendering with normal, distortion and median-depth outputs
- World-space Gaussian evaluation with differentiable accumulated normals
- Contribution-index enumeration for iterative analysis and custom compositing
- Direct covariance or quaternion/scale geometry
- RGB, z/Euclidean depth, per-ray hit distance, expected-depth and combined render modes
- Classic and antialiased rasterization with `radius_clip`
- Dependency-free PNG/K-means parameter compression
- Adam and visibility-masked SelectiveAdam optimizers
- Ruby reference and optional OpenMP native backends

Image IO is optional. Install `ruby-vips` for best performance or `chunky_png` as a portable
fallback once the IO layer is enabled.

## Installation

Add the gem from this repository while it is under development:

```ruby
gem "gsplat", path: "/path/to/gsplat"
```

Then run:

```bash
bundle install
```

## Development

```bash
bundle install
bundle exec rake test
bundle exec rubocop
```

The public rendering API and examples will be documented as their implementation phases land.

## Image fitting example

Run the self-consistency image fit with:

```bash
bundle exec ruby examples/fit_image.rb --gaussians 2000 --steps 300
```

The example constructs a deterministic smooth target using the same Gaussian geometry, then
optimizes its colors through the complete differentiable rendering pipeline and reports PSNR.

## Training a COLMAP capture

Prepare a standard COLMAP project with `sparse/0/{cameras,images,points3D}.bin` and an `images/`
directory. For a downsample factor greater than one, place correspondingly resized images in
`images_N/`; camera dimensions and intrinsics are scaled automatically.

```bash
gem install ruby-vips # or: gem install chunky_png
bundle exec ruby examples/simple_trainer.rb \
  --data /path/to/capture \
  --output results/capture \
  --steps 30000 \
  --data-factor 1
```

Use `--strategy mcmc` for the MCMC relocation strategy. The trainer writes an NPZ checkpoint and
an Inria-compatible `splats.ply` at the configured final step.

Render an orbit from the exported PLY with:

```bash
bundle exec ruby examples/render_path.rb \
  --ply results/capture/splats.ply \
  --output results/capture/path \
  --frames 120
```

The image layer selects `ruby-vips` first and falls back to `chunky_png`. The latter supports the
portable PNG path; `ruby-vips` is recommended for dataset training.
