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
