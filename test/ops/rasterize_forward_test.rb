# frozen_string_literal: true

require "test_helper"

class RasterizeForwardTest < Minitest::Test
  %w[raster_features8 raster_features40].each do |fixture_name|
    define_method("test_matches_python_#{fixture_name}") do
      fixture = golden(fixture_name)
      rendered, alphas = Gsplat.rasterize_to_pixels(
        fixture.fetch("means2d"),
        fixture.fetch("conics"),
        fixture.fetch("colors"),
        fixture.fetch("opacities"),
        64,
        48,
        16,
        fixture.fetch("isect_offsets"),
        fixture.fetch("flatten_ids"),
        backgrounds: fixture.fetch("backgrounds")
      )

      assert_allclose rendered, fixture.fetch("render_colors"), atol: 1e-4, rtol: 1e-4
      assert_allclose alphas, fixture.fetch("render_alphas"), atol: 1e-4, rtol: 1e-4
    end
  end

  def setup
    @previous_backend = Gsplat.backend
    Gsplat.backend = :ruby
  end

  def teardown
    Gsplat.backend = @previous_backend
  end

  def test_single_gaussian_at_center_pixel
    means2d = Numo::DFloat[[[0.5, 0.5]]]
    conics = Numo::DFloat[[[1, 0, 1]]]
    colors = Numo::DFloat[[[0.2, 0.4, 0.6]]]
    opacities = Numo::DFloat[[0.8]]
    offsets = Numo::Int32[[[0]]]
    flatten_ids = Numo::Int32[0]

    rendered, alphas = Gsplat.rasterize_to_pixels(
      means2d,
      conics,
      colors,
      opacities,
      1,
      1,
      16,
      offsets,
      flatten_ids
    )

    assert_allclose rendered, Numo::DFloat[[[[0.16, 0.32, 0.48]]]], atol: 1e-12, rtol: 0.0
    assert_allclose alphas, Numo::DFloat[[[[0.8]]]], atol: 1e-12, rtol: 0.0
  end

  def test_matches_brute_force_accumulate_across_partial_tiles
    width = 20
    height = 18
    tile_size = 16
    means2d = Numo::DFloat[[[5.5, 6.5], [12.5, 10.5], [17.5, 15.5]]]
    conics = Numo::DFloat[
      [[0.08, 0.01, 0.06], [0.05, -0.005, 0.07], [0.09, 0.0, 0.09]]
    ]
    colors = Numo::DFloat[[[0.2, 0.4], [0.8, 0.1], [0.3, 0.9]]]
    opacities = Numo::DFloat[[0.7, 0.5, 0.4]]
    backgrounds = Numo::DFloat[[0.1, 0.25]]
    radii = Numo::Int32[[100, 100, 100]]
    depths = Numo::DFloat[[1, 2, 3]]
    tile_width = (width.to_f / tile_size).ceil
    tile_height = (height.to_f / tile_size).ceil
    _, keys, flatten_ids = Gsplat.isect_tiles(
      means2d, radii, depths, tile_size, tile_width, tile_height
    )
    offsets = Gsplat.isect_offset_encode(keys, 1, tile_width, tile_height)

    tiled_colors, tiled_alphas = Gsplat.rasterize_to_pixels(
      means2d,
      conics,
      colors,
      opacities,
      width,
      height,
      tile_size,
      offsets,
      flatten_ids,
      backgrounds: backgrounds
    )
    reference_colors, reference_alphas = Gsplat.accumulate(
      means2d,
      conics,
      opacities,
      colors,
      width: width,
      height: height,
      backgrounds: backgrounds
    )

    assert_allclose tiled_colors, reference_colors, atol: 1e-12, rtol: 0.0
    assert_allclose tiled_alphas, reference_alphas, atol: 1e-12, rtol: 0.0
  end

  def test_tile_mask_returns_background
    rendered, alphas = Gsplat.rasterize_to_pixels(
      Numo::SFloat[[[0.5, 0.5]]],
      Numo::SFloat[[[1, 0, 1]]],
      Numo::SFloat[[[1, 0]]],
      Numo::SFloat[[0.8]],
      1,
      1,
      16,
      Numo::Int32[[[0]]],
      Numo::Int32[0],
      backgrounds: Numo::SFloat[[0.2, 0.3]],
      masks: Numo::Bit[[[0]]]
    )

    assert_allclose rendered, Numo::SFloat[[[[0.2, 0.3]]]], atol: 0.0, rtol: 0.0
    assert_allclose alphas, Numo::SFloat.zeros(1, 1, 1, 1), atol: 0.0, rtol: 0.0
  end

  def test_matches_python_raster_golden_data
    fixture = golden("raster_rgb")
    rendered, alphas = Gsplat.rasterize_to_pixels(
      fixture.fetch("means2d"),
      fixture.fetch("conics"),
      fixture.fetch("colors"),
      fixture.fetch("opacities"),
      64,
      48,
      16,
      fixture.fetch("isect_offsets"),
      fixture.fetch("flatten_ids"),
      backgrounds: fixture.fetch("backgrounds")
    )

    assert_allclose rendered, fixture.fetch("render_colors"), atol: 1e-4, rtol: 1e-4
    assert_allclose alphas, fixture.fetch("render_alphas"), atol: 1e-4, rtol: 1e-4
  end
end
