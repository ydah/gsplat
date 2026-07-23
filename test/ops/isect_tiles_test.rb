# frozen_string_literal: true

require "test_helper"

class IsectTilesTest < Minitest::Test
  def setup
    @previous_backend = Gsplat.backend
    Gsplat.backend = :ruby
    @means2d = Numo::SFloat[[[8, 8], [24, 8], [16, 8]]]
    @radii = Numo::Int32[[4, 4, 8]]
    @depths = Numo::SFloat[[2, 1, 3]]
  end

  def teardown
    Gsplat.backend = @previous_backend
  end

  def test_three_gaussians_across_two_tiles_have_exact_intersections
    tiles_per_gauss, isect_ids, flatten_ids = Gsplat.isect_tiles(
      @means2d,
      @radii,
      @depths,
      16,
      2,
      1
    )

    expected_keys = [
      intersection_key(0, 0, 2, 1),
      intersection_key(0, 0, 3, 1),
      intersection_key(0, 1, 1, 1),
      intersection_key(0, 1, 3, 1)
    ]
    assert_equal [[1, 1, 2]], tiles_per_gauss.to_a
    assert_equal expected_keys, isect_ids.to_a
    assert_equal [0, 2, 1, 2], flatten_ids.to_a
  end

  def test_sort_false_retains_gaussian_enumeration_order
    _, isect_ids, flatten_ids = Gsplat.isect_tiles(
      @means2d,
      @radii,
      @depths,
      16,
      2,
      1,
      sort: false
    )

    assert_equal [0, 1, 2, 2], flatten_ids.to_a
    tile_ids = isect_ids.to_a.map { |key| (key >> 32) & 1 }
    assert_equal [0, 1, 0, 1], tile_ids
  end

  def test_offset_encode_uses_prefix_offsets_for_empty_tiles
    keys = Numo::Int64[
      intersection_key(0, 1, 1, 2),
      intersection_key(0, 1, 2, 2)
    ]

    offsets = Gsplat.isect_offset_encode(keys, 1, 3, 1)

    assert_equal Numo::Int32, offsets.class
    assert_equal [[[0, 0, 2]]], offsets.to_a
  end

  def test_elliptical_radii_and_zero_radius
    radii = Numo::Int32[[[4, 12], [0, 0], [8, 4]]]

    tiles_per_gauss, = Gsplat.isect_tiles(@means2d, radii, @depths, 16, 2, 1)

    assert_equal [[1, 0, 2]], tiles_per_gauss.to_a
  end

  def test_matches_python_golden_data
    fixture = golden("isect_c3_n1000")
    tile_width = 4
    tile_height = 3
    tiles_per_gauss, isect_ids, flatten_ids = Gsplat.isect_tiles(
      fixture.fetch("means2d"),
      fixture.fetch("radii"),
      fixture.fetch("depths"),
      16,
      tile_width,
      tile_height
    )
    offsets = Gsplat.isect_offset_encode(isect_ids, 3, tile_width, tile_height)

    assert_equal fixture.fetch("tiles_per_gauss").to_a, tiles_per_gauss.to_a
    assert_equal fixture.fetch("isect_ids").to_a, isect_ids.to_a
    assert_equal fixture.fetch("flatten_ids").to_a, flatten_ids.to_a
    assert_equal fixture.fetch("isect_offsets").to_a, offsets.to_a
  end

  private

  def intersection_key(camera_id, tile_id, depth, tile_bits)
    depth_bits = [depth.to_f].pack("g").unpack1("N")
    (((camera_id << tile_bits) | tile_id) << 32) | depth_bits
  end
end
