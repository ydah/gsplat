# frozen_string_literal: true

require "test_helper"

class RasterizeToIndicesInRangeTest < Minitest::Test
  def setup
    @previous_backend = Gsplat.backend
    Gsplat.backend = :ruby
  end

  def teardown
    Gsplat.backend = @previous_backend
  end

  def test_enumerates_contributions_in_pixel_then_depth_order
    outputs = indices(
      means2d: floats[[[0.5, 0.5], [1.5, 0.5]]],
      conics: floats[[[1, 0, 1], [1, 0, 1]]],
      opacities: floats[[0.5, 0.5]],
      width: 2,
      tile_size: 2,
      flatten_ids: Numo::Int32[0, 1]
    )

    assert_equal [0, 1, 0, 1], outputs[0].to_a
    assert_equal [0, 0, 1, 1], outputs[1].to_a
    assert_equal [0, 0, 0, 0], outputs[2].to_a
  end

  def test_range_selects_tile_intersection_batches
    values = scene(gaussian_count: 3)

    first = indices(**values, range_start: 0, range_end: 1)
    remaining = indices(**values, range_start: 1, range_end: 3)

    assert_equal [0], first[0].to_a
    assert_equal [1, 2], remaining[0].to_a
    assert_equal [0, 0], remaining[1].to_a
  end

  def test_decodes_flattened_ids_for_each_image
    outputs = indices(
      means2d: floats[[[0.5, 0.5]], [[0.5, 0.5]]],
      conics: floats[[[1, 0, 1]], [[1, 0, 1]]],
      opacities: floats[[0.5], [0.5]],
      isect_offsets: Numo::Int32[[[0]], [[1]]],
      flatten_ids: Numo::Int32[0, 1]
    )

    assert_equal [0, 0], outputs[0].to_a
    assert_equal [0, 0], outputs[1].to_a
    assert_equal [0, 1], outputs[2].to_a
  end

  def test_saturating_contribution_is_excluded
    values = scene(gaussian_count: 2, opacity: 0.999)
    gaussian_ids, = indices(**values)

    assert_equal [0], gaussian_ids.to_a
  end

  def test_native_selection_matches_reference
    values = scene(gaussian_count: 3)
    expected = indices(**values)
    actual = with_backend(:native) { indices(**values) }

    expected.zip(actual).each { |left, right| assert_equal left.to_a, right.to_a }
  end

  def test_accepts_variables_without_recording_a_graph
    values = scene(gaussian_count: 1)
    values[:means2d] = Gsplat::Autograd::Variable.new(values[:means2d], requires_grad: true)

    outputs = indices(**values)

    outputs.each { |output| assert_instance_of Numo::Int64, output }
  end

  def test_rejects_invalid_range_and_shapes
    values = scene(gaussian_count: 1)

    assert_raises(ArgumentError) { indices(**values, range_start: 2, range_end: 1) }
    assert_raises(Gsplat::ShapeError) do
      indices(**values, transmittances: floats.ones(1, 2, 1))
    end
  end

  # rubocop:disable Metrics/AbcSize
  def test_outputs_match_cuda_golden
    fixture = golden("raster_indices")
    arguments = {
      transmittances: fixture.fetch("transmittances"),
      means2d: fixture.fetch("means2d"),
      conics: fixture.fetch("conics"),
      opacities: fixture.fetch("opacities"),
      width: fixture.fetch("width").to_i,
      height: fixture.fetch("height").to_i,
      tile_size: fixture.fetch("tile_size").to_i,
      isect_offsets: fixture.fetch("isect_offsets"),
      flatten_ids: fixture.fetch("flatten_ids")
    }
    full = indices(**arguments)
    first = indices(**arguments, range_end: 1)

    assert_equal fixture.fetch("gaussian_ids").to_a, full[0].to_a
    assert_equal fixture.fetch("pixel_ids").to_a, full[1].to_a
    assert_equal fixture.fetch("image_ids").to_a, full[2].to_a
    assert_equal fixture.fetch("first_gaussian_ids").to_a, first[0].to_a
    assert_equal fixture.fetch("first_pixel_ids").to_a, first[1].to_a
    assert_equal fixture.fetch("first_image_ids").to_a, first[2].to_a
  end
  # rubocop:enable Metrics/AbcSize

  private

  # rubocop:disable Metrics/ParameterLists
  def indices(means2d:, conics:, opacities:, flatten_ids:, range_start: 0,
              range_end: 1_000_000_000, transmittances: nil, width: 1, height: 1,
              tile_size: 1, isect_offsets: nil)
    # rubocop:enable Metrics/ParameterLists
    raw_means = Gsplat::Ops::TensorOps.data(means2d)
    transmittances ||= floats.ones(raw_means.shape[0], height, width)
    isect_offsets ||= Numo::Int32.zeros(
      raw_means.shape[0], (height.to_f / tile_size).ceil, (width.to_f / tile_size).ceil
    )
    Gsplat.rasterize_to_indices_in_range(
      range_start, range_end, transmittances, means2d, conics, opacities,
      width, height, tile_size, isect_offsets, flatten_ids
    )
  end

  def scene(gaussian_count:, opacity: 0.2)
    {
      means2d: floats.cast([Array.new(gaussian_count, [0.5, 0.5])]),
      conics: floats.cast([Array.new(gaussian_count, [1.0, 0.0, 1.0])]),
      opacities: floats.cast([Array.new(gaussian_count, opacity)]),
      flatten_ids: Numo::Int32.cast((0...gaussian_count).to_a)
    }
  end

  def floats
    Numo::DFloat
  end
end
