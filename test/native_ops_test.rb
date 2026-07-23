# frozen_string_literal: true

require "test_helper"

class NativeOpsTest < Minitest::Test
  def setup
    skip "run bundle exec rake compile to build the native extension" unless Gsplat::Native.available?
  end

  def test_spherical_harmonics_matches_ruby_backend
    directions = Numo::SFloat.new(64, 3).rand(-1, 1)
    coefficients = Numo::SFloat.new(64, 25, 5).rand(-0.5, 0.5)

    expected = with_backend(:ruby) do
      Gsplat.spherical_harmonics(4, directions, coefficients)
    end
    actual = with_backend(:native) do
      Gsplat.spherical_harmonics(4, directions, coefficients)
    end

    assert_allclose actual, expected, atol: 2e-6, rtol: 2e-5
  end

  def test_pinhole_and_ortho_projection_match_ruby_backend
    %w[pinhole ortho].each do |camera_model|
      expected = with_backend(:ruby) { project(camera_model) }
      actual = with_backend(:native) { project(camera_model) }

      assert_equal expected[0].to_a, actual[0].to_a
      expected.drop(1).zip(actual.drop(1)).each do |ruby_output, native_output|
        assert_allclose native_output, ruby_output, atol: 2e-4, rtol: 2e-4
      end
    end
  end

  def test_intersections_and_offsets_match_ruby_backend
    means2d = Numo::SFloat[[[8, 8], [24, 8], [16, 8]]]
    radii = Numo::Int32[[[4, 12], [4, 4], [8, 4]]]
    depths = Numo::SFloat[[2, 1, 3]]
    expected = with_backend(:ruby) do
      Gsplat.isect_tiles(means2d, radii, depths, 16, 2, 1)
    end
    actual = with_backend(:native) do
      Gsplat.isect_tiles(means2d, radii, depths, 16, 2, 1)
    end

    expected.zip(actual).each { |ruby_output, native_output| assert_equal ruby_output.to_a, native_output.to_a }
    expected_offsets = with_backend(:ruby) do
      Gsplat.isect_offset_encode(expected[1], 1, 2, 1)
    end
    actual_offsets = with_backend(:native) do
      Gsplat.isect_offset_encode(actual[1], 1, 2, 1)
    end
    assert_equal expected_offsets.to_a, actual_offsets.to_a
  end

  def test_raster_forward_and_backward_match_ruby_backend
    inputs = raster_inputs
    expected = Gsplat::Backend::RubyRasterizeToPixels.forward(*inputs)
    actual = Gsplat::NativeRasterOps.forward(*inputs)
    expected.first(2).zip(actual.first(2)).each do |ruby_output, native_output|
      assert_allclose native_output, ruby_output, atol: 2e-6, rtol: 2e-5
    end
    assert_equal expected[2].to_a, actual[2].to_a

    color_grad = Numo::SFloat.new(*actual[0].shape).rand(-0.5, 0.5)
    alpha_grad = Numo::SFloat.new(*actual[1].shape).rand(-0.5, 0.5)
    backward_args = [*inputs, expected[1], expected[2], color_grad, alpha_grad]
    expected_gradients = Gsplat::Backend::RubyRasterizeToPixelsBackward.backward(
      *backward_args, absgrad: true
    )
    actual_gradients = Gsplat::NativeRasterOps.backward(*backward_args, absgrad: true)
    expected_gradients.flatten.zip(actual_gradients.flatten).each do |ruby_output, native_output|
      assert_allclose native_output, ruby_output, atol: 3e-5, rtol: 3e-5
    end
  end

  private

  def raster_inputs
    means = Numo::SFloat[[[1.2, 1.1], [3.6, 2.8], [5.1, 3.9]]]
    conics = Numo::SFloat[
      [[0.3, 0.02, 0.25], [0.2, -0.01, 0.28], [0.35, 0.015, 0.22]]
    ]
    colors = Numo::SFloat[[[0.2, 0.7, 0.1, 0.4], [0.8, 0.1, 0.5, 0.2], [0.4, 0.5, 0.9, 0.3]]]
    opacities = Numo::SFloat[[0.55, 0.45, 0.35]]
    _, keys, ids = Gsplat::Backend::RubyIsectTiles.forward(
      means, Numo::SFloat[[20, 20, 20]], Numo::SFloat[[1, 2, 3]], 4, 2, 2, sort: true
    )
    offsets = Gsplat::Backend::RubyIsectTiles.offset_encode(keys, 1, 2, 2)
    [
      means, conics, colors, opacities, Numo::SFloat[[0.1, 0.2, 0.3, 0.4]],
      Numo::Bit[[[1, 0], [1, 1]]], 7, 5, 4, offsets, ids
    ]
  end

  def project(camera_model)
    rng = Random.new(17)
    means = Numo::SFloat.cast(Array.new(48) { Array.new(3) { rng.rand - 0.5 } })
    means[true, 2] += 2
    quaternions = Numo::SFloat.cast(Array.new(48) { Array.new(4) { (2 * rng.rand) - 1 } })
    scales = Numo::SFloat.cast(Array.new(48) { Array.new(3) { 0.02 + (0.18 * rng.rand) } })
    viewmats = Numo::SFloat.zeros(2, 4, 4)
    2.times { |index| viewmats[index, true, true] = Numo::SFloat.eye(4) }
    viewmats[1, 0, 3] = 0.1
    intrinsics = Numo::SFloat.zeros(2, 3, 3)
    2.times do |index|
      intrinsics[index, true, true] = Numo::SFloat[[52, 0, 32], [0, 52, 24], [0, 0, 1]]
    end
    Gsplat.fully_fused_projection(
      means,
      quats: quaternions,
      scales: scales,
      viewmats: viewmats,
      ks: intrinsics,
      width: 64,
      height: 48,
      calc_compensations: true,
      camera_model: camera_model
    )
  end
end
