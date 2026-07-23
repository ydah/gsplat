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

  private

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
