# frozen_string_literal: true

require "test_helper"

class UtilsTest < Minitest::Test
  def test_knn_matches_hand_calculated_distances
    points = Numo::DFloat[[0, 0, 0], [3, 0, 0], [0, 4, 0]]

    distances = Gsplat::Utils.knn(points, k: 3)

    assert_allclose distances, Numo::DFloat[[0, 3, 4], [0, 3, 5], [0, 4, 5]],
                    atol: 0.0, rtol: 0.0
  end

  def test_rgb_and_sh_are_inverse
    colors = Numo::DFloat[[0, 0.5, 1], [0.2, 0.4, 0.6]]

    restored = Gsplat::Utils.sh_to_rgb(Gsplat::Utils.rgb_to_sh(colors))

    assert_allclose restored, colors, atol: 1e-12, rtol: 1e-12
  end

  def test_initializes_all_trainable_point_parameters
    points = Numo::DFloat[[0, 0, 0], [1, 0, 0], [0, 2, 0], [0, 0, 3]]
    colors = Numo::DFloat.ones(4, 3) * 0.5

    params = Gsplat::Utils.init_from_points(points, colors, rng: Random.new(5))

    assert_equal [4, 3], params[:means].data.shape
    assert_equal [4, 15, 3], params[:shN].data.shape
    assert params.values.all?(&:requires_grad?)
    expected_scale = ::Math.log(::Math.sqrt((1 + 4 + 9) / 3.0))
    assert_in_delta expected_scale, params[:scales].data[0, 0], 1e-12
    assert_allclose params[:sh0].data, Numo::DFloat.zeros(4, 1, 3), atol: 1e-12, rtol: 0.0
  end

  def test_scene_scale_matches_camera_center_spread
    cameras = Numo::DFloat.zeros(3, 4, 4)
    3.times { |index| cameras[index, true, true] = Numo::DFloat.eye(4) }
    cameras[0, 0, 3] = -2
    cameras[1, 0, 3] = 1
    cameras[2, 0, 3] = 1

    assert_in_delta 2.0, Gsplat::Utils.scene_scale(cameras), 1e-12
  end
end
