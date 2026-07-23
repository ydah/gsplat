# frozen_string_literal: true

require "test_helper"

class ProjectionForwardTest < Minitest::Test
  def setup
    @previous_backend = Gsplat.backend
    Gsplat.backend = :ruby
  end

  def teardown
    Gsplat.backend = @previous_backend
  end

  def test_world_to_cam_transforms_means_and_covariances
    means = Numo::DFloat[[1, 2, 3]]
    covars = Numo::DFloat[[[4, 0, 0], [0, 9, 0], [0, 0, 16]]]
    viewmats = Numo::DFloat.zeros(1, 4, 4)
    viewmats[0, true, true] = Numo::DFloat[
      [0, -1, 0, 10],
      [1, 0, 0, -2],
      [0, 0, 1, 1],
      [0, 0, 0, 1]
    ]

    camera_means, camera_covars = Gsplat.world_to_cam(means, covars, viewmats)

    assert_allclose camera_means, Numo::DFloat[[[8, -1, 4]]], atol: 1e-12, rtol: 0.0
    assert_allclose(
      camera_covars,
      Numo::DFloat[[[[9, 0, 0], [0, 4, 0], [0, 0, 16]]]],
      atol: 1e-12,
      rtol: 0.0
    )
  end

  def test_persp_proj_matches_hand_calculated_gaussian
    means = Numo::DFloat[[[0, 0, 2]]]
    covars = Numo::DFloat[[[[0.01, 0, 0], [0, 0.01, 0], [0, 0, 0.01]]]]

    means2d, covars2d = Gsplat.persp_proj(means, covars, intrinsics, 100, 80)

    assert_allclose means2d, Numo::DFloat[[[50, 40]]], atol: 1e-12, rtol: 0.0
    assert_allclose covars2d, Numo::DFloat[[[[25, 0], [0, 25]]]], atol: 1e-12, rtol: 0.0
  end

  def test_fully_fused_projection_accepts_covariances
    means = Numo::DFloat[[0, 0, 2]]
    covars = Numo::DFloat[[[0.01, 0, 0], [0, 0.01, 0], [0, 0, 0.01]]]

    radii, means2d, depths, conics, compensations = Gsplat.fully_fused_projection(
      means,
      covars: covars,
      viewmats: identity_viewmats,
      ks: intrinsics,
      width: 100,
      height: 80,
      calc_compensations: true
    )

    assert_equal Numo::Int32, radii.class
    assert_equal [[1, 1, 2], [1, 1, 2], [1, 1], [1, 1, 3], [1, 1]],
                 [radii.shape, means2d.shape, depths.shape, conics.shape, compensations.shape]
    assert_equal [17, 17], radii[0, 0, true].to_a
    assert_allclose means2d, Numo::DFloat[[[50, 40]]], atol: 1e-12, rtol: 0.0
    assert_allclose depths, Numo::DFloat[[2]], atol: 1e-12, rtol: 0.0
    assert_allclose conics, Numo::DFloat[[[1.0 / 25.3, 0, 1.0 / 25.3]]], atol: 1e-12, rtol: 0.0
    assert_allclose compensations, Numo::DFloat[[25.0 / 25.3]], atol: 1e-12, rtol: 0.0
  end

  def test_quaternion_scale_path_and_culling_rules
    means = Numo::SFloat[[0, 0, 2], [0, 0, 0.005], [100, 0, 2]]
    quaternions = Numo::SFloat[[1, 0, 0, 0], [1, 0, 0, 0], [1, 0, 0, 0]]
    scales = Numo::SFloat.ones(3, 3) * 0.1

    radii, = Gsplat.fully_fused_projection(
      means,
      quats: quaternions,
      scales: scales,
      viewmats: identity_viewmats(Numo::SFloat),
      ks: intrinsics(Numo::SFloat),
      width: 100,
      height: 80
    )

    assert_equal [17, 17, 0, 0, 0, 0], radii.to_a.flatten

    clipped, = Gsplat.fully_fused_projection(
      means[0...1, true],
      quats: quaternions[0...1, true],
      scales: scales[0...1, true],
      viewmats: identity_viewmats(Numo::SFloat),
      ks: intrinsics(Numo::SFloat),
      width: 100,
      height: 80,
      radius_clip: 17
    )
    assert_equal [0, 0], clipped[0, 0, true].to_a
  end

  def test_projection_reports_axis_aligned_elliptical_radii
    covariance = Numo::DFloat[[[0.01, 0, 0], [0, 0.04, 0], [0, 0, 0.01]]]
    radii, = Gsplat.fully_fused_projection(
      Numo::DFloat[[0, 0, 2]],
      covars: covariance,
      viewmats: identity_viewmats,
      ks: intrinsics,
      width: 100,
      height: 80
    )

    assert_equal [17, 34], radii[0, 0, true].to_a
  end

  def test_matches_python_pinhole_golden_data
    fixture = golden("proj_pinhole_c1_n1000")
    radii, means2d, depths, conics, compensations = Gsplat.fully_fused_projection(
      fixture.fetch("means"),
      quats: fixture.fetch("quats"),
      scales: fixture.fetch("scales"),
      viewmats: fixture.fetch("viewmats"),
      ks: fixture.fetch("ks"),
      width: fixture.fetch("width").to_i,
      height: fixture.fetch("height").to_i,
      calc_compensations: true
    )
    assert_equal fixture.fetch("radii").to_a, radii.to_a
    assert_allclose means2d, fixture.fetch("means2d"), atol: 1e-4, rtol: 1e-5
    assert_allclose depths, fixture.fetch("depths"), atol: 1e-5, rtol: 1e-5
    assert_allclose conics, fixture.fetch("conics"), atol: 1e-4, rtol: 1e-4
    assert_allclose compensations, fixture.fetch("compensations"), atol: 1e-5, rtol: 1e-5
  end

  private

  def identity_viewmats(type = Numo::DFloat)
    output = type.zeros(1, 4, 4)
    output[0, true, true] = type.eye(4)
    output
  end

  def intrinsics(type = Numo::DFloat)
    output = type.zeros(1, 3, 3)
    output[0, true, true] = type[[100, 0, 50], [0, 100, 40], [0, 0, 1]]
    output
  end
end
