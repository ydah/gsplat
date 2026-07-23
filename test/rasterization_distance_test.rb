# frozen_string_literal: true

require "test_helper"

class RasterizationDistanceTest < Minitest::Test
  def setup
    @previous_backend = Gsplat.backend
    Gsplat.backend = :ruby
  end

  def teardown
    Gsplat.backend = @previous_backend
  end

  def test_euclidean_order_uses_camera_space_distance
    means = Numo::DFloat[[1, 0, 2], [0, 0, 2.1]]
    _, _, meta = scene_render(means: means, global_z_order: false)

    assert_allclose meta.fetch(:depths),
                    Numo::DFloat[[::Math.sqrt(5), 2.1]],
                    atol: 1e-12, rtol: 0.0
  end

  def test_hit_distance_modes_use_per_pixel_ray_intersection
    accumulated, alpha, = hit_render(render_mode: "d")
    expected, expected_alpha, = hit_render(render_mode: "Ed")
    combined, = hit_render(render_mode: "RGB-d")
    combined_expected, = hit_render(render_mode: "RGB-Ed")
    native_expected, = with_backend(:native) { hit_render(render_mode: "Ed") }

    assert_allclose alpha, Numo::DFloat[[[[0.5]]]], atol: 1e-12, rtol: 0.0
    assert_allclose accumulated, Numo::DFloat[[[[1.0]]]], atol: 1e-12, rtol: 0.0
    assert_allclose expected_alpha, alpha, atol: 0.0, rtol: 0.0
    assert_allclose expected, Numo::DFloat[[[[2.0]]]], atol: 1e-12, rtol: 0.0
    assert_allclose combined, Numo::DFloat[[[[0.1, 0.2, 0.3, 1.0]]]], atol: 1e-12, rtol: 0.0
    assert_allclose combined_expected, Numo::DFloat[[[[0.1, 0.2, 0.3, 2.0]]]], atol: 1e-12, rtol: 0.0
    assert_allclose native_expected, expected, atol: 0.0, rtol: 0.0
  end

  def test_hit_distance_backward_reaches_world_geometry
    means = Gsplat::Autograd::Variable.new(Numo::DFloat[[0, 0, 2]], requires_grad: true)
    rendered, = hit_render(means: means, render_mode: "Ed")
    rendered.backward(Numo::DFloat.ones(*rendered.data.shape))

    assert_allclose means.grad, Numo::DFloat[[0, 0, 1]], atol: 1e-6, rtol: 1e-6
  end

  def test_hit_distance_modes_match_analytic_golden
    fixture = golden("hit_distance_modes")
    {
      "d" => "render_d",
      "Ed" => "render_ed",
      "RGB-d" => "render_rgb_d",
      "RGB-Ed" => "render_rgb_ed"
    }.each do |mode, name|
      rendered, alphas, = hit_render(render_mode: mode)
      assert_allclose rendered, fixture.fetch(name), atol: 1e-6, rtol: 1e-6
      assert_allclose alphas, fixture.fetch("render_alphas"), atol: 1e-6, rtol: 1e-6
    end
  end

  private

  def hit_render(render_mode:, means: Numo::DFloat[[0, 0, 2]])
    scene_render(
      means: means,
      scales: Numo::DFloat[[0.5, 0.5, 0.5]],
      opacities: Numo::DFloat[0.5],
      colors: Numo::DFloat[[0.2, 0.4, 0.6]],
      width: 1,
      height: 1,
      tile_size: 1,
      ks: Numo::DFloat[[[1, 0, 0.5], [0, 1, 0.5], [0, 0, 1]]],
      global_z_order: false,
      render_mode: render_mode
    )
  end

  def scene_render(means:, **overrides)
    count = Gsplat::Ops::TensorOps.data(means).shape[0]
    defaults = {
      means: means,
      quats: Numo::DFloat.cast(Array.new(count, [1, 0, 0, 0])),
      scales: Numo::DFloat.ones(count, 3) * 0.1,
      opacities: Numo::DFloat.ones(count) * 0.5,
      colors: Numo::DFloat.ones(count, 3) * 0.2,
      viewmats: Numo::DFloat.eye(4).reshape(1, 4, 4),
      ks: Numo::DFloat[[[8, 0, 2], [0, 8, 1.5], [0, 0, 1]]],
      width: 4,
      height: 3,
      tile_size: 4
    }
    Gsplat.rasterization(**defaults, **overrides)
  end
end
