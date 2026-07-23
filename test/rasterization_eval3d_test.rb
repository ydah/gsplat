# frozen_string_literal: true

require "test_helper"

class RasterizationEval3dTest < Minitest::Test
  def setup
    @previous_backend = Gsplat.backend
    Gsplat.backend = :ruby
  end

  def teardown
    Gsplat.backend = @previous_backend
  end

  def test_world_space_rgb_and_normals_match_analytic_single_ray
    rendered, alphas, meta = render(with_eval3d: true, return_normals: true)

    assert_allclose rendered, floats[[[[0.1, 0.2, 0.3]]]], atol: 1e-12, rtol: 0.0
    assert_allclose alphas, floats[[[[0.5]]]], atol: 1e-12, rtol: 0.0
    assert_allclose meta.fetch(:normals), floats[[[[0.0, 0.0, -0.5]]]], atol: 1e-12, rtol: 0.0
  end

  def test_native_selection_shares_world_space_reference
    expected = render(with_eval3d: true, return_normals: true)
    actual = with_backend(:native) { render(with_eval3d: true, return_normals: true) }

    expected.first(2).zip(actual.first(2)).each do |expected_value, actual_value|
      assert_allclose actual_value, expected_value, atol: 0.0, rtol: 0.0
    end
    assert_allclose actual.last.fetch(:normals), expected.last.fetch(:normals), atol: 0.0, rtol: 0.0
  end

  def test_normal_backward_reaches_quaternion
    quaternion = variable(floats[[1.0, 0.0, 0.0, 0.0]])
    _, _, meta = render(quats: quaternion, with_eval3d: true, return_normals: true)
    normals = meta.fetch(:normals)
    normals.backward(floats.ones(*normals.data.shape))

    assert_allclose quaternion.grad,
                    floats[[0.0, 1.0, -1.0, 0.0]],
                    atol: 1e-8, rtol: 1e-8
  end

  def test_world_space_rgb_backward_reaches_scene_parameters
    inputs = {
      means: variable(floats[[0.0, 0.0, 2.0]]),
      quats: variable(floats[[1.0, 0.0, 0.0, 0.0]]),
      scales: variable(floats[[0.5, 0.5, 0.5]]),
      opacities: variable(floats[0.5]),
      colors: variable(floats[[0.2, 0.4, 0.6]])
    }
    rendered, = render(**inputs, with_eval3d: true)
    rendered.backward(floats.ones(*rendered.data.shape))

    inputs.each_value { |input| refute_nil input.grad }
    assert_operator inputs.fetch(:colors).grad.abs.sum, :>, 0.0
    assert_operator inputs.fetch(:opacities).grad.abs.sum, :>, 0.0
  end

  def test_eval3d_option_constraints
    assert_raises(ArgumentError) { render(return_normals: true) }
    assert_raises(ArgumentError) { render(with_eval3d: true, rasterize_mode: "antialiased") }
    assert_raises(ArgumentError) do
      render(quats: nil, scales: nil, covars: floats.eye(3).reshape(1, 3, 3), with_eval3d: true)
    end
  end

  def test_eval3d_outputs_match_cuda_golden
    fixture = golden("render_eval3d_normals")
    rendered, alphas, meta = render(
      means: fixture.fetch("means"),
      quats: fixture.fetch("quats"),
      scales: fixture.fetch("scales"),
      opacities: fixture.fetch("opacities"),
      colors: fixture.fetch("colors"),
      viewmats: fixture.fetch("viewmats"),
      ks: fixture.fetch("ks"),
      width: fixture.fetch("width").to_i,
      height: fixture.fetch("height").to_i,
      tile_size: 8,
      with_eval3d: true,
      return_normals: true
    )

    assert_allclose rendered, fixture.fetch("render_colors"), atol: 2e-4, rtol: 2e-4
    assert_allclose alphas, fixture.fetch("render_alphas"), atol: 2e-4, rtol: 2e-4
    assert_allclose meta.fetch(:normals), fixture.fetch("render_normals"), atol: 2e-4, rtol: 2e-4
  end

  private

  def render(**overrides)
    defaults = {
      means: floats[[0.0, 0.0, 2.0]],
      quats: floats[[1.0, 0.0, 0.0, 0.0]],
      scales: floats[[0.5, 0.5, 0.5]],
      opacities: floats[0.5],
      colors: floats[[0.2, 0.4, 0.6]],
      viewmats: floats.eye(4).reshape(1, 4, 4),
      ks: floats[[[1.0, 0.0, 0.5], [0.0, 1.0, 0.5], [0.0, 0.0, 1.0]]],
      width: 1,
      height: 1,
      tile_size: 1
    }
    Gsplat.rasterization(**defaults, **overrides)
  end

  def variable(value)
    Gsplat::Autograd::Variable.new(value, requires_grad: true)
  end

  def floats
    Numo::DFloat
  end
end
