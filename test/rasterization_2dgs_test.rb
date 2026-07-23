# frozen_string_literal: true

require "test_helper"

class Rasterization2dgsTest < Minitest::Test
  def setup
    @previous_backend = Gsplat.backend
    Gsplat.backend = :ruby
    @means = Numo::DFloat[[0, 0, 2]]
    @quats = Numo::DFloat[[1, 0, 0, 0]]
    @scales = Numo::DFloat[[0.2, 0.2, 0.01]]
    @opacities = Numo::DFloat[0.7]
    @colors = Numo::DFloat[[0.2, 0.4, 0.6]]
    @views = Numo::DFloat.eye(4).reshape(1, 4, 4)
    @intrinsics = Numo::DFloat[[[4, 0, 1], [0, 4, 1], [0, 0, 1]]]
  end

  def teardown
    Gsplat.backend = @previous_backend
  end

  def test_returns_complete_auxiliary_buffers
    colors, alphas, normals, surface_normals, distortion, median, meta = render(distloss: true)

    assert_equal [1, 2, 2, 3], colors.shape
    assert_equal [1, 2, 2, 1], alphas.shape
    assert_equal [1, 2, 2, 3], normals.shape
    assert_equal [1, 2, 2, 3], surface_normals.shape
    assert_equal [1, 2, 2, 1], distortion.shape
    assert_equal [1, 2, 2, 1], median.shape
    assert_equal [1, 1, 3, 3], meta.fetch(:ray_transforms).shape
    assert_equal [1, 1, 3], meta.fetch(:normals).shape
    assert_same meta.fetch(:means2d), meta.fetch(:gradient_2dgs)
  end

  def test_primary_render_is_differentiable
    variables = [@means, @quats, @scales, @opacities, @colors].map do |value|
      Gsplat::Autograd::Variable.new(value, requires_grad: true)
    end
    rendered, = render(
      means: variables[0], quats: variables[1], scales: variables[2],
      opacities: variables[3], colors: variables[4]
    )
    rendered.backward(Numo::DFloat.ones(*rendered.data.shape))

    variables.each { |variable| refute_nil variable.grad }
  end

  def test_projection_returns_2dgs_metadata_shapes
    outputs = Gsplat.fully_fused_projection_2dgs(
      @means, quats: @quats, scales: @scales, viewmats: @views, ks: @intrinsics,
              width: 2, height: 2
    )

    assert_equal [1, 1, 2], outputs[0].shape
    assert_equal [1, 1, 2], outputs[1].shape
    assert_equal [1, 1], outputs[2].shape
    assert_equal [1, 1, 3, 3], outputs[3].shape
    assert_equal [1, 1, 3], outputs[4].shape
  end

  def test_matches_python_2dgs_golden
    fixture = golden("raster_2dgs")
    outputs = render

    %w[render_colors render_alphas render_normals render_surface_normals render_distort render_median]
      .zip(outputs).each do |name, actual|
        assert_allclose actual, fixture.fetch(name), atol: 3e-3, rtol: 3e-3
      end
  end

  private

  def render(**overrides)
    Gsplat.rasterization_2dgs(
      means: @means, quats: @quats, scales: @scales, opacities: @opacities,
      colors: @colors, viewmats: @views, ks: @intrinsics, width: 2, height: 2,
      tile_size: 2, **overrides
    )
  end
end
