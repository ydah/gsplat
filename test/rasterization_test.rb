# frozen_string_literal: true

require "test_helper"

class RasterizationTest < Minitest::Test
  REQUIRED_META_KEYS = %i[
    camera_ids gaussian_ids radii means2d depths conics opacities tile_width tile_height
    tiles_per_gauss isect_ids flatten_ids isect_offsets width height tile_size n_cameras
  ].freeze

  class RenderLoss < Gsplat::Autograd::Function
    def self.forward(context, colors, alphas)
      context.save(Numo::DFloat.ones(*colors.shape), Numo::DFloat.ones(*alphas.shape) * 0.2)
      colors.class.cast(colors.sum + (alphas * 0.2).sum)
    end

    def self.backward(context, grad_output)
      context.saved_values.map { |gradient| gradient * grad_output.to_f }
    end
  end

  def setup
    @previous_backend = Gsplat.backend
    Gsplat.backend = :ruby
    @means = Numo::DFloat[[0, 0, 2], [0.2, -0.1, 2.5]]
    @quaternions = Numo::DFloat[[1, 0, 0, 0], [0.9, 0.1, -0.2, 0.3]]
    @scales = Numo::DFloat[[0.08, 0.1, 0.09], [0.1, 0.07, 0.08]]
    @opacities = Numo::DFloat[0.7, 0.5]
    @colors = Numo::DFloat[[0.2, 0.4, 0.6], [0.8, 0.1, 0.3]]
    @viewmats = Numo::DFloat.zeros(1, 4, 4)
    @viewmats[0, true, true] = Numo::DFloat.eye(4)
    @intrinsics = Numo::DFloat[[[8, 0, 2], [0, 8, 1.5], [0, 0, 1]]]
  end

  def teardown
    Gsplat.backend = @previous_backend
  end

  def test_rgb_pipeline_returns_complete_metadata
    rendered, alphas, meta = render

    assert_equal [1, 3, 4, 3], rendered.shape
    assert_equal [1, 3, 4, 1], alphas.shape
    assert_empty REQUIRED_META_KEYS - meta.keys
    assert_equal 1, meta.fetch(:n_cameras)
    assert_equal 4, meta.fetch(:width)
    assert_equal 3, meta.fetch(:height)
  end

  def test_depth_render_modes_and_expected_depth_normalization
    depth, alpha, = render(render_mode: "D")
    expected_depth, expected_alpha, = render(render_mode: "ED")
    rgb_depth, = render(render_mode: "RGB+D")
    rgb_expected, = render(render_mode: "RGB+ED")
    visible = alpha[true, true, true, 0].gt(0)

    assert_equal [1, 3, 4, 1], depth.shape
    assert_equal [1, 3, 4, 1], expected_depth.shape
    assert_equal [1, 3, 4, 4], rgb_depth.shape
    assert_equal [1, 3, 4, 4], rgb_expected.shape
    assert_allclose expected_alpha, alpha, atol: 0.0, rtol: 0.0
    assert expected_depth[true, true, true, 0][visible].ge(2.0 - 1e-12).all?
    assert expected_depth[true, true, true, 0][visible].le(2.5).all?
  end

  def test_antialiased_and_spherical_harmonic_paths
    classic, = render
    antialiased, = render(rasterize_mode: "antialiased")
    coefficients = Numo::DFloat.zeros(2, 1, 3)
    coefficients[true, 0, true] = @colors
    spherical, = render(colors: coefficients, sh_degree: 0)

    refute_equal classic.to_a, antialiased.to_a
    assert_equal classic.shape, spherical.shape
    assert spherical.ge(0).all?
  end

  def test_backward_reaches_scene_inputs_and_publishes_absgrad
    variables = [@means, @quaternions, @scales, @opacities, @colors].map do |value|
      Gsplat::Autograd::Variable.new(value, requires_grad: true)
    end
    rendered, alphas, meta = render(
      means: variables[0],
      quats: variables[1],
      scales: variables[2],
      opacities: variables[3],
      colors: variables[4],
      absgrad: true
    )

    RenderLoss.apply(rendered, alphas).backward

    variables.each { |variable| refute_nil variable.grad }
    refute_nil meta.fetch(:means2d_absgrad)
    assert_allclose meta.fetch(:means2d_absgrad), meta.fetch(:means2d).absgrad,
                    atol: 0.0, rtol: 0.0
  end

  def test_shape_errors_include_expected_and_actual_shapes
    error = assert_raises(Gsplat::ShapeError) { render(colors: Numo::DFloat.zeros(3, 3)) }

    assert_includes error.message, "expected colors"
    assert_includes error.message, "[3, 3]"
  end

  def test_nd_features_match_across_chunk_sizes_and_backends
    [8, 40].each do |channels|
      colors = Numo::SFloat.cast(
        Array.new(2) { |gaussian| Array.new(channels) { |channel| (gaussian + channel + 1) / 50.0 } }
      )
      backgrounds = Numo::SFloat.cast([Array.new(channels) { |channel| channel / 100.0 }])
      expected = with_backend(:ruby) do
        render_typed(colors, backgrounds, channel_chunk: channels + 1)
      end
      chunked = with_backend(:ruby) do
        render_typed(colors, backgrounds, channel_chunk: 7)
      end
      native = with_backend(:native) do
        render_typed(colors, backgrounds, channel_chunk: 7)
      end

      assert_equal [1, 3, 4, channels], chunked[0].shape
      assert_allclose chunked[0], expected[0], atol: 2e-6, rtol: 2e-5
      assert_allclose chunked[1], expected[1], atol: 2e-6, rtol: 2e-5
      assert_allclose native[0], expected[0], atol: 3e-5, rtol: 3e-5
      assert_allclose native[1], expected[1], atol: 3e-5, rtol: 3e-5
    end
  end

  def test_chunked_feature_backward_and_radius_clip
    colors = Gsplat::Autograd::Variable.new(Numo::DFloat.new(2, 40).rand, requires_grad: true)
    rendered, alphas, = render(colors: colors, channel_chunk: 8)

    RenderLoss.apply(rendered, alphas).backward

    assert_equal [2, 40], colors.grad.shape
    assert colors.grad.abs.gt(0).any?
    clipped, clipped_alpha, = render(
      colors: Numo::DFloat.ones(2, 8),
      backgrounds: Numo::DFloat.ones(1, 8) * 0.25,
      radius_clip: 100
    )
    assert_allclose clipped, Numo::DFloat.ones(1, 3, 4, 8) * 0.25, atol: 0.0, rtol: 0.0
    assert_allclose clipped_alpha, Numo::DFloat.zeros(1, 3, 4, 1), atol: 0.0, rtol: 0.0
  end

  private

  def render_typed(colors, backgrounds, channel_chunk:)
    Gsplat.rasterization(
      means: Numo::SFloat.cast(@means),
      quats: Numo::SFloat.cast(@quaternions),
      scales: Numo::SFloat.cast(@scales),
      opacities: Numo::SFloat.cast(@opacities),
      colors: colors,
      viewmats: Numo::SFloat.cast(@viewmats),
      ks: Numo::SFloat.cast(@intrinsics),
      backgrounds: backgrounds,
      width: 4,
      height: 3,
      tile_size: 4,
      channel_chunk: channel_chunk
    )
  end

  def render(**overrides)
    Gsplat.rasterization(
      means: @means,
      quats: @quaternions,
      scales: @scales,
      opacities: @opacities,
      colors: @colors,
      viewmats: @viewmats,
      ks: @intrinsics,
      width: 4,
      height: 3,
      tile_size: 4,
      **overrides
    )
  end
end
