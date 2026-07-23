# frozen_string_literal: true

require "test_helper"

class RasterizeBackwardTest < Minitest::Test
  class RasterLoss < Gsplat::Autograd::Function
    def self.forward(context, colors, alphas, color_weight:, alpha_weight:)
      context.save(color_weight, alpha_weight)
      colors.class.cast((colors * color_weight).sum + (alphas * alpha_weight).sum)
    end

    def self.backward(context, grad_output)
      context.saved_values.map { |weight| weight * grad_output.to_f }
    end
  end

  def setup
    @previous_backend = Gsplat.backend
    Gsplat.backend = :ruby
    @width = 4
    @height = 3
    @tile_size = 4
    @means2d = Numo::DFloat[[[1.2, 1.1], [2.6, 1.8], [3.1, 0.9]]]
    @conics = Numo::DFloat[
      [[0.3, 0.02, 0.25], [0.2, -0.01, 0.28], [0.35, 0.015, 0.22]]
    ]
    @colors = Numo::DFloat[[[0.2, 0.7], [0.8, 0.1], [0.4, 0.5]]]
    @opacities = Numo::DFloat[[0.55, 0.45, 0.35]]
    @backgrounds = Numo::DFloat[[0.1, 0.25]]
    @color_weight = (Numo::DFloat.new(1, @height, @width, 2).seq * 0.01) - 0.08
    @alpha_weight = (Numo::DFloat.new(1, @height, @width, 1).seq * -0.015) + 0.07
    radii = Numo::Int32[[20, 20, 20]]
    depths = Numo::DFloat[[1, 2, 3]]
    _, keys, @flatten_ids = Gsplat.isect_tiles(
      @means2d, radii, depths, @tile_size, 1, 1
    )
    @offsets = Gsplat.isect_offset_encode(keys, 1, 1, 1)
  end

  def teardown
    Gsplat.backend = @previous_backend
  end

  def test_backward_matches_float64_central_difference
    variables = [@means2d, @conics, @colors, @opacities, @backgrounds].map do |value|
      Gsplat::Autograd::Variable.new(value, requires_grad: true)
    end
    rendered, alphas = rasterize(*variables)

    RasterLoss.apply(
      rendered,
      alphas,
      color_weight: @color_weight,
      alpha_weight: @alpha_weight
    ).backward

    values = [@means2d, @conics, @colors, @opacities, @backgrounds]
    values.each_with_index do |value, input_index|
      numeric = central_difference(value) do |perturbed|
        inputs = values.dup
        inputs[input_index] = perturbed
        weighted_raster(*inputs)
      end
      assert_allclose variables[input_index].grad, numeric, atol: 2e-6, rtol: 2e-5
    end
  end

  def test_tiled_gradients_match_brute_force_reference
    tiled_variables = [@means2d, @conics, @colors, @opacities, @backgrounds].map do |value|
      Gsplat::Autograd::Variable.new(value, requires_grad: true)
    end
    tiled_render, tiled_alpha = rasterize(*tiled_variables)
    RasterLoss.apply(
      tiled_render,
      tiled_alpha,
      color_weight: @color_weight,
      alpha_weight: @alpha_weight
    ).backward
    reference_variables = [@means2d, @conics, @opacities, @colors, @backgrounds].map do |value|
      Gsplat::Autograd::Variable.new(value, requires_grad: true)
    end
    reference_render, reference_alpha = Gsplat.accumulate(
      *reference_variables.first(4),
      width: @width,
      height: @height,
      backgrounds: reference_variables[4]
    )
    RasterLoss.apply(
      reference_render,
      reference_alpha,
      color_weight: @color_weight,
      alpha_weight: @alpha_weight
    ).backward
    reference_order = [0, 1, 3, 2, 4]

    tiled_variables.each_with_index do |variable, index|
      assert_allclose variable.grad, reference_variables[reference_order[index]].grad,
                      atol: 1e-12, rtol: 0.0
    end
  end

  def test_matches_python_raster_gradients
    fixture = golden("raster_rgb")
    inputs = %w[means2d conics colors opacities backgrounds].map do |name|
      Gsplat::Autograd::Variable.new(fixture.fetch(name), requires_grad: true)
    end
    rendered, alphas = Gsplat.rasterize_to_pixels(
      *inputs.first(4),
      64,
      48,
      16,
      fixture.fetch("isect_offsets"),
      fixture.fetch("flatten_ids"),
      backgrounds: inputs[4]
    )

    RasterLoss.apply(
      rendered,
      alphas,
      color_weight: fixture.fetch("weight_colors"),
      alpha_weight: fixture.fetch("weight_alphas")
    ).backward

    %w[grad_means2d grad_conics grad_colors grad_opacities grad_backgrounds].each_with_index do |name, index|
      assert_allclose inputs[index].grad, fixture.fetch(name), atol: 1e-4, rtol: 1e-4
    end
  end

  private

  def rasterize(means2d, conics, colors, opacities, backgrounds)
    Gsplat.rasterize_to_pixels(
      means2d,
      conics,
      colors,
      opacities,
      @width,
      @height,
      @tile_size,
      @offsets,
      @flatten_ids,
      backgrounds: backgrounds
    )
  end

  def weighted_raster(means2d, conics, colors, opacities, backgrounds)
    rendered, alphas = rasterize(means2d, conics, colors, opacities, backgrounds)
    (rendered * @color_weight).sum + (alphas * @alpha_weight).sum
  end

  def central_difference(input, epsilon: 1e-6)
    gradient = Numo::DFloat.zeros(*input.shape)
    input.size.times do |index|
      positive = input.flatten.dup
      negative = input.flatten.dup
      positive[index] += epsilon
      negative[index] -= epsilon
      gradient[index] = (
        yield(positive.reshape(*input.shape)) - yield(negative.reshape(*input.shape))
      ) / (2.0 * epsilon)
    end
    gradient
  end
end
