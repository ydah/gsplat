# frozen_string_literal: true

require "test_helper"

class RasterizeAbsgradTest < Minitest::Test
  class ColorLoss < Gsplat::Autograd::Function
    def self.forward(context, rendered, weight:)
      context.save(weight)
      rendered.class.cast((rendered * weight).sum)
    end

    def self.backward(context, grad_output)
      [context.saved_values.fetch(0) * grad_output.to_f]
    end
  end

  def setup
    @previous_backend = Gsplat.backend
    Gsplat.backend = :ruby
    @means = Numo::DFloat[[[0.8, 0.7], [1.9, 1.4]]]
    @conics = Numo::DFloat[[[0.4, 0.02, 0.35], [0.3, -0.01, 0.25]]]
    @colors = Numo::DFloat[[[0.2, 0.7], [0.8, 0.1]]]
    @opacities = Numo::DFloat[[0.6, 0.5]]
    _, keys, @flatten_ids = Gsplat.isect_tiles(
      @means,
      Numo::Int32[[10, 10]],
      Numo::DFloat[[1, 2]],
      4,
      1,
      1
    )
    @offsets = Gsplat.isect_offset_encode(keys, 1, 1, 1)
    @weight = Numo::DFloat[
      [[[0.3, -0.2], [-0.4, 0.5]], [[0.1, 0.6], [-0.7, 0.2]]]
    ]
  end

  def teardown
    Gsplat.backend = @previous_backend
  end

  def test_absgrad_accumulates_absolute_pixel_contributions
    means = Gsplat::Autograd::Variable.new(@means, requires_grad: true)
    rendered, = rasterize(means, absgrad: true)

    ColorLoss.apply(rendered, weight: @weight).backward

    refute_nil means.absgrad
    assert_equal @means.shape, means.absgrad.shape
    assert means.absgrad.ge(0).all?
    assert means.absgrad.ge(means.grad.abs).all?
  end

  def test_absgrad_is_not_allocated_when_disabled
    means = Gsplat::Autograd::Variable.new(@means, requires_grad: true)
    rendered, = rasterize(means, absgrad: false)

    ColorLoss.apply(rendered, weight: @weight).backward

    assert_nil means.absgrad
  end

  def test_matches_python_absgrad
    fixture = golden("raster_rgb")
    means = Gsplat::Autograd::Variable.new(fixture.fetch("means2d"), requires_grad: true)
    rendered, = Gsplat.rasterize_to_pixels(
      means,
      fixture.fetch("conics"),
      fixture.fetch("colors"),
      fixture.fetch("opacities"),
      64,
      48,
      16,
      fixture.fetch("isect_offsets"),
      fixture.fetch("flatten_ids"),
      backgrounds: fixture.fetch("backgrounds"),
      absgrad: true
    )

    ColorLoss.apply(rendered, weight: fixture.fetch("weight_colors")).backward

    assert_allclose means.absgrad, fixture.fetch("means2d_absgrad"), atol: 1e-4, rtol: 1e-4
  end

  private

  def rasterize(means, absgrad:)
    Gsplat.rasterize_to_pixels(
      means,
      @conics,
      @colors,
      @opacities,
      2,
      2,
      4,
      @offsets,
      @flatten_ids,
      absgrad: absgrad
    )
  end
end
