# frozen_string_literal: true

require_relative "../backend/ruby/tile_compositor"
require_relative "../backend/ruby/rasterize_to_pixels"
require_relative "../backend/ruby/tile_compositor_backward"
require_relative "../backend/ruby/rasterize_to_pixels_backward"

# Differentiable tile rasterization API.
module Gsplat
  module Ops
    # Tile-accelerated alpha compositor.
    class RasterizeToPixels < Autograd::Function
      class << self
        # rubocop:disable Metrics/ParameterLists
        def forward(context, means2d, conics, colors, opacities, backgrounds, masks, width, height,
                    tile_size, isect_offsets, flatten_ids, absgrad:)
          # rubocop:enable Metrics/ParameterLists
          render_colors, render_alphas, last_ids = Backend.dispatch(
            :rasterize_to_pixels_forward,
            means2d,
            conics,
            colors,
            opacities,
            backgrounds,
            masks,
            width,
            height,
            tile_size,
            isect_offsets,
            flatten_ids
          )
          context.save(
            means2d, conics, colors, opacities, backgrounds, masks, width, height,
            tile_size, isect_offsets, flatten_ids, render_alphas, last_ids, absgrad
          )
          [render_colors, render_alphas]
        end

        # Propagates pixel color and alpha gradients to projected attributes.
        # @api private
        def backward(context, grad_render_colors, grad_render_alphas)
          saved = context.saved_values
          means2d, conics, colors, opacities, backgrounds, masks, width, height,
            tile_size, offsets, flatten_ids, render_alphas, last_ids, absgrad = saved
          gradients, means2d_absgrad = Backend.dispatch(
            :rasterize_to_pixels_backward,
            means2d,
            conics,
            colors,
            opacities,
            backgrounds,
            masks,
            width,
            height,
            tile_size,
            offsets,
            flatten_ids,
            render_alphas,
            last_ids,
            grad_render_colors,
            grad_render_alphas,
            absgrad: absgrad
          )
          if absgrad && context.inputs[0].is_a?(Autograd::Variable)
            context.inputs[0].accumulate_absgrad(means2d_absgrad)
          end
          [*gradients, nil, nil, nil, nil, nil, nil]
        end
      end
    end
  end

  class << self
    # Alpha-composites sorted tile intersections.
    #
    # Projected inputs have shapes `means2d [C,N,2]`, `conics [C,N,3]`,
    # `colors [C,N,D]`, and `opacities [C,N]`. Outputs are color
    # `[C,H,W,D]` and alpha `[C,H,W,1]`.
    #
    # @return [Array<(Numo::NArray, Autograd::Variable)>]
    # rubocop:disable Metrics/ParameterLists
    def rasterize_to_pixels(means2d, conics, colors, opacities, width, height, tile_size,
                            isect_offsets, flatten_ids, backgrounds: nil, masks: nil, absgrad: false)
      # rubocop:enable Metrics/ParameterLists
      inputs = [
        means2d, conics, colors, opacities, backgrounds, masks, width, height,
        tile_size, isect_offsets, flatten_ids
      ]
      return Ops::RasterizeToPixels.apply(*inputs, absgrad: absgrad) if inputs.any?(Autograd::Variable)

      Backend.dispatch(:rasterize_to_pixels_forward, *inputs).first(2)
    end
  end

  Backend.register(
    :rasterize_to_pixels_forward,
    :ruby,
    Backend::RubyRasterizeToPixels.method(:forward)
  )
  Backend.register(
    :rasterize_to_pixels_backward,
    :ruby,
    Backend::RubyRasterizeToPixelsBackward.method(:backward)
  )
end
