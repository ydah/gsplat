# frozen_string_literal: true

require_relative "../backend/ruby/tile_compositor"
require_relative "../backend/ruby/rasterize_to_pixels"

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

        def backward(_context, *_grad_outputs)
          raise Gsplat::Error, "rasterize_to_pixels backward is not implemented yet"
        end
      end
    end
  end

  class << self
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
end
