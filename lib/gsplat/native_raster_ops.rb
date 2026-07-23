# frozen_string_literal: true

module Gsplat
  # Validated float32 bridge for native tile rasterization.
  module NativeRasterOps
    module_function

    # rubocop:disable Metrics/ParameterLists
    def forward(means2d, conics, colors, opacities, backgrounds, masks, width, height,
                tile_size, isect_offsets, flatten_ids)
      # rubocop:enable Metrics/ParameterLists
      prepared = Backend::RubyRasterizeToPixels.send(
        :validate_inputs,
        means2d, conics, colors, opacities, backgrounds, masks, width, height,
        tile_size, isect_offsets, flatten_ids
      )
      unless means2d.is_a?(Numo::SFloat)
        return Backend::RubyRasterizeToPixels.forward(
          means2d, conics, colors, opacities, backgrounds, masks, width, height,
          tile_size, isect_offsets, flatten_ids
        )
      end

      typed_means, typed_conics, typed_colors, typed_opacities,
        typed_backgrounds, typed_masks, typed_offsets, typed_ids = prepared
      Native.rasterize_forward_sfloat(
        typed_means.dup, typed_conics.dup, typed_colors.dup, typed_opacities.dup,
        typed_backgrounds&.dup, native_masks(typed_masks), width, height, tile_size,
        typed_offsets.dup, typed_ids.dup
      )
    end

    # rubocop:disable Metrics/ParameterLists
    def backward(means2d, conics, colors, opacities, backgrounds, masks, width, height,
                 tile_size, isect_offsets, flatten_ids, render_alphas, last_ids,
                 grad_render_colors, grad_render_alphas, absgrad:)
      # rubocop:enable Metrics/ParameterLists
      unless means2d.is_a?(Numo::SFloat)
        return Backend::RubyRasterizeToPixelsBackward.backward(
          means2d, conics, colors, opacities, backgrounds, masks, width, height,
          tile_size, isect_offsets, flatten_ids, render_alphas, last_ids,
          grad_render_colors, grad_render_alphas, absgrad: absgrad
        )
      end
      prepared = Backend::RubyRasterizeToPixels.send(
        :validate_inputs,
        means2d, conics, colors, opacities, backgrounds, masks, width, height,
        tile_size, isect_offsets, flatten_ids
      )
      typed_means, typed_conics, typed_colors, typed_opacities,
        typed_backgrounds, typed_masks, typed_offsets, typed_ids = prepared
      color_shape = [typed_means.shape[0], height, width, typed_colors.shape[-1]]
      alpha_shape = [typed_means.shape[0], height, width, 1]
      typed_color_grad = validate_gradient(grad_render_colors, color_shape)
      typed_alpha_grad = validate_gradient(grad_render_alphas, alpha_shape)
      Native.rasterize_backward_sfloat(
        typed_means.dup, typed_conics.dup, typed_colors.dup, typed_opacities.dup,
        typed_backgrounds&.dup, native_masks(typed_masks), width, height, tile_size,
        typed_offsets.dup, typed_ids.dup, Numo::SFloat.cast(render_alphas).dup,
        Numo::Int32.cast(last_ids).dup, typed_color_grad.dup, typed_alpha_grad.dup, absgrad
      )
    end

    def native_masks(masks)
      masks && Numo::Int32.cast(masks).dup
    end
    private_class_method :native_masks

    def validate_gradient(gradient, shape)
      Backend::RubyRasterizeToPixelsBackward.send(
        :validate_gradient, Numo::SFloat, gradient, shape
      )
    end
    private_class_method :validate_gradient
  end
end
