# frozen_string_literal: true

module Gsplat
  module Backend
    # Tile orchestration for rasterization VJPs.
    module RubyRasterizeToPixelsBackward
      module_function

      # rubocop:disable Metrics/AbcSize, Metrics/ParameterLists
      def backward(means2d, conics, colors, opacities, backgrounds, masks, width, height,
                   tile_size, isect_offsets, flatten_ids, render_alphas, last_ids,
                   grad_render_colors, grad_render_alphas, absgrad:)
        # rubocop:enable Metrics/ParameterLists
        type = means2d.class
        color_gradient_shape = [means2d.shape[0], height, width, colors.shape[-1]]
        grad_render_colors = validate_gradient(type, grad_render_colors, color_gradient_shape)
        grad_render_alphas = validate_gradient(type, grad_render_alphas, [means2d.shape[0], height, width, 1])
        camera_count, gaussian_count = means2d.shape[0...2]
        total_gaussians = camera_count * gaussian_count
        grad_means = type.zeros(total_gaussians, 2)
        grad_conics = type.zeros(total_gaussians, 3)
        grad_colors = type.zeros(total_gaussians, colors.shape[-1])
        grad_opacities = type.zeros(total_gaussians)
        grad_backgrounds = backgrounds && type.zeros(*backgrounds.shape)
        grad_means_abs = absgrad ? type.zeros(total_gaussians, 2) : nil
        flattened = flatten_inputs(means2d, conics, colors, opacities)
        tile_height, tile_width = isect_offsets.shape[-2..]
        camera_count.times do |camera_index|
          tile_height.times do |tile_y|
            tile_width.times do |tile_x|
              backward_tile!(
                grad_means, grad_conics, grad_colors, grad_opacities, grad_backgrounds,
                *flattened, backgrounds, masks, isect_offsets, flatten_ids, render_alphas,
                last_ids, grad_render_colors, grad_render_alphas, camera_index, tile_x, tile_y,
                width, height, tile_size, grad_means_abs
              )
            end
          end
        end
        gradients = [
          grad_means.reshape(*means2d.shape),
          grad_conics.reshape(*conics.shape),
          grad_colors.reshape(*colors.shape),
          grad_opacities.reshape(*opacities.shape),
          grad_backgrounds
        ]
        [gradients, grad_means_abs&.reshape(*means2d.shape)]
      end
      # rubocop:enable Metrics/AbcSize

      def validate_gradient(type, gradient, expected_shape)
        value = type.cast(gradient)
        unless value.shape == expected_shape
          raise ShapeError, "expected output gradient #{expected_shape.inspect}, got #{value.shape.inspect}"
        end

        value
      end
      private_class_method :validate_gradient

      def flatten_inputs(means2d, conics, colors, opacities)
        total = means2d.shape[0] * means2d.shape[1]
        [
          means2d.reshape(total, 2),
          conics.reshape(total, 3),
          colors.reshape(total, colors.shape[-1]),
          opacities.reshape(total)
        ]
      end
      private_class_method :flatten_inputs

      # rubocop:disable Metrics/AbcSize, Metrics/ParameterLists
      def backward_tile!(grad_means, grad_conics, grad_colors, grad_opacities, grad_backgrounds,
                         means2d, conics, colors, opacities, backgrounds, masks, offsets,
                         flatten_ids, render_alphas, last_ids, grad_render_colors, grad_render_alphas,
                         camera_index, tile_x, tile_y, width, height, tile_size, grad_means_abs)
        # rubocop:enable Metrics/ParameterLists
        x_start = tile_x * tile_size
        y_start = tile_y * tile_size
        x_end = [x_start + tile_size, width].min
        y_end = [y_start + tile_size, height].min
        return if x_start >= width || y_start >= height

        alpha_tile = render_alphas[camera_index, y_start...y_end, x_start...x_end, 0].flatten
        color_gradient = grad_render_colors[camera_index, y_start...y_end, x_start...x_end, true]
        color_gradient = color_gradient.reshape(alpha_tile.size, colors.shape[-1])
        alpha_gradient = grad_render_alphas[camera_index, y_start...y_end, x_start...x_end, 0].flatten
        last_id_tile = last_ids[camera_index, y_start...y_end, x_start...x_end].flatten
        if grad_backgrounds
          grad_backgrounds[camera_index, true] += (
            color_gradient * (1 - alpha_tile).reshape(alpha_tile.size, 1)
          ).sum(axis: 0)
        end
        return if masks && masks[camera_index, tile_y, tile_x].zero?

        tile_width = offsets.shape[-1]
        tile_height = offsets.shape[-2]
        flat_tile = (((camera_index * tile_height) + tile_y) * tile_width) + tile_x
        range_start = offsets[flat_tile]
        range_end = flat_tile + 1 < offsets.size ? offsets[flat_tile + 1] : flatten_ids.size
        pixels_x, pixels_y = RubyRasterizeToPixels.send(
          :tile_coordinates,
          means2d.class,
          x_start,
          x_end,
          y_start,
          y_end
        )
        RubyTileCompositorBackward.backward!(
          grad_means, grad_conics, grad_colors, grad_opacities, means2d, conics,
          colors, opacities, flatten_ids, range_start, range_end, pixels_x, pixels_y,
          alpha_tile, last_id_tile, color_gradient, alpha_gradient,
          backgrounds && backgrounds[camera_index, true],
          grad_means_abs: grad_means_abs
        )
      end
      # rubocop:enable Metrics/AbcSize
      private_class_method :backward_tile!
    end
  end
end
