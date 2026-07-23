# frozen_string_literal: true

module Gsplat
  module Backend
    # Reverse per-tile compositor with scatter-add into Gaussian gradients.
    module RubyTileCompositorBackward
      module_function

      # rubocop:disable Metrics/AbcSize, Metrics/BlockLength, Metrics/MethodLength, Metrics/ParameterLists
      def backward!(grad_means, grad_conics, grad_colors, grad_opacities, means2d, conics,
                    colors, opacities, flatten_ids, range_start, range_end, pixels_x, pixels_y,
                    render_alphas, last_ids, grad_render_colors, grad_render_alphas, background,
                    grad_means_abs: nil)
        # rubocop:enable Metrics/ParameterLists
        pixel_count = pixels_x.size
        current_transmittance = 1 - render_alphas
        after_transmittance = means2d.class.ones(pixel_count)
        remaining_color = means2d.class.zeros(pixel_count, colors.shape[-1])
        remaining_color[true, true] = background if background
        (range_start...range_end).reverse_each do |intersection_index|
          gaussian_index = flatten_ids[intersection_index]
          delta_x = means2d[gaussian_index, 0] - pixels_x
          delta_y = means2d[gaussian_index, 1] - pixels_y
          sigma = (0.5 * (
            (conics[gaussian_index, 0] * (delta_x**2)) +
            (conics[gaussian_index, 2] * (delta_y**2))
          )) + (conics[gaussian_index, 1] * delta_x * delta_y)
          gaussian_response = Numo::NMath.exp(-sigma)
          raw_alpha = opacities[gaussian_index] * gaussian_response
          alpha = raw_alpha.dup
          clamped = alpha.gt(RubyAccumulate::ALPHA_CLAMP)
          alpha[clamped] = RubyAccumulate::ALPHA_CLAMP if clamped.any?
          valid = render_alphas.gt(0) & sigma.ge(0) & alpha.ge(RubyAccumulate::ALPHA_SKIP)
          valid &= last_ids.ge(intersection_index)
          valid &= current_transmittance.lt(1)
          valid &= alpha.lt(1)
          next unless valid.any?

          transmittance_before = current_transmittance.dup
          transmittance_before[valid] = current_transmittance[valid] / (1 - alpha[valid])
          visibility = means2d.class.zeros(pixel_count)
          visibility[valid] = (transmittance_before * alpha)[valid]
          color = colors[gaussian_index, true]
          grad_colors[gaussian_index, true] += (
            grad_render_colors * visibility.reshape(pixel_count, 1)
          ).sum(axis: 0)
          color_difference = color.reshape(1, colors.shape[-1]) - remaining_color
          grad_alpha = transmittance_before * (
            (grad_render_colors * color_difference).sum(axis: 1) +
            (grad_render_alphas * after_transmittance)
          )
          unclamped = valid & raw_alpha.lt(RubyAccumulate::ALPHA_CLAMP)
          grad_raw_alpha = means2d.class.zeros(pixel_count)
          grad_raw_alpha[unclamped] = grad_alpha[unclamped] if unclamped.any?
          grad_opacities[gaussian_index] += (grad_raw_alpha * gaussian_response).sum
          grad_sigma = -grad_raw_alpha * raw_alpha
          mean_x = grad_sigma * (
            (conics[gaussian_index, 0] * delta_x) +
            (conics[gaussian_index, 1] * delta_y)
          )
          mean_y = grad_sigma * (
            (conics[gaussian_index, 2] * delta_y) +
            (conics[gaussian_index, 1] * delta_x)
          )
          grad_means[gaussian_index, 0] += mean_x.sum
          grad_means[gaussian_index, 1] += mean_y.sum
          if grad_means_abs
            grad_means_abs[gaussian_index, 0] += mean_x.abs.sum
            grad_means_abs[gaussian_index, 1] += mean_y.abs.sum
          end
          grad_conics[gaussian_index, 0] += (grad_sigma * 0.5 * (delta_x**2)).sum
          grad_conics[gaussian_index, 1] += (grad_sigma * delta_x * delta_y).sum
          grad_conics[gaussian_index, 2] += (grad_sigma * 0.5 * (delta_y**2)).sum
          updated_color = (alpha.reshape(pixel_count, 1) * color.reshape(1, colors.shape[-1])) +
                          ((1 - alpha).reshape(pixel_count, 1) * remaining_color)
          remaining_color[valid, true] = updated_color[valid, true]
          after_transmittance[valid] *= 1 - alpha[valid]
          current_transmittance[valid] = transmittance_before[valid]
        end
      end
      # rubocop:enable Metrics/AbcSize, Metrics/BlockLength, Metrics/MethodLength
    end
  end
end
