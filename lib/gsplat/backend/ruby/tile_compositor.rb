# frozen_string_literal: true

module Gsplat
  module Backend
    # Vectorized per-tile alpha compositing kernel.
    module RubyTileCompositor
      module_function

      # rubocop:disable Metrics/AbcSize, Metrics/ParameterLists
      def composite(means2d, conics, colors, opacities, flatten_ids, range_start, range_end,
                    pixels_x, pixels_y)
        # rubocop:enable Metrics/ParameterLists
        pixel_count = pixels_x.size
        channel_count = colors.shape[-1]
        transmittance = means2d.class.ones(pixel_count)
        composited = means2d.class.zeros(pixel_count, channel_count)
        last_ids = Numo::Int32.zeros(pixel_count)
        active = Numo::Bit.ones(pixel_count)
        (range_start...range_end).each do |intersection_index|
          gaussian_index = flatten_ids[intersection_index]
          delta_x = means2d[gaussian_index, 0] - pixels_x
          delta_y = means2d[gaussian_index, 1] - pixels_y
          sigma = (0.5 * (
            (conics[gaussian_index, 0] * (delta_x**2)) +
            (conics[gaussian_index, 2] * (delta_y**2))
          )) + (conics[gaussian_index, 1] * delta_x * delta_y)
          alpha = opacities[gaussian_index] * Numo::NMath.exp(-sigma)
          above_clamp = alpha.gt(RubyAccumulate::ALPHA_CLAMP)
          alpha[above_clamp] = RubyAccumulate::ALPHA_CLAMP if above_clamp.any?
          candidate = active & sigma.ge(0) & alpha.ge(RubyAccumulate::ALPHA_SKIP)
          next_transmittance = transmittance * (1 - alpha)
          accepted = candidate & next_transmittance.gt(RubyAccumulate::TRANSMITTANCE_STOP)
          if accepted.any?
            contribution = means2d.class.zeros(pixel_count)
            contribution[accepted] = (alpha * transmittance)[accepted]
            composited += contribution.reshape(pixel_count, 1) *
                          colors[gaussian_index, true].reshape(1, channel_count)
            transmittance[accepted] = next_transmittance[accepted]
            last_ids[accepted] = intersection_index
          end
          terminating = candidate & next_transmittance.le(RubyAccumulate::TRANSMITTANCE_STOP)
          active[terminating] = 0 if terminating.any?
          break unless active.any?
        end
        [composited, transmittance, last_ids]
      end
      # rubocop:enable Metrics/AbcSize
    end
  end
end
