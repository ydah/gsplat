# frozen_string_literal: true

module Gsplat
  module Backend
    # Brute-force Numo alpha compositor used as a rasterization reference.
    module RubyAccumulate
      ALPHA_CLAMP = 0.999
      ALPHA_SKIP = 1.0 / 255.0
      TRANSMITTANCE_STOP = 1e-4

      module_function

      # rubocop:disable Metrics/ParameterLists
      def forward(means2d, conics, opacities, colors, backgrounds, width, height)
        # rubocop:enable Metrics/ParameterLists
        inputs = validate_inputs(means2d, conics, opacities, colors, backgrounds, width, height)
        means2d, conics, opacities, colors, backgrounds = inputs
        camera_count, gaussian_count = means2d.shape[0...2]
        channel_count = colors.shape[-1]
        pixel_count = width * height
        pixels_x, pixels_y = pixel_coordinates(means2d.class, width, height)
        render_colors = means2d.class.zeros(camera_count, pixel_count, channel_count)
        render_alphas = means2d.class.zeros(camera_count, pixel_count)
        camera_count.times do |camera_index|
          composited, transmittance = composite_camera(
            means2d[camera_index, true, true],
            conics[camera_index, true, true],
            opacities[camera_index, true],
            colors[camera_index, true, true],
            pixels_x,
            pixels_y,
            gaussian_count
          )
          if backgrounds
            composited += transmittance.reshape(pixel_count, 1) *
                          backgrounds[camera_index, true].reshape(1, channel_count)
          end
          render_colors[camera_index, true, true] = composited
          render_alphas[camera_index, true] = 1 - transmittance
        end
        [
          render_colors.reshape(camera_count, height, width, channel_count),
          render_alphas.reshape(camera_count, height, width, 1)
        ]
      end

      # rubocop:disable Metrics/AbcSize, Metrics/CyclomaticComplexity, Metrics/ParameterLists, Metrics/PerceivedComplexity
      def validate_inputs(means2d, conics, opacities, colors, backgrounds, width, height)
        # rubocop:enable Metrics/AbcSize, Metrics/CyclomaticComplexity, Metrics/ParameterLists, Metrics/PerceivedComplexity
        unless means2d.is_a?(Numo::NArray) && [Numo::SFloat, Numo::DFloat].include?(means2d.class)
          raise ArgumentError, "means2d must be Numo::SFloat or Numo::DFloat"
        end

        valid_means = means2d.ndim == 3 && means2d.shape[-1] == 2
        raise ShapeError, "expected means2d [C,N,2], got #{means2d.shape.inspect}" unless valid_means

        leading_shape = means2d.shape[0...2]
        conics = means2d.class.cast(conics)
        opacities = means2d.class.cast(opacities)
        colors = means2d.class.cast(colors)
        valid = conics.shape == leading_shape + [3] &&
                opacities.shape == leading_shape &&
                colors.ndim == 3 && colors.shape[0...2] == leading_shape
        unless valid
          raise ShapeError,
                "expected conics [C,N,3], opacities [C,N], and colors [C,N,D], " \
                "got #{conics.shape.inspect}, #{opacities.shape.inspect}, and #{colors.shape.inspect}"
        end
        backgrounds = validate_backgrounds(means2d.class, backgrounds, means2d.shape[0], colors.shape[-1])
        valid_size = width.is_a?(Integer) && width.positive? && height.is_a?(Integer) && height.positive?
        raise ArgumentError, "width and height must be positive integers" unless valid_size

        [means2d, conics, opacities, colors, backgrounds]
      end
      private_class_method :validate_inputs

      def validate_backgrounds(type, backgrounds, camera_count, channel_count)
        return nil unless backgrounds

        value = type.cast(backgrounds)
        expected = [camera_count, channel_count]
        unless value.shape == expected
          raise ShapeError, "expected backgrounds #{expected.inspect}, got #{value.shape.inspect}"
        end

        value
      end
      private_class_method :validate_backgrounds

      def pixel_coordinates(type, width, height)
        x_values = Array.new(height) { (0...width).map { |column| column + 0.5 } }.flatten
        y_values = (0...height).flat_map { |row| Array.new(width, row + 0.5) }
        [type.cast(x_values), type.cast(y_values)]
      end
      private_class_method :pixel_coordinates

      # rubocop:disable Metrics/AbcSize, Metrics/ParameterLists
      def composite_camera(means2d, conics, opacities, colors, pixels_x, pixels_y, gaussian_count)
        # rubocop:enable Metrics/ParameterLists
        pixel_count = pixels_x.size
        channel_count = colors.shape[-1]
        transmittance = means2d.class.ones(pixel_count)
        composited = means2d.class.zeros(pixel_count, channel_count)
        active = Numo::Bit.ones(pixel_count)
        gaussian_count.times do |gaussian_index|
          delta_x = means2d[gaussian_index, 0] - pixels_x
          delta_y = means2d[gaussian_index, 1] - pixels_y
          sigma = (0.5 * (
            (conics[gaussian_index, 0] * (delta_x**2)) +
            (conics[gaussian_index, 2] * (delta_y**2))
          )) + (conics[gaussian_index, 1] * delta_x * delta_y)
          alpha = opacities[gaussian_index] * Numo::NMath.exp(-sigma)
          above_clamp = alpha.gt(ALPHA_CLAMP)
          alpha[above_clamp] = ALPHA_CLAMP if above_clamp.any?
          candidate = active & sigma.ge(0) & alpha.ge(ALPHA_SKIP)
          next_transmittance = transmittance * (1 - alpha)
          accepted = candidate & next_transmittance.gt(TRANSMITTANCE_STOP)
          contribution = means2d.class.zeros(pixel_count)
          contribution[accepted] = (alpha * transmittance)[accepted] if accepted.any?
          composited += contribution.reshape(pixel_count, 1) *
                        colors[gaussian_index, true].reshape(1, channel_count)
          transmittance[accepted] = next_transmittance[accepted] if accepted.any?
          terminating = candidate & next_transmittance.le(TRANSMITTANCE_STOP)
          active[terminating] = 0 if terminating.any?
          break unless active.any?
        end
        [composited, transmittance]
      end
      # rubocop:enable Metrics/AbcSize
      private_class_method :composite_camera
    end
  end
end
