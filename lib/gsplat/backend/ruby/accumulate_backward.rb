# frozen_string_literal: true

module Gsplat
  module Backend
    # Brute-force compositor VJP routed through the one-tile reverse kernel.
    module RubyAccumulateBackward
      module_function

      # rubocop:disable Metrics/ParameterLists
      def backward(means2d, conics, opacities, colors, backgrounds, width, height,
                   render_alphas, last_ids, grad_render_colors, grad_render_alphas)
        # rubocop:enable Metrics/ParameterLists
        camera_count, gaussian_count = means2d.shape[0...2]
        offsets = Numo::Int32.zeros(camera_count, 1, 1)
        camera_count.times { |camera_index| offsets[camera_index, 0, 0] = camera_index * gaussian_count }
        flatten_ids = Numo::Int32.new(camera_count * gaussian_count).seq
        gradients, = RubyRasterizeToPixelsBackward.backward(
          means2d,
          conics,
          colors,
          opacities,
          backgrounds,
          nil,
          width,
          height,
          [width, height].max,
          offsets,
          flatten_ids,
          render_alphas,
          last_ids,
          grad_render_colors,
          grad_render_alphas,
          absgrad: false
        )
        grad_means, grad_conics, grad_colors, grad_opacities, grad_backgrounds = gradients
        [grad_means, grad_conics, grad_opacities, grad_colors, grad_backgrounds]
      end
    end
  end
end
