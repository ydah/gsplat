# frozen_string_literal: true

require_relative "../backend/ruby/accumulate"
require_relative "../backend/ruby/accumulate_backward"

# Brute-force reference rasterization API.
module Gsplat
  # Differentiable operation classes and low-level tensor adapters.
  module Ops
    # Differentiable reference alpha compositor.
    class Accumulate < Autograd::Function
      class << self
        # rubocop:disable Metrics/ParameterLists
        def forward(context, means2d, conics, opacities, colors, backgrounds, width, height)
          # rubocop:enable Metrics/ParameterLists
          render_colors, render_alphas, last_ids = Backend.dispatch(
            :accumulate_forward,
            means2d,
            conics,
            opacities,
            colors,
            backgrounds,
            width,
            height
          )
          context.save(
            means2d, conics, opacities, colors, backgrounds, width, height,
            render_alphas, last_ids
          )
          [render_colors, render_alphas]
        end

        # Propagates color and alpha gradients through the reference compositor.
        # @api private
        def backward(context, grad_render_colors, grad_render_alphas)
          saved = context.saved_values
          gradients = Backend.dispatch(
            :accumulate_backward,
            *saved,
            grad_render_colors,
            grad_render_alphas
          )
          [*gradients, nil, nil]
        end
      end
    end
  end

  class << self
    # Composites every Gaussian over every pixel without tile acceleration.
    # rubocop:disable Metrics/ParameterLists
    def accumulate(means2d, conics, opacities, colors, width:, height:, backgrounds: nil)
      # rubocop:enable Metrics/ParameterLists
      inputs = [means2d, conics, opacities, colors, backgrounds]
      return Ops::Accumulate.apply(*inputs, width, height) if inputs.any?(Autograd::Variable)

      Backend.dispatch(
        :accumulate_forward,
        *inputs,
        width,
        height
      ).first(2)
    end
  end

  Backend.register(:accumulate_forward, :ruby, Backend::RubyAccumulate.method(:forward))
  Backend.register(:accumulate_backward, :ruby, Backend::RubyAccumulateBackward.method(:backward))
end
