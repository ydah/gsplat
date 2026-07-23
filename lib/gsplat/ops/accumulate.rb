# frozen_string_literal: true

require_relative "../backend/ruby/accumulate"

# Brute-force reference rasterization API.
module Gsplat
  module Ops
    # Differentiable reference alpha compositor.
    class Accumulate < Autograd::Function
      class << self
        # rubocop:disable Metrics/ParameterLists
        def forward(context, means2d, conics, opacities, colors, backgrounds, width, height)
          # rubocop:enable Metrics/ParameterLists
          context.save(means2d, conics, opacities, colors, backgrounds, width, height)
          Backend.dispatch(
            :accumulate_forward,
            means2d,
            conics,
            opacities,
            colors,
            backgrounds,
            width,
            height
          )
        end

        def backward(_context, *_grad_outputs)
          raise Gsplat::Error, "accumulate backward is not implemented yet"
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
      )
    end
  end

  Backend.register(:accumulate_forward, :ruby, Backend::RubyAccumulate.method(:forward))
end
