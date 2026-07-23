# frozen_string_literal: true

require_relative "../backend/ruby/eval3d_rasterizer"

module Gsplat
  module Ops
    # Differentiable world-space reference rasterizer for hit-distance modes.
    class Eval3dRasterize < Autograd::Function
      class << self
        # rubocop:disable Metrics/ParameterLists
        def forward(context, means, quats, scales, colors, opacities, backgrounds, **options)
          # rubocop:enable Metrics/ParameterLists
          context.save(means, quats, scales, colors, opacities, backgrounds, options)
          Backend::RubyEval3dRasterizer.forward(
            means, quats, scales, colors, opacities, backgrounds, **options
          )
        end

        def backward(context, grad_rendered, grad_alphas)
          *inputs, options = context.saved_values
          context.needs_input_grad.each_with_index.map do |needed, input_index|
            numerical_vjp(inputs, input_index, grad_rendered, grad_alphas, options) if needed
          end
        end

        private

        def numerical_vjp(inputs, input_index, grad_rendered, grad_alphas, options)
          input = inputs[input_index]
          epsilon = input.is_a?(Numo::DFloat) ? 1e-5 : 1e-3
          gradient = input.class.zeros(*input.shape)
          input.size.times do |element|
            positive_inputs = inputs.dup
            negative_inputs = inputs.dup
            positive_inputs[input_index] = perturb(input, element, epsilon)
            negative_inputs[input_index] = perturb(input, element, -epsilon)
            positive = objective(positive_inputs, grad_rendered, grad_alphas, options)
            negative = objective(negative_inputs, grad_rendered, grad_alphas, options)
            gradient[element] = (positive - negative) / (2 * epsilon)
          end
          gradient
        end

        def perturb(input, index, amount)
          flat = input.flatten.dup
          flat[index] += amount
          flat.reshape(*input.shape)
        end

        def objective(inputs, grad_rendered, grad_alphas, options)
          rendered, alphas = Backend::RubyEval3dRasterizer.forward(*inputs, **options)
          ((rendered * grad_rendered).sum + (alphas * grad_alphas).sum).to_f
        end
      end
    end
  end
end
