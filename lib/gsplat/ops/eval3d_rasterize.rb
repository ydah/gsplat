# frozen_string_literal: true

require_relative "../backend/ruby/eval3d_rasterizer"

module Gsplat
  module Ops
    # Differentiable world-space reference rasterizer with optional normals.
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

        def backward(context, grad_rendered, grad_alphas, grad_normals)
          *inputs, options = context.saved_values
          gradient_outputs = [grad_rendered, grad_alphas, grad_normals]
          context.needs_input_grad.each_with_index.map do |needed, input_index|
            next unless needed

            numerical_vjp(inputs, input_index, gradient_outputs, options)
          end
        end

        private

        def numerical_vjp(inputs, input_index, gradient_outputs, options)
          input = inputs[input_index]
          epsilon = input.is_a?(Numo::DFloat) ? 1e-5 : 1e-3
          gradient = input.class.zeros(*input.shape)
          input.size.times do |element|
            positive_inputs = inputs.dup
            negative_inputs = inputs.dup
            positive_inputs[input_index] = perturb(input, element, epsilon)
            negative_inputs[input_index] = perturb(input, element, -epsilon)
            positive = objective(positive_inputs, gradient_outputs, options)
            negative = objective(negative_inputs, gradient_outputs, options)
            gradient[element] = (positive - negative) / (2 * epsilon)
          end
          gradient
        end

        def perturb(input, index, amount)
          flat = input.flatten.dup
          flat[index] += amount
          flat.reshape(*input.shape)
        end

        def objective(inputs, gradient_outputs, options)
          outputs = Backend::RubyEval3dRasterizer.forward(*inputs, **options)
          outputs.zip(gradient_outputs).sum { |output, gradient| (output * gradient).sum }.to_f
        end
      end
    end
  end
end
