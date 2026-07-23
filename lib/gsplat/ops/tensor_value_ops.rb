# frozen_string_literal: true

module Gsplat
  module Ops
    # Adds an offset and applies a lower clamp.
    class AddClampMin < Autograd::Function
      class << self
        def forward(context, value, offset:, minimum:)
          shifted = value + offset
          context.save(shifted.gt(minimum))
          below = shifted.lt(minimum)
          shifted[below] = minimum if below.any?
          shifted
        end

        # rubocop:disable Metrics/AbcSize
        def backward(context, gradient)
          mask = gradient.class.cast(context.saved_values.fetch(0))
          [gradient * mask]
        end
      end
    end

    # Elementwise multiplication with VJPs for both inputs.
    class Multiply < Autograd::Function
      class << self
        def forward(context, left, right)
          unless left.shape == right.shape
            raise ShapeError, "multiply shape mismatch: #{left.shape.inspect} and #{right.shape.inspect}"
          end

          context.save(left, right)
          left * right
        end

        def backward(context, gradient)
          left, right = context.saved_values
          [gradient * right, gradient * left]
        end
      end
    end

    # Divides one selected rendered depth channel by accumulated alpha.
    class NormalizeDepth < Autograd::Function
      class << self
        def forward(context, rendered, alphas, depth_index:)
          expected = rendered.shape[0...-1] + [1]
          unless alphas.shape == expected
            raise ShapeError, "expected alphas #{expected.inspect}, got #{alphas.shape.inspect}"
          end

          alpha = alphas[*Array.new(alphas.ndim - 1, true).push(0)]
          depth = rendered[*Array.new(rendered.ndim - 1, true).push(depth_index)]
          valid = alpha.gt(0)
          output = rendered.dup
          normalized = rendered.class.zeros(*alpha.shape)
          normalized[valid] = depth[valid] / alpha[valid] if valid.any?
          output[*Array.new(output.ndim - 1, true), depth_index] = normalized
          context.save(alpha, depth, valid, depth_index)
          output
        end

        def backward(context, gradient)
          alpha, depth, valid, depth_index = context.saved_values
          grad_rendered = gradient.dup
          grad_depth = gradient[*Array.new(gradient.ndim - 1, true).push(depth_index)]
          normalized_grad = gradient.class.zeros(*alpha.shape)
          normalized_grad[valid] = grad_depth[valid] / alpha[valid] if valid.any?
          grad_rendered[*Array.new(gradient.ndim - 1, true), depth_index] = normalized_grad
          grad_alpha = gradient.class.zeros(*alpha.shape)
          grad_alpha[valid] = -grad_depth[valid] * depth[valid] / (alpha[valid]**2) if valid.any?
          [grad_rendered, grad_alpha.reshape(*(alpha.shape + [1]))]
        end
        # rubocop:enable Metrics/AbcSize
      end
    end
  end
end
