# frozen_string_literal: true

module Gsplat
  module Training
    # Differentiable image losses and scalar image-quality metrics.
    module Losses
      # Mean absolute error with a zero subgradient at exact equality.
      class L1 < Autograd::Function
        class << self
          def forward(context, prediction, target)
            unless prediction.shape == target.shape
              raise ShapeError, "L1 shapes differ: #{prediction.shape.inspect} and #{target.shape.inspect}"
            end

            difference = prediction - target
            context.save(difference, prediction.size)
            prediction.class.cast(difference.abs.mean)
          end

          def backward(context, grad_output)
            difference, count = context.saved_values
            signs = difference.class.zeros(*difference.shape)
            signs[difference.gt(0)] = 1
            signs[difference.lt(0)] = -1
            gradient = signs * (grad_output.to_f / count)
            [gradient, -gradient]
          end
        end
      end

      # Structural similarity with the standard 11x11, sigma 1.5 window.
      class StructuralSimilarity < Autograd::Function
        class << self
          def forward(context, image_a, image_b, layout:)
            score, cache = Math::Ssim.forward(image_a, image_b, layout: layout)
            context.save(cache)
            image_a.class.cast(score)
          end

          def backward(context, grad_output)
            Math::Ssim.backward(context.saved_values.first, grad_output)
          end
        end
      end

      module_function

      # Returns mean absolute error, retaining an autograd graph when needed.
      def l1(prediction, target)
        return L1.apply(prediction, target) if [prediction, target].any?(Autograd::Variable)

        L1.forward(Autograd::Context.new([false, false], [prediction, target]), prediction, target)
      end

      # Returns mean SSIM over batches, channels, and pixels.
      def ssim(image_a, image_b, layout: :auto)
        if [image_a, image_b].any?(Autograd::Variable)
          return StructuralSimilarity.apply(image_a, image_b, layout: layout)
        end

        Math::Ssim.forward(image_a, image_b, layout: layout).first
      end

      # Returns peak signal-to-noise ratio in decibels.
      def psnr(prediction, target, max_value: 1.0)
        prediction = Ops::TensorOps.data(prediction)
        target = Ops::TensorOps.data(target)
        unless prediction.shape == target.shape
          raise ShapeError, "PSNR shapes differ: #{prediction.shape.inspect} and #{target.shape.inspect}"
        end
        raise ArgumentError, "max_value must be positive" unless max_value.positive?

        mean_squared_error = ((prediction - target)**2).mean.to_f
        return Float::INFINITY if mean_squared_error.zero?

        10 * ::Math.log10((max_value**2) / mean_squared_error)
      end
    end
  end
end
