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

      # Weighted L1 plus (1 - SSIM) reconstruction objective.
      class Reconstruction < Autograd::Function
        class << self
          def forward(context, prediction, target, ssim_lambda:, layout:)
            unless prediction.shape == target.shape
              raise ShapeError, "reconstruction shapes differ: #{prediction.shape.inspect} and #{target.shape.inspect}"
            end

            difference = prediction - target
            score, ssim_cache = Math::Ssim.forward(prediction, target, layout: layout)
            context.save(difference, prediction.size, ssim_cache, ssim_lambda)
            prediction.class.cast(((1 - ssim_lambda) * difference.abs.mean.to_f) + (ssim_lambda * (1 - score)))
          end

          def backward(context, grad_output)
            difference, count, ssim_cache, ssim_lambda = context.saved_values
            signs = difference.class.zeros(*difference.shape)
            signs[difference.gt(0)] = 1
            signs[difference.lt(0)] = -1
            l1_gradient = signs * ((1 - ssim_lambda) * grad_output.to_f / count)
            ssim_a, ssim_b = Math::Ssim.backward(ssim_cache, grad_output)
            [l1_gradient - (ssim_lambda * ssim_a), -l1_gradient - (ssim_lambda * ssim_b)]
          end
        end
      end

      # Reconstruction loss with activated opacity and scale mean penalties.
      class RegularizedReconstruction < Autograd::Function
        class << self
          # rubocop:disable Metrics/ParameterLists
          def forward(context, prediction, target, opacities, scales, ssim_lambda:, opacity_reg:, scale_reg:, layout:)
            # rubocop:enable Metrics/ParameterLists
            difference = prediction - target
            score, ssim_cache = Math::Ssim.forward(prediction, target, layout: layout)
            context.save(
              difference, prediction.size, ssim_cache, ssim_lambda,
              opacities.shape, scales.shape, opacity_reg, scale_reg
            )
            reconstruction = ((1 - ssim_lambda) * difference.abs.mean.to_f) + (ssim_lambda * (1 - score))
            prediction.class.cast(
              reconstruction + (opacity_reg * opacities.mean.to_f) + (scale_reg * scales.mean.to_f)
            )
          end

          # rubocop:disable Metrics/AbcSize
          def backward(context, grad_output)
            difference, count, cache, ssim_lambda,
              opacity_shape, scale_shape, opacity_reg, scale_reg = context.saved_values
            signs = difference.class.zeros(*difference.shape)
            signs[difference.gt(0)] = 1
            signs[difference.lt(0)] = -1
            l1_gradient = signs * ((1 - ssim_lambda) * grad_output.to_f / count)
            ssim_a, ssim_b = Math::Ssim.backward(cache, grad_output)
            opacity_gradient = difference.class.ones(*opacity_shape) *
                               opacity_reg * grad_output.to_f / opacity_shape.inject(:*)
            scale_gradient = difference.class.ones(*scale_shape) *
                             scale_reg * grad_output.to_f / scale_shape.inject(:*)
            [
              l1_gradient - (ssim_lambda * ssim_a),
              -l1_gradient - (ssim_lambda * ssim_b),
              opacity_gradient,
              scale_gradient
            ]
          end
          # rubocop:enable Metrics/AbcSize
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

      # Returns `(1-lambda)*L1 + lambda*(1-SSIM)`.
      def reconstruction(prediction, target, ssim_lambda: 0.2, layout: :auto)
        raise ArgumentError, "ssim_lambda must be between 0 and 1" unless ssim_lambda.between?(0.0, 1.0)

        if [prediction, target].any?(Autograd::Variable)
          return Reconstruction.apply(
            prediction,
            target,
            ssim_lambda: ssim_lambda,
            layout: layout
          )
        end

        l1_value = (prediction - target).abs.mean.to_f
        ((1 - ssim_lambda) * l1_value) + (ssim_lambda * (1 - ssim(prediction, target, layout: layout)))
      end

      # Reconstruction objective plus optional MCMC opacity and scale regularizers.
      def regularized_reconstruction(prediction, target, opacities, scales, **options)
        RegularizedReconstruction.apply(
          prediction,
          target,
          opacities,
          scales,
          ssim_lambda: options.fetch(:ssim_lambda, 0.2),
          opacity_reg: options.fetch(:opacity_reg, 0.0),
          scale_reg: options.fetch(:scale_reg, 0.0),
          layout: options.fetch(:layout, :auto)
        )
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
