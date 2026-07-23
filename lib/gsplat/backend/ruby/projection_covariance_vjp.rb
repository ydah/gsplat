# frozen_string_literal: true

module Gsplat
  module Backend
    # Covariance-output VJP helpers for RubyProjectionBackward.
    module RubyProjectionBackward
      module_function

      # rubocop:disable Metrics/AbcSize
      def covariance_output_vjp(covars2d, grad_conics, grad_compensations, eps2d)
        regularized = covars2d.dup
        regularized[true, true, 0, 0] += eps2d
        regularized[true, true, 1, 1] += eps2d
        grad_regularized = conic_vjp(regularized, grad_conics)
        return grad_regularized unless grad_compensations

        original_det = Math::Mat.det2x2(covars2d)
        regularized_det = Math::Mat.det2x2(regularized)
        ratio = original_det / regularized_det
        positive = ratio.gt(0)
        compensation = ratio.class.zeros(*ratio.shape)
        compensation[positive] = ratio[positive]**0.5 if positive.any?
        grad_original_det = ratio.class.zeros(*ratio.shape)
        if positive.any?
          grad_original_det[positive] = grad_compensations[positive] /
                                        (2 * compensation[positive] * regularized_det[positive])
        end
        grad_regularized_det = -grad_compensations * compensation / (2 * regularized_det)
        grad_regularized + determinant_vjp(regularized, grad_regularized_det) +
          determinant_vjp(covars2d, grad_original_det)
      end
      # rubocop:enable Metrics/AbcSize
      private_class_method :covariance_output_vjp

      # rubocop:disable Metrics/AbcSize
      def conic_vjp(covariance, gradient)
        count = gradient.size / 3
        matrices = covariance.reshape(count, 2, 2)
        determinant = Math::Mat.det2x2(matrices)
        inverse = matrices.class.zeros(count, 2, 2)
        inverse[true, 0, 0] = matrices[true, 1, 1] / determinant
        inverse[true, 0, 1] = -matrices[true, 0, 1] / determinant
        inverse[true, 1, 0] = -matrices[true, 1, 0] / determinant
        inverse[true, 1, 1] = matrices[true, 0, 0] / determinant
        grad_flat = gradient.reshape(count, 3)
        grad_inverse = matrices.class.zeros(count, 2, 2)
        grad_inverse[true, 0, 0] = grad_flat[true, 0]
        grad_inverse[true, 0, 1] = grad_flat[true, 1] * 0.5
        grad_inverse[true, 1, 0] = grad_flat[true, 1] * 0.5
        grad_inverse[true, 1, 1] = grad_flat[true, 2]
        inverse_transpose = inverse.transpose(0, 2, 1)
        result = -Math::Mat.matmul_batch(
          Math::Mat.matmul_batch(inverse_transpose, grad_inverse),
          inverse_transpose
        )
        result.reshape(*covariance.shape)
      end
      # rubocop:enable Metrics/AbcSize
      private_class_method :conic_vjp

      def determinant_vjp(matrix, gradient)
        output = matrix.class.zeros(*matrix.shape)
        output[true, true, 0, 0] = gradient * matrix[true, true, 1, 1]
        output[true, true, 0, 1] = -gradient * matrix[true, true, 1, 0]
        output[true, true, 1, 0] = -gradient * matrix[true, true, 0, 1]
        output[true, true, 1, 1] = gradient * matrix[true, true, 0, 0]
        output
      end
      private_class_method :determinant_vjp
    end
  end
end
