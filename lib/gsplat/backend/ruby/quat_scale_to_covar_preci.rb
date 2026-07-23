# frozen_string_literal: true

module Gsplat
  module Backend
    # Numo implementation of quaternion/scale covariance conversion.
    module RubyQuatScaleToCovarPreci
      module_function

      def forward(quaternions, scales, compute_covar:, compute_preci:, triu:)
        quaternions, scales, leading_shape = validate_inputs(quaternions, scales, compute_preci)
        count = quaternions.size / 4
        rotations = Math::Quaternion.to_rotmat(quaternions).reshape(count, 3, 3)
        scales_flat = scales.reshape(count, 3)
        covariance = covariance_matrix(rotations, scales_flat) if compute_covar
        precision = precision_matrix(rotations, scales_flat) if compute_preci
        [
          format_output(covariance, leading_shape, triu),
          format_output(precision, leading_shape, triu)
        ]
      end

      def backward(quaternions, scales, grad_covar, grad_preci, triu:)
        quaternions, scales, leading_shape = validate_inputs(quaternions, scales, !grad_preci.nil?)
        count = quaternions.size / 4
        rotations = Math::Quaternion.to_rotmat(quaternions).reshape(count, 3, 3)
        scales_flat = scales.reshape(count, 3)
        grad_rotations = quaternions.class.zeros(count, 3, 3)
        grad_scales = quaternions.class.zeros(count, 3)
        if grad_covar
          accumulate_covar_gradients!(
            grad_rotations,
            grad_scales,
            rotations,
            scales_flat,
            expand_gradient(grad_covar, leading_shape, triu, quaternions.class)
          )
        end
        if grad_preci
          accumulate_preci_gradients!(
            grad_rotations,
            grad_scales,
            rotations,
            scales_flat,
            expand_gradient(grad_preci, leading_shape, triu, quaternions.class)
          )
        end
        grad_quaternions = Math::Quaternion.to_rotmat_vjp(
          quaternions,
          grad_rotations.reshape(*(leading_shape + [3, 3]))
        )
        [grad_quaternions, grad_scales.reshape(*(leading_shape + [3]))]
      end

      def validate_inputs(quaternions, scales, compute_preci)
        unless quaternions.is_a?(Numo::NArray) && [Numo::SFloat, Numo::DFloat].include?(quaternions.class)
          raise ArgumentError, "quaternions must be Numo::SFloat or Numo::DFloat"
        end

        expected_scales = quaternions.shape[0...-1] + [3]
        unless quaternions.ndim >= 1 && quaternions.shape[-1] == 4 && scales.shape == expected_scales
          raise ShapeError,
                "expected quaternions [...,4] and scales #{expected_scales.inspect}, " \
                "got #{quaternions.shape.inspect} and #{scales.shape.inspect}"
        end

        scales = quaternions.class.cast(scales)
        raise Gsplat::Error, "precision is undefined for zero scales" if compute_preci && scales.eq(0).any?

        [quaternions, scales, quaternions.shape[0...-1]]
      end
      private_class_method :validate_inputs

      def covariance_matrix(rotations, scales)
        transform = rotations * scales.reshape(scales.shape[0], 1, 3)
        Math::Mat.matmul_batch(transform, transform.transpose(0, 2, 1))
      end
      private_class_method :covariance_matrix

      def precision_matrix(rotations, scales)
        transform = rotations * (1.0 / scales).reshape(scales.shape[0], 1, 3)
        Math::Mat.matmul_batch(transform, transform.transpose(0, 2, 1))
      end
      private_class_method :precision_matrix

      def format_output(matrix, leading_shape, triu)
        return nil unless matrix
        return matrix.reshape(*(leading_shape + [3, 3])) unless triu

        output = matrix.class.zeros(matrix.shape[0], 6)
        [[0, 0], [0, 1], [0, 2], [1, 1], [1, 2], [2, 2]].each_with_index do |(row, column), index|
          output[true, index] = matrix[true, row, column]
        end
        output.reshape(*(leading_shape + [6]))
      end
      private_class_method :format_output

      def expand_gradient(gradient, leading_shape, triu, type)
        gradient = type.cast(gradient)
        expected = leading_shape + [triu ? 6 : 3, triu ? nil : 3].compact
        unless gradient.shape == expected
          raise ShapeError, "expected gradient #{expected.inspect}, got #{gradient.shape.inspect}"
        end
        return gradient.reshape(gradient.size / 9, 3, 3) unless triu

        flat = gradient.reshape(gradient.size / 6, 6)
        output = type.zeros(flat.shape[0], 3, 3)
        [[0, 0], [1, 1], [2, 2]].each_with_index do |(row, column), diagonal_index|
          output[true, row, column] = flat[true, [0, 3, 5].fetch(diagonal_index)]
        end
        [[0, 1, 1], [0, 2, 2], [1, 2, 4]].each do |row, column, index|
          half = flat[true, index] * 0.5
          output[true, row, column] = half
          output[true, column, row] = half
        end
        output
      end
      private_class_method :expand_gradient

      def accumulate_covar_gradients!(grad_rotations, grad_scales, rotations, scales, gradient)
        symmetric = gradient + gradient.transpose(0, 2, 1)
        transform = rotations * scales.reshape(scales.shape[0], 1, 3)
        grad_transform = Math::Mat.matmul_batch(symmetric, transform)
        grad_rotations[] = grad_rotations + (grad_transform * scales.reshape(scales.shape[0], 1, 3))
        grad_scales[] = grad_scales + (grad_transform * rotations).sum(axis: 1)
      end
      private_class_method :accumulate_covar_gradients!

      def accumulate_preci_gradients!(grad_rotations, grad_scales, rotations, scales, gradient)
        inverse_scales = 1.0 / scales
        symmetric = gradient + gradient.transpose(0, 2, 1)
        transform = rotations * inverse_scales.reshape(scales.shape[0], 1, 3)
        grad_transform = Math::Mat.matmul_batch(symmetric, transform)
        grad_rotations[] = grad_rotations + (grad_transform * inverse_scales.reshape(scales.shape[0], 1, 3))
        grad_scales[] = grad_scales - ((grad_transform * rotations).sum(axis: 1) / (scales**2))
      end
      private_class_method :accumulate_preci_gradients!
    end
  end
end
