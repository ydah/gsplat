# frozen_string_literal: true

require_relative "small_matrix_primitives"

module Gsplat
  # Quaternion, projection, image metric, and small-matrix primitives.
  module Math
    # Closed-form batched operations for small matrices.
    module Mat
      extend SmallMatrixPrimitives

      module_function

      # Determinant of batched 2x2 matrices.
      #
      # @param matrices [Numo::NArray] [...,2,2]
      # @param dtype [Class, Symbol, nil] optional float32/float64 calculation type
      # @return [Numo::NArray] [...]
      def det2x2(matrices, dtype: nil)
        _, leading_shape, flat = prepare_matrix(matrices, 2, dtype)
        determinant = (flat[true, 0, 0] * flat[true, 1, 1]) - (flat[true, 0, 1] * flat[true, 1, 0])
        reshape_vector(determinant, leading_shape)
      end

      # Inverse of batched 2x2 matrices.
      #
      # @param matrices [Numo::NArray] [...,2,2]
      # @param dtype [Class, Symbol, nil] optional float32/float64 calculation type
      # @param epsilon [Numeric, nil] singularity threshold
      # @return [Numo::NArray] [...,2,2]
      def inv2x2(matrices, dtype: nil, epsilon: nil)
        typed, leading_shape, flat = prepare_matrix(matrices, 2, dtype)
        determinant = (flat[true, 0, 0] * flat[true, 1, 1]) - (flat[true, 0, 1] * flat[true, 1, 0])
        ensure_invertible!(determinant, epsilon || default_epsilon(typed.class))
        output = typed.class.zeros(flat.shape[0], 2, 2)
        output[true, 0, 0] = flat[true, 1, 1] / determinant
        output[true, 0, 1] = -flat[true, 0, 1] / determinant
        output[true, 1, 0] = -flat[true, 1, 0] / determinant
        output[true, 1, 1] = flat[true, 0, 0] / determinant
        reshape_matrix(output, leading_shape, 2, 2)
      end

      # Ordered real eigenvalues of batched 2x2 matrices.
      #
      # @param matrices [Numo::NArray] [...,2,2]
      # @param dtype [Class, Symbol, nil] optional float32/float64 calculation type
      # @return [Numo::NArray] [...,2] in ascending order
      def eigvals2x2(matrices, dtype: nil)
        typed, leading_shape, flat = prepare_matrix(matrices, 2, dtype)
        half_trace = (flat[true, 0, 0] + flat[true, 1, 1]) * 0.5
        half_difference = (flat[true, 0, 0] - flat[true, 1, 1]) * 0.5
        discriminant = (half_difference**2) + (flat[true, 0, 1] * flat[true, 1, 0])
        discriminant[discriminant.lt(0)] = 0
        root = discriminant**0.5
        output = typed.class.zeros(flat.shape[0], 2)
        output[true, 0] = half_trace - root
        output[true, 1] = half_trace + root
        reshape_matrix(output, leading_shape, 2)
      end

      # Determinant of batched 3x3 matrices.
      #
      # @param matrices [Numo::NArray] [...,3,3]
      # @param dtype [Class, Symbol, nil] optional float32/float64 calculation type
      # @return [Numo::NArray] [...]
      def det3x3(matrices, dtype: nil)
        _, leading_shape, flat = prepare_matrix(matrices, 3, dtype)
        determinant = determinant3(flat)
        reshape_vector(determinant, leading_shape)
      end

      # Inverse of batched 3x3 matrices.
      #
      # @param matrices [Numo::NArray] [...,3,3]
      # @param dtype [Class, Symbol, nil] optional float32/float64 calculation type
      # @param epsilon [Numeric, nil] singularity threshold
      # @return [Numo::NArray] [...,3,3]
      def inv3x3(matrices, dtype: nil, epsilon: nil)
        typed, leading_shape, flat = prepare_matrix(matrices, 3, dtype)
        determinant = determinant3(flat)
        ensure_invertible!(determinant, epsilon || default_epsilon(typed.class))
        output = typed.class.zeros(flat.shape[0], 3, 3)
        fill_inverse3!(output, flat, determinant)
        reshape_matrix(output, leading_shape, 3, 3)
      end

      # Batched matrix multiplication for small matrices.
      #
      # @param left [Numo::NArray] [...,M,K]
      # @param right [Numo::NArray] [...,K,N]
      # @param dtype [Class, Symbol, nil] optional float32/float64 calculation type
      # @return [Numo::NArray] [...,M,N]
      def matmul_batch(left, right, dtype: nil)
        left = cast_float(left, dtype)
        right = cast_float(right, left.class)
        validate_rank!(left, 2)
        validate_rank!(right, 2)
        leading_shape = left.shape[0...-2]
        unless right.shape[0...-2] == leading_shape && left.shape[-1] == right.shape[-2]
          raise ShapeError, "matmul shape mismatch: left #{left.shape.inspect}, right #{right.shape.inspect}"
        end

        rows = left.shape[-2]
        shared = left.shape[-1]
        columns = right.shape[-1]
        batch_size = leading_shape.empty? ? 1 : leading_shape.inject(:*)
        left_flat = left.reshape(batch_size, rows, shared)
        right_flat = right.reshape(batch_size, shared, columns)
        output = multiply_flat(left_flat, right_flat, rows, shared, columns)
        reshape_matrix(output, leading_shape, rows, columns)
      end
    end
  end
end
