# frozen_string_literal: true

module Gsplat
  module Math
    # Internal shape, casting, and closed-form primitives for Mat.
    module SmallMatrixPrimitives
      private

      def prepare_matrix(matrices, size, dtype)
        typed = cast_float(matrices, dtype)
        validate_rank!(typed, 2)
        unless typed.shape[-2, 2] == [size, size]
          raise ShapeError, "expected [...,#{size},#{size}], got #{typed.shape.inspect}"
        end

        leading_shape = typed.shape[0...-2]
        batch_size = leading_shape.empty? ? 1 : leading_shape.inject(:*)
        [typed, leading_shape, typed.reshape(batch_size, size, size)]
      end

      def cast_float(array, dtype)
        raise ArgumentError, "expected a Numo::NArray" unless array.is_a?(Numo::NArray)

        type = case dtype
               when nil then array.class
               when :float32 then Numo::SFloat
               when :float64 then Numo::DFloat
               else dtype
               end
        unless [Numo::SFloat, Numo::DFloat].include?(type)
          raise ArgumentError, "dtype must be Numo::SFloat or Numo::DFloat"
        end

        type.cast(array)
      end

      def validate_rank!(array, trailing_dimensions)
        return if array.ndim >= trailing_dimensions

        raise ShapeError, "expected at least #{trailing_dimensions} dimensions, got #{array.shape.inspect}"
      end

      def reshape_vector(vector, leading_shape)
        return vector.class.cast(vector[0]) if leading_shape.empty?

        vector.reshape(*leading_shape)
      end

      def reshape_matrix(matrix, leading_shape, *trailing_shape)
        matrix.reshape(*(leading_shape + trailing_shape))
      end

      def ensure_invertible!(determinant, epsilon)
        return unless determinant.abs.le(epsilon).any?

        raise Gsplat::Error, "matrix is singular within epsilon #{epsilon}"
      end

      def default_epsilon(type)
        type == Numo::DFloat ? 1e-12 : 1e-6
      end

      def determinant3(matrix)
        a = matrix[true, 0, 0]
        b = matrix[true, 0, 1]
        c = matrix[true, 0, 2]
        d = matrix[true, 1, 0]
        e = matrix[true, 1, 1]
        f = matrix[true, 1, 2]
        g = matrix[true, 2, 0]
        h = matrix[true, 2, 1]
        i = matrix[true, 2, 2]
        (a * ((e * i) - (f * h))) - (b * ((d * i) - (f * g))) + (c * ((d * h) - (e * g)))
      end

      # rubocop:disable Metrics/AbcSize
      def fill_inverse3!(output, matrix, determinant)
        a = matrix[true, 0, 0]
        b = matrix[true, 0, 1]
        c = matrix[true, 0, 2]
        d = matrix[true, 1, 0]
        e = matrix[true, 1, 1]
        f = matrix[true, 1, 2]
        g = matrix[true, 2, 0]
        h = matrix[true, 2, 1]
        i = matrix[true, 2, 2]
        output[true, 0, 0] = ((e * i) - (f * h)) / determinant
        output[true, 0, 1] = ((c * h) - (b * i)) / determinant
        output[true, 0, 2] = ((b * f) - (c * e)) / determinant
        output[true, 1, 0] = ((f * g) - (d * i)) / determinant
        output[true, 1, 1] = ((a * i) - (c * g)) / determinant
        output[true, 1, 2] = ((c * d) - (a * f)) / determinant
        output[true, 2, 0] = ((d * h) - (e * g)) / determinant
        output[true, 2, 1] = ((b * g) - (a * h)) / determinant
        output[true, 2, 2] = ((a * e) - (b * d)) / determinant
      end
      # rubocop:enable Metrics/AbcSize

      def multiply_flat(left, right, rows, shared, columns)
        output = left.class.zeros(left.shape[0], rows, columns)
        rows.times do |row|
          columns.times do |column|
            shared.times do |inner|
              output[true, row, column] += left[true, row, inner] * right[true, inner, column]
            end
          end
        end
        output
      end
    end
  end
end
