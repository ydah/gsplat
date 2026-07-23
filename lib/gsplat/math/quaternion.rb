# frozen_string_literal: true

module Gsplat
  module Math
    # Batched wxyz quaternion normalization, rotation conversion, and VJPs.
    module Quaternion
      module_function

      # Normalizes wxyz quaternions along the last dimension.
      #
      # @param quaternions [Numo::NArray] [...,4]
      # @param dtype [Class, Symbol, nil] optional float32/float64 calculation type
      # @param epsilon [Numeric, nil] minimum norm
      # @return [Numo::NArray] [...,4]
      def normalize(quaternions, dtype: nil, epsilon: nil)
        typed, leading_shape, flat = prepare(quaternions, 4, dtype)
        denominator = norm_denominator(flat, epsilon || default_epsilon(typed.class))
        (flat / denominator.reshape(flat.shape[0], 1)).reshape(*(leading_shape + [4]))
      end

      # VJP of quaternion normalization.
      #
      # @param quaternions [Numo::NArray] [...,4]
      # @param grad_output [Numo::NArray] [...,4]
      # @param dtype [Class, Symbol, nil] optional float32/float64 calculation type
      # @param epsilon [Numeric, nil] minimum norm
      # @return [Numo::NArray] [...,4]
      def normalize_vjp(quaternions, grad_output, dtype: nil, epsilon: nil)
        typed, leading_shape, flat = prepare(quaternions, 4, dtype)
        gradient, = prepare_matching(grad_output, leading_shape, 4, typed.class)
        denominator = norm_denominator(flat, epsilon || default_epsilon(typed.class))
        normalized = flat / denominator.reshape(flat.shape[0], 1)
        dot = (normalized * gradient).sum(axis: 1)
        result = (gradient - (normalized * dot.reshape(flat.shape[0], 1))) / denominator.reshape(flat.shape[0], 1)
        result.reshape(*(leading_shape + [4]))
      end

      # Converts wxyz quaternions to rotation matrices.
      #
      # @param quaternions [Numo::NArray] [...,4]
      # @param dtype [Class, Symbol, nil] optional float32/float64 calculation type
      # @return [Numo::NArray] [...,3,3]
      def to_rotmat(quaternions, dtype: nil)
        typed, leading_shape, = prepare(quaternions, 4, dtype)
        flat = normalize(typed).reshape(typed.size / 4, 4)
        quat_w = flat[true, 0]
        quat_x = flat[true, 1]
        quat_y = flat[true, 2]
        quat_z = flat[true, 3]
        output = typed.class.zeros(flat.shape[0], 3, 3)
        fill_rotation!(output, quat_w, quat_x, quat_y, quat_z)
        output.reshape(*(leading_shape + [3, 3]))
      end

      # VJP of quaternion-to-rotation-matrix conversion.
      #
      # @param quaternions [Numo::NArray] [...,4]
      # @param grad_output [Numo::NArray] [...,3,3]
      # @param dtype [Class, Symbol, nil] optional float32/float64 calculation type
      # @return [Numo::NArray] [...,4]
      def to_rotmat_vjp(quaternions, grad_output, dtype: nil)
        typed, leading_shape, = prepare(quaternions, 4, dtype)
        gradient = cast_float(grad_output, typed.class)
        unless gradient.shape == leading_shape + [3, 3]
          raise ShapeError, "expected gradient #{(leading_shape + [3, 3]).inspect}, got #{gradient.shape.inspect}"
        end

        normalized = normalize(typed).reshape(typed.size / 4, 4)
        grad_matrix = gradient.reshape(normalized.shape[0], 3, 3)
        grad_normalized = rotation_vjp(normalized, grad_matrix)
        normalize_vjp(typed, grad_normalized.reshape(*(leading_shape + [4])))
      end

      def prepare(quaternions, width, dtype)
        typed = cast_float(quaternions, dtype)
        unless typed.ndim >= 1 && typed.shape[-1] == width
          raise ShapeError, "expected [...,#{width}], got #{typed.shape.inspect}"
        end

        leading_shape = typed.shape[0...-1]
        [typed, leading_shape, typed.reshape(typed.size / width, width)]
      end
      private_class_method :prepare

      def prepare_matching(value, leading_shape, width, dtype)
        typed = cast_float(value, dtype)
        expected = leading_shape + [width]
        raise ShapeError, "expected #{expected.inspect}, got #{typed.shape.inspect}" unless typed.shape == expected

        [typed.reshape(typed.size / width, width), leading_shape]
      end
      private_class_method :prepare_matching

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
      private_class_method :cast_float

      def norm_denominator(flat, epsilon)
        norms = (flat**2).sum(axis: 1)**0.5
        norms[norms.lt(epsilon)] = epsilon
        norms
      end
      private_class_method :norm_denominator

      def default_epsilon(type)
        type == Numo::DFloat ? 1e-12 : 1e-6
      end
      private_class_method :default_epsilon

      # rubocop:disable Metrics/AbcSize
      def fill_rotation!(output, quat_w, quat_x, quat_y, quat_z)
        output[true, 0, 0] = 1 - (2 * ((quat_y**2) + (quat_z**2)))
        output[true, 0, 1] = 2 * ((quat_x * quat_y) - (quat_w * quat_z))
        output[true, 0, 2] = 2 * ((quat_x * quat_z) + (quat_w * quat_y))
        output[true, 1, 0] = 2 * ((quat_x * quat_y) + (quat_w * quat_z))
        output[true, 1, 1] = 1 - (2 * ((quat_x**2) + (quat_z**2)))
        output[true, 1, 2] = 2 * ((quat_y * quat_z) - (quat_w * quat_x))
        output[true, 2, 0] = 2 * ((quat_x * quat_z) - (quat_w * quat_y))
        output[true, 2, 1] = 2 * ((quat_y * quat_z) + (quat_w * quat_x))
        output[true, 2, 2] = 1 - (2 * ((quat_x**2) + (quat_y**2)))
      end
      # rubocop:enable Metrics/AbcSize
      private_class_method :fill_rotation!

      # rubocop:disable Metrics/AbcSize
      def rotation_vjp(quaternion, gradient)
        quat_w = quaternion[true, 0]
        quat_x = quaternion[true, 1]
        quat_y = quaternion[true, 2]
        quat_z = quaternion[true, 3]
        grad_at = ->(row, column) { gradient[true, row, column] }
        output = quaternion.class.zeros(*quaternion.shape)
        output[true, 0] = (
          (-2 * quat_z * grad_at.call(0, 1)) + (2 * quat_y * grad_at.call(0, 2)) +
          (2 * quat_z * grad_at.call(1, 0)) - (2 * quat_x * grad_at.call(1, 2)) -
          (2 * quat_y * grad_at.call(2, 0)) + (2 * quat_x * grad_at.call(2, 1))
        )
        output[true, 1] = (
          (2 * quat_y * grad_at.call(0, 1)) + (2 * quat_z * grad_at.call(0, 2)) +
          (2 * quat_y * grad_at.call(1, 0)) - (4 * quat_x * grad_at.call(1, 1)) -
          (2 * quat_w * grad_at.call(1, 2)) + (2 * quat_z * grad_at.call(2, 0)) +
          (2 * quat_w * grad_at.call(2, 1)) - (4 * quat_x * grad_at.call(2, 2))
        )
        output[true, 2] = (
          (-4 * quat_y * grad_at.call(0, 0)) + (2 * quat_x * grad_at.call(0, 1)) +
          (2 * quat_w * grad_at.call(0, 2)) + (2 * quat_x * grad_at.call(1, 0)) +
          (2 * quat_z * grad_at.call(1, 2)) - (2 * quat_w * grad_at.call(2, 0)) +
          (2 * quat_z * grad_at.call(2, 1)) - (4 * quat_y * grad_at.call(2, 2))
        )
        output[true, 3] = (
          (-4 * quat_z * grad_at.call(0, 0)) - (2 * quat_w * grad_at.call(0, 1)) +
          (2 * quat_x * grad_at.call(0, 2)) + (2 * quat_w * grad_at.call(1, 0)) -
          (4 * quat_z * grad_at.call(1, 1)) + (2 * quat_y * grad_at.call(1, 2)) +
          (2 * quat_x * grad_at.call(2, 0)) + (2 * quat_y * grad_at.call(2, 1))
        )
        output
      end
      # rubocop:enable Metrics/AbcSize
      private_class_method :rotation_vjp
    end
  end
end
