# frozen_string_literal: true

module Gsplat
  module Backend
    # Numo forward/backward implementation of real spherical harmonics.
    module RubySphericalHarmonics
      module_function

      def forward(degree, directions, coefficients, masks: nil)
        directions, coefficients, masks, leading_shape = validate_inputs(degree, directions, coefficients, masks)
        count = directions.size / 3
        basis_count = coefficients.shape[-2]
        channel_count = coefficients.shape[-1]
        normalized = normalize_directions(directions).reshape(count, 3)
        bases, = Math::SphericalHarmonicBasis.evaluate(normalized, degree, basis_count)
        apply_masks!(bases, nil, masks)
        colors = (
          bases.reshape(count, basis_count, 1) *
          coefficients.reshape(count, basis_count, channel_count)
        ).sum(axis: 1)
        colors.reshape(*(leading_shape + [channel_count]))
      end

      # rubocop:disable Metrics/ParameterLists
      def backward(degree, directions, coefficients, grad_output, masks: nil, grad_dirs: true, grad_coeffs: true)
        # rubocop:enable Metrics/ParameterLists
        directions, coefficients, masks, leading_shape = validate_inputs(degree, directions, coefficients, masks)
        count = directions.size / 3
        basis_count = coefficients.shape[-2]
        channel_count = coefficients.shape[-1]
        gradient = directions.class.cast(grad_output)
        expected = leading_shape + [channel_count]
        unless gradient.shape == expected
          raise ShapeError, "expected output gradient #{expected.inspect}, got #{gradient.shape.inspect}"
        end

        normalized = normalize_directions(directions).reshape(count, 3)
        bases, derivatives = Math::SphericalHarmonicBasis.evaluate(normalized, degree, basis_count)
        apply_masks!(bases, derivatives, masks)
        gradient = gradient.reshape(count, channel_count)
        coefficient_values = coefficients.reshape(count, basis_count, channel_count)
        coefficient_gradient = if grad_coeffs
                                 bases.reshape(count, basis_count, 1) * gradient.reshape(count, 1, channel_count)
                               end
        if grad_dirs
          direction_gradient = direction_gradient(
            directions,
            coefficient_values,
            gradient,
            derivatives,
            leading_shape
          )
        end
        [
          direction_gradient,
          coefficient_gradient&.reshape(*(leading_shape + [basis_count, channel_count]))
        ]
      end

      def validate_inputs(degree, directions, coefficients, masks)
        raise ArgumentError, "degree must be an integer from 0 through 4" unless (0..4).cover?(degree)

        validate_array_types!(directions, coefficients, masks)
        leading_shape = directions.shape[0...-1]
        valid_shapes = directions.ndim >= 1 && directions.shape[-1] == 3 &&
                       coefficients.ndim >= 2 &&
                       coefficients.shape[0...-2] == leading_shape &&
                       coefficients.shape[-2] >= ((degree + 1)**2)
        unless valid_shapes
          raise ShapeError,
                "expected directions [...,3] and coefficients [...,K,D] with K >= #{(degree + 1)**2}, " \
                "got #{directions.shape.inspect} and #{coefficients.shape.inspect}"
        end
        if masks && masks.shape != leading_shape
          raise ShapeError, "expected masks #{leading_shape.inspect}, got #{masks.shape.inspect}"
        end

        [directions, directions.class.cast(coefficients), masks && Numo::Bit.cast(masks), leading_shape]
      end
      private_class_method :validate_inputs

      def validate_array_types!(directions, coefficients, masks)
        unless directions.is_a?(Numo::NArray) && [Numo::SFloat, Numo::DFloat].include?(directions.class)
          raise ArgumentError, "directions must be Numo::SFloat or Numo::DFloat"
        end
        raise ArgumentError, "coefficients must be a Numo::NArray" unless coefficients.is_a?(Numo::NArray)
        raise ArgumentError, "masks must be a Numo::NArray" if masks && !masks.is_a?(Numo::NArray)
      end
      private_class_method :validate_array_types!

      def normalize_directions(directions)
        flat = directions.reshape(directions.size / 3, 3)
        norms = (flat**2).sum(axis: 1)**0.5
        norms[norms.lt(1e-12)] = 1e-12
        (flat / norms.reshape(flat.shape[0], 1)).reshape(*directions.shape)
      end
      private_class_method :normalize_directions

      def apply_masks!(bases, derivatives, masks)
        return unless masks

        hidden = masks.reshape(masks.size).eq(0)
        bases[hidden, true] = 0
        derivatives[hidden, true, true] = 0 if derivatives
      end
      private_class_method :apply_masks!

      def direction_gradient(directions, coefficients, gradient, derivatives, leading_shape)
        count = directions.size / 3
        basis_count = coefficients.shape[1]
        grad_bases = (coefficients * gradient.reshape(count, 1, gradient.shape[1])).sum(axis: 2)
        grad_normalized = (derivatives * grad_bases.reshape(count, basis_count, 1)).sum(axis: 1)
        normalized_direction_vjp(
          directions,
          grad_normalized.reshape(*(leading_shape + [3]))
        )
      end
      private_class_method :direction_gradient

      def normalized_direction_vjp(directions, gradient)
        flat = directions.reshape(directions.size / 3, 3)
        grad_flat = gradient.reshape(flat.shape[0], 3)
        norms = (flat**2).sum(axis: 1)**0.5
        clamped = norms.lt(1e-12)
        norms[clamped] = 1e-12
        normalized = flat / norms.reshape(flat.shape[0], 1)
        dot = (normalized * grad_flat).sum(axis: 1)
        result = (grad_flat - (normalized * dot.reshape(flat.shape[0], 1))) / norms.reshape(flat.shape[0], 1)
        result[clamped, true] = grad_flat[clamped, true] / 1e-12 if clamped.any?
        result.reshape(*directions.shape)
      end
      private_class_method :normalized_direction_vjp
    end
  end
end
