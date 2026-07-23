# frozen_string_literal: true

require_relative "../backend/ruby/spherical_harmonics"

# Differentiable real spherical harmonics operation.
module Gsplat
  module Ops
    # Evaluates real SH bases through degree four.
    class SphericalHarmonics < Autograd::Function
      class << self
        # Evaluates SH and stores inputs for its analytic VJP.
        # @api private
        def forward(context, degree, directions, coefficients, masks: nil)
          context.save(degree, directions, coefficients, masks)
          Backend.dispatch(
            :spherical_harmonics_forward,
            degree,
            directions,
            coefficients,
            masks: masks
          )
        end

        # Propagates an SH value gradient to directions and coefficients.
        # @api private
        def backward(context, grad_output)
          degree, directions, coefficients, masks = context.saved_values
          grad_directions, grad_coefficients = Backend.dispatch(
            :spherical_harmonics_backward,
            degree,
            directions,
            coefficients,
            grad_output,
            masks: masks,
            grad_dirs: context.needs_input_grad.fetch(1),
            grad_coeffs: context.needs_input_grad.fetch(2)
          )
          [nil, grad_directions, grad_coefficients]
        end
      end
    end
  end

  class << self
    # Evaluates real spherical harmonics.
    #
    # @param degree [Integer] degree 0..4
    # @param directions [Autograd::Variable, Numo::NArray] [...,3]
    # @param coefficients [Autograd::Variable, Numo::NArray] [...,K,D], K >= (degree+1)^2
    # @param masks [Numo::Bit, nil] [...]
    # @return [Autograd::Variable, Numo::NArray] [...,D]
    def spherical_harmonics(degree, directions, coefficients, masks: nil)
      if [directions, coefficients].any?(Autograd::Variable)
        return Ops::SphericalHarmonics.apply(degree, directions, coefficients, masks: masks)
      end

      Backend.dispatch(:spherical_harmonics_forward, degree, directions, coefficients, masks: masks)
    end
  end

  Backend.register(
    :spherical_harmonics_forward,
    :ruby,
    Backend::RubySphericalHarmonics.method(:forward)
  )
  Backend.register(
    :spherical_harmonics_backward,
    :ruby,
    Backend::RubySphericalHarmonics.method(:backward)
  )
end
