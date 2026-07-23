# frozen_string_literal: true

require_relative "../backend/ruby/quat_scale_to_covar_preci"

# Differentiable quaternion/scale covariance operation.
module Gsplat
  module Ops
    # Differentiable conversion from wxyz quaternions/scales to covariance and precision.
    class QuatScaleToCovarPreci < Autograd::Function
      class << self
        # Applies the operation while retaining Python gsplat's two tuple positions.
        #
        # @param quaternions [Autograd::Variable, Numo::NArray] [...,4]
        # @param scales [Autograd::Variable, Numo::NArray] [...,3]
        # @param compute_covar [Boolean]
        # @param compute_preci [Boolean]
        # @param triu [Boolean] return [...,6] upper triangles when true
        # @return [Array<(Autograd::Variable, nil)>]
        def apply(quaternions, scales, compute_covar: true, compute_preci: true, triu: false)
          validate_selection!(compute_covar, compute_preci)
          outputs = super
          return outputs if compute_covar && compute_preci
          return [outputs, nil] if compute_covar

          [nil, outputs]
        end

        def forward(context, quaternions, scales, **options)
          compute_covar = options.fetch(:compute_covar)
          compute_preci = options.fetch(:compute_preci)
          triu = options.fetch(:triu)
          context.save(quaternions, scales, compute_covar, compute_preci, triu)
          covariance, precision = Backend.dispatch(
            :quat_scale_to_covar_preci_forward,
            quaternions,
            scales,
            compute_covar: compute_covar,
            compute_preci: compute_preci,
            triu: triu
          )
          return [covariance, precision] if compute_covar && compute_preci

          compute_covar ? covariance : precision
        end

        def backward(context, *grad_outputs)
          quaternions, scales, compute_covar, compute_preci, triu = context.saved_values
          grad_covar = compute_covar ? grad_outputs.shift : nil
          grad_preci = compute_preci ? grad_outputs.shift : nil
          Backend.dispatch(
            :quat_scale_to_covar_preci_backward,
            quaternions,
            scales,
            grad_covar,
            grad_preci,
            triu: triu
          )
        end

        private

        def validate_selection!(compute_covar, compute_preci)
          return if compute_covar || compute_preci

          raise ArgumentError, "at least one of compute_covar or compute_preci must be true"
        end
      end
    end
  end

  class << self
    # Converts wxyz quaternions and scales to covariance and/or precision matrices.
    #
    # @param quaternions [Autograd::Variable, Numo::NArray] [...,4]
    # @param scales [Autograd::Variable, Numo::NArray] [...,3]
    # @param compute_covar [Boolean]
    # @param compute_preci [Boolean]
    # @param triu [Boolean] return [...,6] upper triangles when true
    # @return [Array<(Autograd::Variable, Numo::NArray, nil)>]
    def quat_scale_to_covar_preci(quaternions, scales, compute_covar: true, compute_preci: true, triu: false)
      if [quaternions, scales].any?(Autograd::Variable)
        return Ops::QuatScaleToCovarPreci.apply(
          quaternions,
          scales,
          compute_covar: compute_covar,
          compute_preci: compute_preci,
          triu: triu
        )
      end

      raise ArgumentError, "at least one output must be requested" unless compute_covar || compute_preci

      Backend.dispatch(
        :quat_scale_to_covar_preci_forward,
        quaternions,
        scales,
        compute_covar: compute_covar,
        compute_preci: compute_preci,
        triu: triu
      )
    end
  end

  Backend.register(
    :quat_scale_to_covar_preci_forward,
    :ruby,
    Backend::RubyQuatScaleToCovarPreci.method(:forward)
  )
  Backend.register(
    :quat_scale_to_covar_preci_backward,
    :ruby,
    Backend::RubyQuatScaleToCovarPreci.method(:backward)
  )
end
