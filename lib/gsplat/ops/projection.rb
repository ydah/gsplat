# frozen_string_literal: true

require_relative "../backend/ruby/projection"
require_relative "../backend/ruby/projection_backward"
require_relative "../backend/ruby/projection_covariance_vjp"
require_relative "../backend/ruby/projection_input_vjp"

# Differentiable camera projection operations.
module Gsplat
  module Ops
    # Fused world-to-camera transform, covariance projection, and culling.
    class FullyFusedProjection < Autograd::Function
      class << self
        # Adds the optional compensation placeholder used by the public tuple.
        # @api private
        def apply(*inputs, calc_compensations: false, **options)
          outputs = super
          calc_compensations ? outputs : [*outputs, nil]
        end

        # rubocop:disable Metrics/ParameterLists
        def forward(context, means, covars, quaternions, scales, viewmats, intrinsics, width, height, **options)
          # rubocop:enable Metrics/ParameterLists
          context.save(means, covars, quaternions, scales, viewmats, intrinsics, width, height, options)
          outputs = Backend.dispatch(
            :fully_fused_projection_forward,
            means,
            covars,
            quaternions,
            scales,
            viewmats,
            intrinsics,
            width,
            height,
            **options
          )
          options.fetch(:calc_compensations) ? outputs : outputs.first(4)
        end

        # rubocop:disable Metrics/ParameterLists
        def backward(context, _grad_radii, grad_means2d, grad_depths, grad_conics, grad_compensations = nil)
          # rubocop:enable Metrics/ParameterLists
          means, covars, quaternions, scales, viewmats, intrinsics, width, height, options = context.saved_values
          input_gradients = Backend.dispatch(
            :fully_fused_projection_backward,
            means,
            covars,
            quaternions,
            scales,
            viewmats,
            intrinsics,
            width,
            height,
            grad_means2d,
            grad_depths,
            grad_conics,
            grad_compensations,
            **options
          )
          [*input_gradients, nil, nil, nil, nil]
        end
      end
    end
  end

  class << self
    # Transforms world-space means/covariances into camera space.
    #
    # @param means [Numo::NArray] [N,3]
    # @param covars [Numo::NArray] [N,3,3] or packed [N,6]
    # @param viewmats [Numo::NArray] [C,4,4] world-to-camera transforms
    # @return [Array<Numo::NArray>] camera means `[C,N,3]` and covariances `[C,N,3,3]`
    def world_to_cam(means, covars, viewmats)
      Math::CameraProjection.world_to_cam(means, covars, viewmats)
    end

    # Projects camera-space means/covariances with a pinhole model.
    #
    # @param means [Numo::NArray] [C,N,3]
    # @param covars [Numo::NArray] [C,N,3,3]
    # @param intrinsics [Numo::NArray] [C,3,3]
    # @return [Array<Numo::NArray>] projected means `[C,N,2]` and covariances `[C,N,2,2]`
    def persp_proj(means, covars, intrinsics, width, height)
      Math::CameraProjection.persp_proj(means, covars, intrinsics, width, height)
    end

    # Projects camera-space means/covariances with an orthographic model.
    #
    # @param means [Numo::NArray] [C,N,3]
    # @param covars [Numo::NArray] [C,N,3,3]
    # @param intrinsics [Numo::NArray] [C,3,3]
    # @return [Array<Numo::NArray>] projected means `[C,N,2]` and covariances `[C,N,2,2]`
    def ortho_proj(means, covars, intrinsics, width, height)
      Math::CameraProjection.ortho_proj(means, covars, intrinsics, width, height)
    end

    # Projects and culls a dense camera batch.
    #
    # Inputs use float32/float64 Numo arrays or {Autograd::Variable}; geometry is
    # `[N,3]`, views `[C,4,4]`, intrinsics `[C,3,3]`, and outputs are
    # radii `[C,N,2]`, means `[C,N,2]`, depths `[C,N]`, conics `[C,N,3]`,
    # plus optional compensations `[C,N]`.
    #
    # @return [Array<(Numo::NArray, Autograd::Variable, nil)>]
    # rubocop:disable Metrics/ParameterLists, Naming/MethodParameterName
    def fully_fused_projection(means, viewmats:, ks:, width:, height:, covars: nil, quats: nil, scales: nil,
                               eps2d: 0.3, near_plane: 0.01, far_plane: 1e10, radius_clip: 0.0,
                               calc_compensations: false, camera_model: "pinhole",
                               radial_coeffs: nil, tangential_coeffs: nil, thin_prism_coeffs: nil,
                               global_z_order: true)
      # rubocop:enable Metrics/ParameterLists, Naming/MethodParameterName
      inputs = [means, covars, quats, scales, viewmats, ks]
      options = {
        eps2d: eps2d,
        near_plane: near_plane,
        far_plane: far_plane,
        radius_clip: radius_clip,
        calc_compensations: calc_compensations,
        camera_model: camera_model,
        radial_coeffs: radial_coeffs,
        tangential_coeffs: tangential_coeffs,
        thin_prism_coeffs: thin_prism_coeffs,
        global_z_order: global_z_order
      }
      if inputs.any?(Autograd::Variable)
        return Ops::FullyFusedProjection.apply(
          *inputs,
          width,
          height,
          **options
        )
      end

      Backend.dispatch(
        :fully_fused_projection_forward,
        *inputs,
        width,
        height,
        **options
      )
    end
  end

  Backend.register(
    :fully_fused_projection_forward,
    :ruby,
    Backend::RubyProjection.method(:forward)
  )
  Backend.register(
    :fully_fused_projection_backward,
    :ruby,
    Backend::RubyProjectionBackward.method(:backward)
  )
end
