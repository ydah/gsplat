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
    def world_to_cam(means, covars, viewmats)
      Math::CameraProjection.world_to_cam(means, covars, viewmats)
    end

    def persp_proj(means, covars, intrinsics, width, height)
      Math::CameraProjection.persp_proj(means, covars, intrinsics, width, height)
    end

    def ortho_proj(means, covars, intrinsics, width, height)
      Math::CameraProjection.ortho_proj(means, covars, intrinsics, width, height)
    end

    # rubocop:disable Metrics/ParameterLists, Naming/MethodParameterName
    def fully_fused_projection(means, viewmats:, ks:, width:, height:, covars: nil, quats: nil, scales: nil,
                               eps2d: 0.3, near_plane: 0.01, far_plane: 1e10, radius_clip: 0.0,
                               calc_compensations: false, camera_model: "pinhole",
                               radial_coeffs: nil, tangential_coeffs: nil, thin_prism_coeffs: nil)
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
        thin_prism_coeffs: thin_prism_coeffs
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
