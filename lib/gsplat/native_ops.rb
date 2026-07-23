# frozen_string_literal: true

# Native operation registration and validated Ruby fallbacks.
module Gsplat
  # Ruby-facing validation and fallback wrappers for native operations.
  module NativeOps
    module_function

    def spherical_harmonics_forward(degree, directions, coefficients, masks: nil)
      prepared = Backend::RubySphericalHarmonics.send(
        :validate_inputs, degree, directions, coefficients, masks
      )
      typed_directions, typed_coefficients, typed_masks, = prepared
      return Backend::RubySphericalHarmonics.forward(degree, directions, coefficients, masks: masks) if typed_masks
      unless typed_directions.is_a?(Numo::SFloat)
        return Backend::RubySphericalHarmonics.forward(degree, directions, coefficients, masks: masks)
      end

      Native.spherical_harmonics_forward_sfloat(
        degree,
        typed_directions.dup,
        typed_coefficients.dup
      )
    end

    # rubocop:disable Metrics/ParameterLists
    def projection_forward(means, covars, quaternions, scales, viewmats, intrinsics, width, height,
                           eps2d:, near_plane:, far_plane:, radius_clip:, calc_compensations:, camera_model:)
      # rubocop:enable Metrics/ParameterLists
      unless means.is_a?(Numo::SFloat)
        return Backend::RubyProjection.forward(
          means, covars, quaternions, scales, viewmats, intrinsics, width, height,
          eps2d: eps2d, near_plane: near_plane, far_plane: far_plane,
          radius_clip: radius_clip, calc_compensations: calc_compensations,
          camera_model: camera_model
        )
      end
      prepared = Backend::RubyProjection.send(
        :prepare_inputs,
        means, covars, quaternions, scales, viewmats, intrinsics, width, height,
        eps2d, near_plane, far_plane, radius_clip, camera_model
      )
      typed_means, typed_covars, typed_views, typed_intrinsics = prepared
      Native.projection_forward_sfloat(
        typed_means.dup,
        typed_covars.dup,
        typed_views.dup,
        typed_intrinsics.dup,
        width,
        height,
        eps2d,
        near_plane,
        far_plane,
        radius_clip,
        calc_compensations,
        camera_model.to_s == "ortho"
      )
    end
  end

  if Native.available?
    Backend.register(
      :spherical_harmonics_forward,
      :native,
      NativeOps.method(:spherical_harmonics_forward)
    )
    Backend.register(
      :spherical_harmonics_backward,
      :native,
      Backend::RubySphericalHarmonics.method(:backward)
    )
    Backend.register(
      :fully_fused_projection_forward,
      :native,
      NativeOps.method(:projection_forward)
    )
    Backend.register(
      :fully_fused_projection_backward,
      :native,
      Backend::RubyProjectionBackward.method(:backward)
    )
  end
end
