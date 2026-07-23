# frozen_string_literal: true

require_relative "native_raster_ops"

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
                           eps2d:, near_plane:, far_plane:, radius_clip:, calc_compensations:, camera_model:,
                           radial_coeffs: nil, tangential_coeffs: nil, thin_prism_coeffs: nil,
                           global_z_order: true)
      # rubocop:enable Metrics/ParameterLists
      extended = camera_model.to_s == "fisheye" ||
                 [radial_coeffs, tangential_coeffs, thin_prism_coeffs].any?
      unless means.is_a?(Numo::SFloat) && !extended && global_z_order
        return Backend::RubyProjection.forward(
          means, covars, quaternions, scales, viewmats, intrinsics, width, height,
          eps2d: eps2d, near_plane: near_plane, far_plane: far_plane,
          radius_clip: radius_clip, calc_compensations: calc_compensations,
          camera_model: camera_model, radial_coeffs: radial_coeffs,
          tangential_coeffs: tangential_coeffs, thin_prism_coeffs: thin_prism_coeffs,
          global_z_order: global_z_order
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

    # rubocop:disable Metrics/ParameterLists
    def isect_tiles(means2d, radii, depths, tile_size, tile_width, tile_height, sort:)
      # rubocop:enable Metrics/ParameterLists
      prepared = Backend::RubyIsectTiles.send(
        :validate_inputs, means2d, radii, depths, tile_size, tile_width, tile_height
      )
      typed_means, typed_radii, typed_depths = prepared
      unless typed_means.is_a?(Numo::SFloat)
        return Backend::RubyIsectTiles.forward(
          means2d, radii, depths, tile_size, tile_width, tile_height, sort: sort
        )
      end
      tile_bits = Backend::RubyIsectTiles.send(:tile_n_bits, tile_width, tile_height)
      Backend::RubyIsectTiles.send(:validate_key_width!, typed_means.shape[0], tile_bits)

      Native.isect_tiles_sfloat(
        typed_means.dup,
        typed_radii.dup,
        typed_depths.dup,
        tile_size,
        tile_width,
        tile_height,
        sort
      )
    end

    def isect_offset_encode(keys, camera_count, tile_width, tile_height)
      Backend::RubyIsectTiles.send(:validate_grid!, camera_count, tile_width, tile_height)
      unless keys.is_a?(Numo::Int64)
        return Backend::RubyIsectTiles.offset_encode(keys, camera_count, tile_width, tile_height)
      end
      raise ShapeError, "expected isect_ids [M]" unless keys.ndim == 1

      Native.isect_offset_encode_int64(keys.dup, camera_count, tile_width, tile_height)
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
    Backend.register(:isect_tiles, :native, NativeOps.method(:isect_tiles))
    Backend.register(
      :isect_offset_encode,
      :native,
      NativeOps.method(:isect_offset_encode)
    )
    Backend.register(
      :rasterize_to_pixels_forward,
      :native,
      NativeRasterOps.method(:forward)
    )
    Backend.register(
      :rasterize_to_pixels_backward,
      :native,
      NativeRasterOps.method(:backward)
    )
  end
end
