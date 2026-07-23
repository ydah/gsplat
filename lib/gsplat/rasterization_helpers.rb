# frozen_string_literal: true

module Gsplat
  # Feature preparation and metadata helpers for the high-level renderer.
  module RasterizationHelpers
    module_function

    def camera_broadcast(value, camera_count)
      shape = Ops::TensorOps.data(value).shape
      return value if shape[0] == camera_count && shape.length >= 2

      Ops::TensorOps.apply(Ops::CameraBroadcast, value, camera_count)
    end

    # rubocop:disable Metrics/ParameterLists
    def prepare_colors(means, colors, viewmats, radii, camera_count, sh_degree)
      # rubocop:enable Metrics/ParameterLists
      return camera_broadcast(colors, camera_count) unless sh_degree

      directions = Ops::TensorOps.apply(Ops::CameraDirections, means, viewmats)
      coefficients = Ops::TensorOps.apply(Ops::CameraBroadcast, colors, camera_count)
      evaluated = Gsplat.spherical_harmonics(
        sh_degree,
        directions,
        coefficients,
        masks: Ops::TensorOps.data(radii).gt(0)
      )
      Ops::TensorOps.apply(Ops::AddClampMin, evaluated, offset: 0.5, minimum: 0.0)
    end

    def render_features(colors, depths, render_mode)
      return [colors, nil] if render_mode == "RGB"

      depth_features = Ops::TensorOps.apply(Ops::DepthFeatures, depths)
      return [depth_features, 0] if %w[D ED].include?(render_mode)

      feature_count = Ops::TensorOps.data(colors).shape[-1]
      [Ops::TensorOps.apply(Ops::ConcatDepth, colors, depth_features), feature_count]
    end

    # rubocop:disable Metrics/ParameterLists
    def metadata(radii, means2d, depths, conics, opacities, tile_width, tile_height,
                 tiles_per_gauss, isect_ids, flatten_ids, isect_offsets, width, height,
                 tile_size, camera_count)
      # rubocop:enable Metrics/ParameterLists
      {
        camera_ids: nil,
        gaussian_ids: nil,
        radii: radii,
        means2d: means2d,
        depths: depths,
        conics: conics,
        opacities: opacities,
        tile_width: tile_width,
        tile_height: tile_height,
        tiles_per_gauss: tiles_per_gauss,
        isect_ids: isect_ids,
        flatten_ids: flatten_ids,
        isect_offsets: isect_offsets,
        width: width,
        height: height,
        tile_size: tile_size,
        n_cameras: camera_count
      }
    end
  end
end
