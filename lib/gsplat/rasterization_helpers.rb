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
      unless sh_degree
        return colors if Ops::TensorOps.data(colors).ndim == 3

        return Ops::TensorOps.apply(Ops::CameraBroadcast, colors, camera_count)
      end

      directions = Ops::TensorOps.apply(Ops::CameraDirections, means, viewmats)
      coefficients = Ops::TensorOps.apply(Ops::CameraBroadcast, colors, camera_count)
      evaluated = Gsplat.spherical_harmonics(
        sh_degree,
        directions,
        coefficients,
        masks: visible_radii(radii)
      )
      Ops::TensorOps.apply(Ops::AddClampMin, evaluated, offset: 0.5, minimum: 0.0)
    end

    def visible_radii(radii)
      values = Ops::TensorOps.data(radii)
      return values.gt(0) unless values.ndim >= 3 && values.shape[-1] == 2

      values.min(axis: values.ndim - 1).gt(0)
    end

    def render_features(colors, depths, render_mode)
      return [colors, nil] if render_mode == "RGB"

      depth_features = Ops::TensorOps.apply(Ops::DepthFeatures, depths)
      return [depth_features, 0] if %w[d Ed D ED].include?(render_mode)

      feature_count = Ops::TensorOps.data(colors).shape[-1]
      [Ops::TensorOps.apply(Ops::ConcatDepth, colors, depth_features), feature_count]
    end

    # rubocop:disable Metrics/ParameterLists
    def rasterize_features(means2d, conics, features, opacities, width, height, tile_size, offsets, flatten_ids,
                           backgrounds:, channel_chunk:, absgrad:)
      # rubocop:enable Metrics/ParameterLists
      channel_count = Ops::TensorOps.data(features).shape[-1]
      if channel_count <= channel_chunk
        return Gsplat.rasterize_to_pixels(
          means2d,
          conics,
          features,
          opacities,
          width,
          height,
          tile_size,
          offsets,
          flatten_ids,
          backgrounds: backgrounds,
          absgrad: absgrad
        )
      end

      rasterize_feature_chunks(
        means2d, conics, features, opacities, width, height, tile_size, offsets, flatten_ids,
        backgrounds: backgrounds, channel_chunk: channel_chunk, absgrad: absgrad
      )
    end

    # rubocop:disable Metrics/ParameterLists
    def rasterize_feature_chunks(means2d, conics, features, opacities, width, height,
                                 tile_size, offsets, flatten_ids,
                                 backgrounds:, channel_chunk:, absgrad:)
      # rubocop:enable Metrics/ParameterLists
      channel_count = Ops::TensorOps.data(features).shape[-1]
      rendered_chunks = []
      alpha = nil
      (0...channel_count).step(channel_chunk) do |start|
        range = start...[start + channel_chunk, channel_count].min
        feature_chunk = Ops::TensorOps.apply(Ops::FeatureSlice, features, range)
        background_chunk = backgrounds && Ops::TensorOps.apply(Ops::FeatureSlice, backgrounds, range)
        rendered, chunk_alpha = Gsplat.rasterize_to_pixels(
          means2d,
          conics,
          feature_chunk,
          opacities,
          width,
          height,
          tile_size,
          offsets,
          flatten_ids,
          backgrounds: background_chunk,
          absgrad: absgrad
        )
        rendered_chunks << rendered
        alpha ||= chunk_alpha
      end
      [Ops::TensorOps.apply(Ops::ConcatFeatures, *rendered_chunks), alpha]
    end
    private_class_method :rasterize_feature_chunks

    # rubocop:disable Metrics/ParameterLists
    def rasterize_eval3d_features(means, quats, scales, features, opacities, viewmats, intrinsics,
                                  width, height, tile_size, offsets, flatten_ids, backgrounds:,
                                  camera_model:, use_hit_distance:, return_normals:)
      # rubocop:enable Metrics/ParameterLists
      raise ArgumentError, "world-space rendering requires quats and scales" unless quats && scales

      inputs = [means, quats, scales, features, opacities, backgrounds]
      options = {
        viewmats: Ops::TensorOps.data(viewmats),
        intrinsics: Ops::TensorOps.data(intrinsics),
        width: width,
        height: height,
        tile_size: tile_size,
        offsets: offsets,
        flatten_ids: flatten_ids,
        camera_model: camera_model.to_s,
        use_hit_distance: use_hit_distance,
        return_normals: return_normals
      }
      return Ops::Eval3dRasterize.apply(*inputs, **options) if inputs.any?(Autograd::Variable)

      Backend::RubyEval3dRasterizer.forward(*inputs, **options)
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
