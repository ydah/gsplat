# frozen_string_literal: true

module Gsplat
  module Backend
    # World-space Gaussian evaluator used by 3DGUT-style rendering.
    module RubyEval3dRasterizer
      ALPHA_CLAMP = 0.99
      ALPHA_SKIP = 1.0 / 255
      TRANSMITTANCE_STOP = 1e-4

      module_function

      # rubocop:disable Metrics/AbcSize, Metrics/ParameterLists
      def forward(means, quats, scales, colors, opacities, backgrounds, viewmats:, intrinsics:,
                  width:, height:, tile_size:, offsets:, flatten_ids:, camera_model:,
                  use_hit_distance: false, return_normals: false)
        # rubocop:enable Metrics/AbcSize, Metrics/ParameterLists
        values = validate_inputs(
          means, quats, scales, colors, opacities, backgrounds, viewmats, intrinsics,
          width, height, offsets, flatten_ids, camera_model
        )
        means, quats, scales, colors, opacities, backgrounds, viewmats, intrinsics = values
        camera_count, gaussian_count, channels = colors.shape
        rendered = means.class.zeros(camera_count, height, width, channels)
        alphas = means.class.zeros(camera_count, height, width, 1)
        rendered_normals = means.class.zeros(camera_count, height, width, 3)
        rotations = Math::Quaternion.to_rotmat(quats)
        camera_count.times do |camera|
          frame = camera_frame(viewmats[camera, true, true])
          height.times do |row|
            width.times do |column|
              tile_x = column / tile_size
              tile_y = row / tile_size
              range = intersection_range(offsets, flatten_ids, camera, tile_x, tile_y)
              ray = ray_for_pixel(frame, intrinsics[camera, true, true], column, row)
              pixel, transmittance, pixel_normal = composite_pixel(
                means, rotations, scales, colors, opacities, flatten_ids, range,
                camera, gaussian_count, ray, use_hit_distance, return_normals
              )
              pixel += backgrounds[camera, true] * transmittance if backgrounds
              rendered[camera, row, column, true] = pixel
              alphas[camera, row, column, 0] = 1 - transmittance
              rendered_normals[camera, row, column, true] = pixel_normal
            end
          end
        end
        [rendered, alphas, rendered_normals]
      end

      # rubocop:disable Metrics/AbcSize, Metrics/CyclomaticComplexity, Metrics/ParameterLists, Metrics/PerceivedComplexity
      def validate_inputs(means, quats, scales, colors, opacities, backgrounds, viewmats,
                          intrinsics, width, height, offsets, flatten_ids, camera_model)
        # rubocop:enable Metrics/AbcSize, Metrics/CyclomaticComplexity, Metrics/ParameterLists, Metrics/PerceivedComplexity
        unless camera_model == "pinhole"
          raise ArgumentError, "hit-distance rendering currently requires pinhole camera_model"
        end

        type = means.class
        quats = type.cast(quats)
        scales = type.cast(scales)
        colors = type.cast(colors)
        opacities = type.cast(opacities)
        viewmats = type.cast(viewmats)
        intrinsics = type.cast(intrinsics)
        camera_count = viewmats.shape[0]
        gaussian_count = means.shape[0]
        valid = means.shape == [gaussian_count, 3] &&
                quats.shape == [gaussian_count, 4] &&
                scales.shape == [gaussian_count, 3] &&
                colors.ndim == 3 && colors.shape[0...2] == [camera_count, gaussian_count] &&
                opacities.shape == [camera_count, gaussian_count] &&
                intrinsics.shape == [camera_count, 3, 3]
        raise ShapeError, "invalid world-space rasterization inputs" unless valid
        raise ArgumentError, "scales must be positive" unless scales.gt(0).all?

        backgrounds = type.cast(backgrounds) if backgrounds
        if backgrounds && backgrounds.shape != [camera_count, colors.shape[-1]]
          raise ShapeError, "expected backgrounds [#{camera_count},#{colors.shape[-1]}]"
        end
        raise ArgumentError, "width and height must be positive" unless width.positive? && height.positive?
        raise ShapeError, "expected offsets [C,TH,TW]" unless offsets.shape[0] == camera_count
        raise ShapeError, "expected flatten_ids [M]" unless flatten_ids.ndim == 1

        [means, quats, scales, colors, opacities, backgrounds, viewmats, intrinsics]
      end
      private_class_method :validate_inputs

      def camera_frame(viewmat)
        rotation = viewmat[0...3, 0...3]
        translation = viewmat[0...3, 3]
        [-translation.dot(rotation), rotation]
      end
      private_class_method :camera_frame

      def ray_for_pixel(frame, intrinsics, column, row)
        origin, rotation = frame
        camera_direction = origin.class[
          (column + 0.5 - intrinsics[0, 2]) / intrinsics[0, 0],
          (row + 0.5 - intrinsics[1, 2]) / intrinsics[1, 1],
          1.0
        ]
        direction = camera_direction.dot(rotation)
        [origin, direction / ::Math.sqrt((direction**2).sum)]
      end
      private_class_method :ray_for_pixel

      def intersection_range(offsets, flatten_ids, camera, tile_x, tile_y)
        tile_height, tile_width = offsets.shape[-2..]
        flat = (((camera * tile_height) + tile_y) * tile_width) + tile_x
        start_index = offsets[flat].to_i
        end_index = flat + 1 < offsets.size ? offsets[flat + 1].to_i : flatten_ids.size
        start_index...end_index
      end
      private_class_method :intersection_range

      # rubocop:disable Metrics/AbcSize, Metrics/ParameterLists
      def composite_pixel(means, rotations, scales, colors, opacities, flatten_ids, range,
                          camera, gaussian_count, ray, use_hit_distance, return_normals)
        # rubocop:enable Metrics/AbcSize, Metrics/ParameterLists
        pixel = means.class.zeros(colors.shape[-1])
        pixel_normal = means.class.zeros(3)
        transmittance = 1.0
        range.each do |intersection|
          flat_id = flatten_ids[intersection].to_i
          gaussian = flat_id % gaussian_count
          gray_distance, hit_distance = ray_gaussian(
            ray, means[gaussian, true], rotations[gaussian, true, true], scales[gaussian, true]
          )
          next unless gray_distance

          opacity = opacities[camera, gaussian].to_f
          alpha = [opacity * ::Math.exp(-0.5 * gray_distance), ALPHA_CLAMP].min
          next if alpha < ALPHA_SKIP

          next_transmittance = transmittance * (1 - alpha)
          break if next_transmittance <= TRANSMITTANCE_STOP

          weight = alpha * transmittance
          feature = colors[camera, gaussian, true].dup
          feature[feature.size - 1] = hit_distance if use_hit_distance
          pixel += feature * weight
          if return_normals
            normal = rotations[gaussian, true, 2].dup
            normal *= -1 if (normal * ray[1]).sum.positive?
            pixel_normal += normal * weight
          end
          transmittance = next_transmittance
        end
        [pixel, transmittance, pixel_normal]
      end
      private_class_method :composite_pixel

      # rubocop:disable Metrics/AbcSize
      def ray_gaussian(ray, mean, rotation, scale)
        origin, direction = ray
        local_origin = (origin - mean).dot(rotation) / scale
        local_direction = direction.dot(rotation) / scale
        local_direction /= ::Math.sqrt((local_direction**2).sum)
        hit_t = -(local_direction * local_origin).sum
        return [nil, nil] if hit_t.negative?

        cross = local_direction.class[
          (local_direction[1] * local_origin[2]) - (local_direction[2] * local_origin[1]),
          (local_direction[2] * local_origin[0]) - (local_direction[0] * local_origin[2]),
          (local_direction[0] * local_origin[1]) - (local_direction[1] * local_origin[0])
        ]
        gray_distance = (cross**2).sum.to_f
        hit_vector = scale * local_direction * hit_t
        [gray_distance, ::Math.sqrt((hit_vector**2).sum)]
      end
      # rubocop:enable Metrics/AbcSize
      private_class_method :ray_gaussian
    end
  end
end
