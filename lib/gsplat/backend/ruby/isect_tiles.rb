# frozen_string_literal: true

module Gsplat
  module Backend
    # Numo/Ruby tile intersection enumeration and offset encoding.
    module RubyIsectTiles
      TILE_KEY_LOW_BITS = 32

      module_function

      # rubocop:disable Metrics/ParameterLists
      def forward(means2d, radii, depths, tile_size, tile_width, tile_height, sort:)
        # rubocop:enable Metrics/ParameterLists
        means2d, radii, depths = validate_inputs(
          means2d,
          radii,
          depths,
          tile_size,
          tile_width,
          tile_height
        )
        camera_count, gaussian_count = means2d.shape[0...2]
        bounds, tiles_per_gauss = tile_bounds(
          means2d,
          radii,
          tile_size,
          tile_width,
          tile_height
        )
        intersection_count = tiles_per_gauss.sum.to_i
        isect_ids = Numo::Int64.zeros(intersection_count)
        flatten_ids = Numo::Int32.zeros(intersection_count)
        tile_bits = tile_n_bits(tile_width, tile_height)
        validate_key_width!(camera_count, tile_bits)
        write_intersections!(
          isect_ids,
          flatten_ids,
          bounds,
          tiles_per_gauss,
          depths,
          gaussian_count,
          tile_width,
          tile_bits
        )
        if sort && intersection_count.positive?
          keys = isect_ids.to_a
          order = (0...intersection_count).sort_by { |index| [keys[index], index] }
          isect_ids = isect_ids[order].dup
          flatten_ids = flatten_ids[order].dup
        end
        [tiles_per_gauss, isect_ids, flatten_ids]
      end

      def offset_encode(isect_ids, camera_count, tile_width, tile_height)
        validate_grid!(camera_count, tile_width, tile_height)
        raise ShapeError, "expected isect_ids [M]" unless isect_ids.is_a?(Numo::NArray) && isect_ids.ndim == 1

        keys = Numo::Int64.cast(isect_ids)
        counts = Numo::Int64.zeros(camera_count, tile_height, tile_width)
        tile_bits = tile_n_bits(tile_width, tile_height)
        tile_mask = (1 << tile_bits) - 1
        keys.each do |key|
          upper = key >> TILE_KEY_LOW_BITS
          camera_id = upper >> tile_bits
          tile_id = upper & tile_mask
          validate_decoded_key!(camera_id, tile_id, camera_count, tile_width, tile_height)
          counts[camera_id, tile_id / tile_width, tile_id % tile_width] += 1
        end
        offsets = Numo::Int32.zeros(camera_count, tile_height, tile_width)
        running = 0
        counts.size.times do |index|
          offsets[index] = running
          running += counts[index]
        end
        offsets
      end

      # rubocop:disable Metrics/ParameterLists
      def validate_inputs(means2d, radii, depths, tile_size, tile_width, tile_height)
        # rubocop:enable Metrics/ParameterLists
        unless means2d.is_a?(Numo::NArray) && [Numo::SFloat, Numo::DFloat].include?(means2d.class)
          raise ArgumentError, "means2d must be Numo::SFloat or Numo::DFloat"
        end

        valid_means = means2d.ndim == 3 && means2d.shape[-1] == 2
        raise ShapeError, "expected means2d [C,N,2], got #{means2d.shape.inspect}" unless valid_means

        leading_shape = means2d.shape[0...2]
        valid_radii = [leading_shape, leading_shape + [2]].include?(radii.shape)
        unless valid_radii && depths.shape == leading_shape
          raise ShapeError,
                "expected radii [C,N] or [C,N,2] and depths #{leading_shape.inspect}, " \
                "got #{radii.shape.inspect} and #{depths.shape.inspect}"
        end
        validate_grid!(means2d.shape[0], tile_width, tile_height)
        valid_tile_size = tile_size.is_a?(Integer) && tile_size.positive?
        raise ArgumentError, "tile_size must be a positive integer" unless valid_tile_size

        [means2d, means2d.class.cast(radii), means2d.class.cast(depths)]
      end
      private_class_method :validate_inputs

      def validate_grid!(camera_count, tile_width, tile_height)
        values = [camera_count, tile_width, tile_height]
        return if values.all? { |value| value.is_a?(Integer) && value.positive? }

        raise ArgumentError, "camera count and tile dimensions must be positive integers"
      end
      private_class_method :validate_grid!

      # rubocop:disable Metrics/AbcSize
      def tile_bounds(means2d, radii, tile_size, tile_width, tile_height)
        camera_count, gaussian_count = means2d.shape[0...2]
        bounds = Numo::Int32.zeros(camera_count, gaussian_count, 4)
        counts = Numo::Int32.zeros(camera_count, gaussian_count)
        elliptical = radii.ndim == 3
        camera_count.times do |camera_index|
          gaussian_count.times do |gaussian_index|
            if elliptical
              radius_x = radii[camera_index, gaussian_index, 0].to_f
              radius_y = radii[camera_index, gaussian_index, 1].to_f
            else
              radius_x = radii[camera_index, gaussian_index].to_f
              radius_y = radius_x
            end
            next unless radius_x.positive? && radius_y.positive?

            mean_x = means2d[camera_index, gaussian_index, 0].to_f
            mean_y = means2d[camera_index, gaussian_index, 1].to_f
            min_x = clamp_tile(((mean_x - radius_x) / tile_size).floor, tile_width)
            max_x = clamp_tile(((mean_x + radius_x) / tile_size).ceil, tile_width)
            min_y = clamp_tile(((mean_y - radius_y) / tile_size).floor, tile_height)
            max_y = clamp_tile(((mean_y + radius_y) / tile_size).ceil, tile_height)
            bounds[camera_index, gaussian_index, true] = [min_x, min_y, max_x, max_y]
            counts[camera_index, gaussian_index] = (max_x - min_x) * (max_y - min_y)
          end
        end
        [bounds, counts]
      end
      # rubocop:enable Metrics/AbcSize
      private_class_method :tile_bounds

      def clamp_tile(value, maximum)
        value.clamp(0, maximum)
      end
      private_class_method :clamp_tile

      # rubocop:disable Metrics/ParameterLists
      def write_intersections!(keys, flatten_ids, bounds, counts, depths, gaussian_count, tile_width, tile_bits)
        # rubocop:enable Metrics/ParameterLists
        write_index = 0
        counts.shape[0].times do |camera_index|
          gaussian_count.times do |gaussian_index|
            next if counts[camera_index, gaussian_index].zero?

            min_x, min_y, max_x, max_y = bounds[camera_index, gaussian_index, true].to_a
            upper_camera = camera_index << tile_bits
            depth_bits = float32_bits(depths[camera_index, gaussian_index])
            (min_y...max_y).each do |tile_y|
              (min_x...max_x).each do |tile_x|
                tile_id = (tile_y * tile_width) + tile_x
                keys[write_index] = ((upper_camera | tile_id) << TILE_KEY_LOW_BITS) | depth_bits
                flatten_ids[write_index] = (camera_index * gaussian_count) + gaussian_index
                write_index += 1
              end
            end
          end
        end
      end
      private_class_method :write_intersections!

      def float32_bits(value)
        [value.to_f].pack("g").unpack1("N")
      end
      private_class_method :float32_bits

      def tile_n_bits(tile_width, tile_height)
        tile_count = tile_width * tile_height
        tile_count == 1 ? 0 : ::Math.log2(tile_count).ceil
      end
      private_class_method :tile_n_bits

      def validate_key_width!(camera_count, tile_bits)
        return if (camera_count - 1).bit_length + tile_bits <= TILE_KEY_LOW_BITS

        raise ArgumentError, "camera and tile ids exceed the 64-bit intersection key layout"
      end
      private_class_method :validate_key_width!

      def validate_decoded_key!(camera_id, tile_id, camera_count, tile_width, tile_height)
        return if camera_id.between?(0, camera_count - 1) && tile_id.between?(0, (tile_width * tile_height) - 1)

        raise Gsplat::Error, "intersection key contains an out-of-range camera or tile id"
      end
      private_class_method :validate_decoded_key!
    end
  end
end
