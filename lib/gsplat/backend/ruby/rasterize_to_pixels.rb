# frozen_string_literal: true

module Gsplat
  module Backend
    # Tile-accelerated Numo rasterization forward pass.
    module RubyRasterizeToPixels
      module_function

      # rubocop:disable Metrics/ParameterLists
      def forward(means2d, conics, colors, opacities, backgrounds, masks, width, height,
                  tile_size, isect_offsets, flatten_ids)
        # rubocop:enable Metrics/ParameterLists
        inputs = validate_inputs(
          means2d, conics, colors, opacities, backgrounds, masks, width, height,
          tile_size, isect_offsets, flatten_ids
        )
        means2d, conics, colors, opacities, backgrounds, masks, isect_offsets, flatten_ids = inputs
        camera_count, gaussian_count = means2d.shape[0...2]
        channel_count = colors.shape[-1]
        tile_height, tile_width = isect_offsets.shape[-2..]
        render_colors = means2d.class.zeros(camera_count, height, width, channel_count)
        render_alphas = means2d.class.zeros(camera_count, height, width, 1)
        last_ids = Numo::Int32.zeros(camera_count, height, width)
        flattened_means = means2d.reshape(camera_count * gaussian_count, 2)
        flattened_conics = conics.reshape(camera_count * gaussian_count, 3)
        flattened_colors = colors.reshape(camera_count * gaussian_count, channel_count)
        flattened_opacities = opacities.reshape(camera_count * gaussian_count)
        camera_count.times do |camera_index|
          tile_height.times do |tile_y|
            tile_width.times do |tile_x|
              render_tile!(
                render_colors,
                render_alphas,
                last_ids,
                flattened_means,
                flattened_conics,
                flattened_colors,
                flattened_opacities,
                backgrounds,
                masks,
                isect_offsets,
                flatten_ids,
                camera_index,
                tile_x,
                tile_y,
                width,
                height,
                tile_size
              )
            end
          end
        end
        [render_colors, render_alphas, last_ids]
      end

      # rubocop:disable Metrics/AbcSize, Metrics/ParameterLists
      def validate_inputs(means2d, conics, colors, opacities, backgrounds, masks, width, height,
                          tile_size, isect_offsets, flatten_ids)
        # rubocop:enable Metrics/AbcSize, Metrics/ParameterLists
        unless means2d.is_a?(Numo::NArray) && [Numo::SFloat, Numo::DFloat].include?(means2d.class)
          raise ArgumentError, "means2d must be Numo::SFloat or Numo::DFloat"
        end

        valid_means = means2d.ndim == 3 && means2d.shape[-1] == 2
        raise ShapeError, "expected means2d [C,N,2], got #{means2d.shape.inspect}" unless valid_means

        leading_shape = means2d.shape[0...2]
        conics = means2d.class.cast(conics)
        colors = means2d.class.cast(colors)
        opacities = means2d.class.cast(opacities)
        valid = conics.shape == leading_shape + [3] &&
                colors.ndim == 3 && colors.shape[0...2] == leading_shape &&
                opacities.shape == leading_shape
        unless valid
          raise ShapeError,
                "expected conics [C,N,3], colors [C,N,D], and opacities [C,N], " \
                "got #{conics.shape.inspect}, #{colors.shape.inspect}, and #{opacities.shape.inspect}"
        end
        backgrounds = validate_backgrounds(means2d.class, backgrounds, means2d.shape[0], colors.shape[-1])
        isect_offsets = Numo::Int32.cast(isect_offsets)
        valid_offsets = isect_offsets.ndim == 3 && isect_offsets.shape[0] == means2d.shape[0]
        raise ShapeError, "expected isect_offsets [C,TH,TW]" unless valid_offsets

        masks = validate_masks(masks, isect_offsets.shape)
        flatten_ids = Numo::Int32.cast(flatten_ids)
        validate_indices!(flatten_ids, means2d.shape[0] * means2d.shape[1], isect_offsets)
        validate_dimensions!(width, height, tile_size, isect_offsets.shape[-1], isect_offsets.shape[-2])
        [means2d, conics, colors, opacities, backgrounds, masks, isect_offsets, flatten_ids]
      end
      private_class_method :validate_inputs

      def validate_backgrounds(type, backgrounds, camera_count, channel_count)
        return nil unless backgrounds

        value = type.cast(backgrounds)
        expected = [camera_count, channel_count]
        unless value.shape == expected
          raise ShapeError, "expected backgrounds #{expected.inspect}, got #{value.shape.inspect}"
        end

        value
      end
      private_class_method :validate_backgrounds

      def validate_masks(masks, expected_shape)
        return nil unless masks

        value = Numo::Bit.cast(masks)
        unless value.shape == expected_shape
          raise ShapeError, "expected masks #{expected_shape.inspect}, got #{value.shape.inspect}"
        end

        value
      end
      private_class_method :validate_masks

      def validate_indices!(flatten_ids, gaussian_count, offsets)
        raise ShapeError, "expected flatten_ids [M]" unless flatten_ids.ndim == 1
        if flatten_ids.size.positive? && (flatten_ids.min.negative? || flatten_ids.max >= gaussian_count)
          raise ShapeError, "flatten_ids contains an out-of-range Gaussian index"
        end

        values = offsets.to_a.flatten
        valid = values.each_cons(2).all? { |left, right| left <= right } &&
                values.all? { |value| value.between?(0, flatten_ids.size) }
        raise Gsplat::Error, "intersection offsets must be sorted indices into flatten_ids" unless valid
      end
      private_class_method :validate_indices!

      def validate_dimensions!(width, height, tile_size, tile_width, tile_height)
        valid = [width, height, tile_size].all? { |value| value.is_a?(Integer) && value.positive? }
        raise ArgumentError, "image dimensions and tile_size must be positive integers" unless valid
        return if (tile_width * tile_size) >= width && (tile_height * tile_size) >= height

        raise ShapeError, "intersection offset grid does not cover the image"
      end
      private_class_method :validate_dimensions!

      # rubocop:disable Metrics/AbcSize, Metrics/ParameterLists
      def render_tile!(render_colors, render_alphas, last_ids, means2d, conics, colors, opacities,
                       backgrounds, masks, offsets, flatten_ids, camera_index, tile_x, tile_y,
                       width, height, tile_size)
        # rubocop:enable Metrics/ParameterLists
        x_start = tile_x * tile_size
        y_start = tile_y * tile_size
        x_end = [x_start + tile_size, width].min
        y_end = [y_start + tile_size, height].min
        return if x_start >= width || y_start >= height

        if masks && masks[camera_index, tile_y, tile_x].zero?
          fill_background!(render_colors, backgrounds, camera_index, x_start, x_end, y_start, y_end)
          return
        end
        tile_width = offsets.shape[-1]
        tile_height = offsets.shape[-2]
        flat_tile = (((camera_index * tile_height) + tile_y) * tile_width) + tile_x
        range_start = offsets[flat_tile]
        range_end = flat_tile + 1 < offsets.size ? offsets[flat_tile + 1] : flatten_ids.size
        pixels_x, pixels_y = tile_coordinates(means2d.class, x_start, x_end, y_start, y_end)
        composited, transmittance, tile_last_ids = RubyTileCompositor.composite(
          means2d, conics, colors, opacities, flatten_ids, range_start, range_end,
          pixels_x, pixels_y
        )
        pixel_height = y_end - y_start
        pixel_width = x_end - x_start
        if backgrounds
          composited += transmittance.reshape(transmittance.size, 1) *
                        backgrounds[camera_index, true].reshape(1, colors.shape[-1])
        end
        render_colors[camera_index, y_start...y_end, x_start...x_end, true] =
          composited.reshape(pixel_height, pixel_width, colors.shape[-1])
        render_alphas[camera_index, y_start...y_end, x_start...x_end, 0] =
          (1 - transmittance).reshape(pixel_height, pixel_width)
        last_ids[camera_index, y_start...y_end, x_start...x_end] =
          tile_last_ids.reshape(pixel_height, pixel_width)
      end
      # rubocop:enable Metrics/AbcSize
      private_class_method :render_tile!

      # rubocop:disable Metrics/ParameterLists
      def fill_background!(render_colors, backgrounds, camera_index, x_start, x_end, y_start, y_end)
        # rubocop:enable Metrics/ParameterLists
        return unless backgrounds

        render_colors[camera_index, y_start...y_end, x_start...x_end, true] =
          backgrounds[camera_index, true]
      end
      private_class_method :fill_background!

      def tile_coordinates(type, x_start, x_end, y_start, y_end)
        width = x_end - x_start
        x_values = Array.new(y_end - y_start) { (x_start...x_end).map { |column| column + 0.5 } }.flatten
        y_values = (y_start...y_end).flat_map { |row| Array.new(width, row + 0.5) }
        [type.cast(x_values), type.cast(y_values)]
      end
      private_class_method :tile_coordinates
    end
  end
end
