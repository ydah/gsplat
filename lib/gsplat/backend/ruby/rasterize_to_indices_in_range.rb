# frozen_string_literal: true

module Gsplat
  module Backend
    # Reference implementation of contribution-index enumeration.
    module RubyRasterizeToIndicesInRange
      module_function

      # rubocop:disable Metrics/ParameterLists
      def forward(range_start, range_end, transmittances, means2d, conics, opacities,
                  width, height, tile_size, isect_offsets, flatten_ids)
        # rubocop:enable Metrics/ParameterLists
        values = validate_inputs(
          range_start, range_end, transmittances, means2d, conics, opacities,
          width, height, tile_size, isect_offsets, flatten_ids
        )
        transmittances, means2d, conics, opacities, offsets, ids = values
        gaussian_ids = []
        pixel_ids = []
        image_ids = []
        means = means2d.reshape(means2d.shape[0] * means2d.shape[1], 2)
        conic_values = conics.reshape(conics.shape[0] * conics.shape[1], 3)
        opacity_values = opacities.flatten
        means2d.shape[0].times do |image|
          height.times do |row|
            width.times do |column|
              enumerate_pixel!(
                gaussian_ids, pixel_ids, image_ids, range_start, range_end,
                transmittances, means, conic_values, opacity_values, offsets, ids,
                image, row, column, means2d.shape[1], width, tile_size
              )
            end
          end
        end
        [Numo::Int64.cast(gaussian_ids), Numo::Int64.cast(pixel_ids), Numo::Int64.cast(image_ids)]
      end

      # rubocop:disable Metrics/AbcSize, Metrics/CyclomaticComplexity, Metrics/ParameterLists, Metrics/PerceivedComplexity
      def validate_inputs(range_start, range_end, transmittances, means2d, conics, opacities,
                          width, height, tile_size, isect_offsets, flatten_ids)
        # rubocop:enable Metrics/AbcSize, Metrics/CyclomaticComplexity, Metrics/ParameterLists, Metrics/PerceivedComplexity
        valid_range = range_start.is_a?(Integer) && range_end.is_a?(Integer) &&
                      range_start >= 0 && range_end >= range_start
        raise ArgumentError, "range must be nonnegative integers with start <= end" unless valid_range
        unless means2d.is_a?(Numo::NArray) && [Numo::SFloat, Numo::DFloat].include?(means2d.class)
          raise ArgumentError, "means2d must be Numo::SFloat or Numo::DFloat"
        end

        valid_means = means2d.ndim == 3 && means2d.shape[-1] == 2
        raise ShapeError, "expected means2d [C,N,2]" unless valid_means

        camera_count, gaussian_count = means2d.shape[0...2]
        type = means2d.class
        transmittances = type.cast(transmittances)
        conics = type.cast(conics)
        opacities = type.cast(opacities)
        valid = transmittances.shape == [camera_count, height, width] &&
                conics.shape == [camera_count, gaussian_count, 3] &&
                opacities.shape == [camera_count, gaussian_count]
        raise ShapeError, "invalid contribution-index tensor shapes" unless valid

        offsets = Numo::Int32.cast(isect_offsets)
        ids = Numo::Int32.cast(flatten_ids)
        valid_offsets = offsets.ndim == 3 && offsets.shape[0] == camera_count
        raise ShapeError, "expected isect_offsets [C,TH,TW]" unless valid_offsets
        raise ShapeError, "expected flatten_ids [M]" unless ids.ndim == 1

        RubyRasterizeToPixels.send(:validate_indices!, ids, camera_count * gaussian_count, offsets)
        RubyRasterizeToPixels.send(
          :validate_dimensions!, width, height, tile_size, offsets.shape[-1], offsets.shape[-2]
        )
        [transmittances, means2d, conics, opacities, offsets, ids]
      end
      private_class_method :validate_inputs

      # rubocop:disable Metrics/AbcSize, Metrics/ParameterLists
      def enumerate_pixel!(gaussian_ids, pixel_ids, image_ids, range_start, range_end,
                           transmittances, means, conics, opacities, offsets, flatten_ids,
                           image, row, column, gaussian_count, width, tile_size)
        # rubocop:enable Metrics/AbcSize, Metrics/ParameterLists
        range = intersection_batch(
          offsets, flatten_ids, image, row, column, tile_size, range_start, range_end
        )
        transmittance = transmittances[image, row, column].to_f
        range.each do |intersection|
          flat_id = flatten_ids[intersection].to_i
          delta_x = means[flat_id, 0].to_f - column - 0.5
          delta_y = means[flat_id, 1].to_f - row - 0.5
          conic = conics[flat_id, true]
          sigma = (0.5 * ((conic[0] * delta_x * delta_x) + (conic[2] * delta_y * delta_y))) +
                  (conic[1] * delta_x * delta_y)
          alpha = [opacities[flat_id].to_f * ::Math.exp(-sigma), RubyAccumulate::ALPHA_CLAMP].min
          next if sigma.negative? || alpha < RubyAccumulate::ALPHA_SKIP

          next_transmittance = transmittance * (1 - alpha)
          break if next_transmittance <= RubyAccumulate::TRANSMITTANCE_STOP

          gaussian_ids << (flat_id % gaussian_count)
          pixel_ids << ((row * width) + column)
          image_ids << image
          transmittance = next_transmittance
        end
      end
      private_class_method :enumerate_pixel!

      # rubocop:disable Metrics/ParameterLists
      def intersection_batch(offsets, flatten_ids, image, row, column, tile_size, first_batch, last_batch)
        # rubocop:enable Metrics/ParameterLists
        tile_height, tile_width = offsets.shape[-2..]
        flat_tile = (((image * tile_height) + (row / tile_size)) * tile_width) + (column / tile_size)
        tile_start = offsets[flat_tile].to_i
        tile_end = flat_tile + 1 < offsets.size ? offsets[flat_tile + 1].to_i : flatten_ids.size
        batch_size = tile_size * tile_size
        start_index = [tile_start + (first_batch * batch_size), tile_end].min
        end_index = [tile_start + (last_batch * batch_size), tile_end].min
        start_index...end_index
      end
      private_class_method :intersection_batch
    end
  end
end
