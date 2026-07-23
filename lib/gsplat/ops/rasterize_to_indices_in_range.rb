# frozen_string_literal: true

require_relative "../backend/ruby/accumulate"
require_relative "../backend/ruby/rasterize_to_pixels"
require_relative "../backend/ruby/rasterize_to_indices_in_range"

# Differentiable Gaussian rasterization primitives.
module Gsplat
  class << self
    # Enumerates Gaussian contributions for tile-list batches in depth order.
    #
    # @return [Array<Numo::Int64>] Gaussian, pixel, and image IDs
    # rubocop:disable Metrics/ParameterLists
    def rasterize_to_indices_in_range(range_start, range_end, transmittances, means2d, conics, opacities,
                                      width, height, tile_size, isect_offsets, flatten_ids)
      # rubocop:enable Metrics/ParameterLists
      tensors = [transmittances, means2d, conics, opacities, isect_offsets, flatten_ids].map do |value|
        Ops::TensorOps.data(value)
      end
      Backend.dispatch(
        :rasterize_to_indices_in_range,
        range_start,
        range_end,
        *tensors.first(4),
        width,
        height,
        tile_size,
        *tensors.last(2)
      )
    end
  end

  Backend.register(
    :rasterize_to_indices_in_range,
    :ruby,
    Backend::RubyRasterizeToIndicesInRange.method(:forward)
  )
end
