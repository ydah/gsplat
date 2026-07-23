# frozen_string_literal: true

require_relative "../backend/ruby/isect_tiles"

# Tile intersection APIs.
module Gsplat
  class << self
    # Enumerates Gaussian/tile intersections.
    # rubocop:disable Metrics/ParameterLists
    def isect_tiles(means2d, radii, depths, tile_size, tile_width, tile_height, sort: true)
      # rubocop:enable Metrics/ParameterLists
      Backend.dispatch(
        :isect_tiles,
        means2d,
        radii,
        depths,
        tile_size,
        tile_width,
        tile_height,
        sort: sort
      )
    end

    # Encodes the starting intersection index for every camera tile.
    def isect_offset_encode(isect_ids, camera_count, tile_width, tile_height)
      Backend.dispatch(
        :isect_offset_encode,
        isect_ids,
        camera_count,
        tile_width,
        tile_height
      )
    end
  end

  Backend.register(:isect_tiles, :ruby, Backend::RubyIsectTiles.method(:forward))
  Backend.register(:isect_offset_encode, :ruby, Backend::RubyIsectTiles.method(:offset_encode))
end
