#!/usr/bin/env ruby
# frozen_string_literal: true

require "gsplat"

Gsplat.backend = ENV.fetch("GSPLAT_BACKEND", "ruby").to_sym
count = Integer(ENV.fetch("N", "1000"), 10)
iterations = Integer(ENV.fetch("ITERATIONS", "5"), 10)
width = Integer(ENV.fetch("WIDTH", "64"), 10)
height = Integer(ENV.fetch("HEIGHT", "48"), 10)
rng = Random.new(42)

means = Numo::SFloat.cast(
  Array.new(count) { [rng.rand - 0.5, rng.rand - 0.5, 2.0 + rng.rand] }
)
quats = Numo::SFloat.zeros(count, 4)
quats[true, 0] = 1
scales = Numo::SFloat.ones(count, 3) * 0.03
opacities = Numo::SFloat.ones(count) * 0.2
colors = Numo::SFloat.new(count, 3).rand
directions = Numo::SFloat.new(count, 3).rand
coefficients = Numo::SFloat.new(count, 16, 3).rand
views = Numo::SFloat.eye(4).reshape(1, 4, 4)
intrinsics = Numo::SFloat[
  [[52, 0, width / 2.0], [0, 52, height / 2.0], [0, 0, 1]]
]

projection = lambda do
  Gsplat.fully_fused_projection(
    means,
    quats: quats,
    scales: scales,
    viewmats: views,
    ks: intrinsics,
    width: width,
    height: height
  )
end
radii, means2d, depths, conics, = projection.call
tile_width = (width / 16.0).ceil
tile_height = (height / 16.0).ceil
intersection = lambda do
  Gsplat.isect_tiles(means2d, radii, depths, 16, tile_width, tile_height)
end
_tiles, isect_ids, flatten_ids = intersection.call
offsets = Gsplat.isect_offset_encode(isect_ids, 1, tile_width, tile_height)
raster = lambda do
  Gsplat.rasterize_to_pixels(
    means2d,
    conics,
    colors.reshape(1, count, 3),
    opacities.reshape(1, count),
    width,
    height,
    16,
    offsets,
    flatten_ids
  )
end

operations = {
  projection: projection,
  spherical_harmonics: -> { Gsplat.spherical_harmonics(3, directions, coefficients) },
  intersections: intersection,
  rasterization: raster
}
puts "ruby=#{RUBY_VERSION} platform=#{RUBY_PLATFORM} backend=#{Gsplat.backend} " \
     "N=#{count} image=#{width}x#{height} iterations=#{iterations}"
operations.each do |name, operation|
  operation.call
  started_at = Process.clock_gettime(Process::CLOCK_MONOTONIC)
  iterations.times { operation.call }
  elapsed = Process.clock_gettime(Process::CLOCK_MONOTONIC) - started_at
  puts format("%<name>-22s %<milliseconds>10.3f ms/call", name: name, milliseconds: elapsed * 1_000 / iterations)
end
