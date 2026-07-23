#!/usr/bin/env ruby
# frozen_string_literal: true

require "gsplat"

Gsplat.backend = ENV.fetch("GSPLAT_BACKEND", "ruby").to_sym
count = Integer(ENV.fetch("N", "1000"), 10)
iterations = Integer(ENV.fetch("ITERATIONS", "5"), 10)
width = Integer(ENV.fetch("WIDTH", "64"), 10)
height = Integer(ENV.fetch("HEIGHT", "48"), 10)
tile_size = 16
rng = Random.new(42)

means = Numo::SFloat.cast(Array.new(count) { [rng.rand - 0.5, rng.rand - 0.5, 2 + rng.rand] })
quaternions = Numo::SFloat.zeros(count, 4)
quaternions[true, 0] = 1
scales = Numo::SFloat.ones(count, 3) * 0.03
views = Numo::SFloat.eye(4).reshape(1, 4, 4)
intrinsics = Numo::SFloat[
  [[52, 0, width / 2.0], [0, 52, height / 2.0], [0, 0, 1]]
]
radii, means2d, depths, conics, = Gsplat.fully_fused_projection(
  means,
  quats: quaternions,
  scales: scales,
  viewmats: views,
  ks: intrinsics,
  width: width,
  height: height
)
colors = Numo::SFloat.new(1, count, 3).rand
opacities = Numo::SFloat.ones(1, count) * 0.2
tile_width = (width.to_f / tile_size).ceil
tile_height = (height.to_f / tile_size).ceil
_, keys, ids = Gsplat.isect_tiles(means2d, radii, depths, tile_size, tile_width, tile_height)
offsets = Gsplat.isect_offset_encode(keys, 1, tile_width, tile_height)
forward_args = [
  means2d, conics, colors, opacities, nil, nil, width, height, tile_size, offsets, ids
]
render_colors, render_alphas, last_ids = Gsplat::Backend.dispatch(
  :rasterize_to_pixels_forward, *forward_args
)
color_grad = Numo::SFloat.ones(*render_colors.shape)
alpha_grad = Numo::SFloat.ones(*render_alphas.shape)
backward_args = [
  *forward_args, render_alphas, last_ids, color_grad, alpha_grad
]

operations = {
  forward: -> { Gsplat::Backend.dispatch(:rasterize_to_pixels_forward, *forward_args) },
  backward: lambda {
    Gsplat::Backend.dispatch(:rasterize_to_pixels_backward, *backward_args, absgrad: true)
  }
}
puts "ruby=#{RUBY_VERSION} platform=#{RUBY_PLATFORM} backend=#{Gsplat.backend} " \
     "N=#{count} image=#{width}x#{height} iterations=#{iterations}"
operations.each do |name, operation|
  operation.call
  started_at = Process.clock_gettime(Process::CLOCK_MONOTONIC)
  iterations.times { operation.call }
  elapsed = Process.clock_gettime(Process::CLOCK_MONOTONIC) - started_at
  puts format("%<name>-12s %<milliseconds>10.3f ms/call",
              name: name, milliseconds: elapsed * 1_000 / iterations)
end
