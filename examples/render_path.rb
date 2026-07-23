#!/usr/bin/env ruby
# frozen_string_literal: true

require "fileutils"
require "optparse"
require "gsplat"

options = { ply: nil, output: "renders", frames: 60, width: 512, height: 512 }
OptionParser.new do |parser|
  parser.banner = "Usage: bundle exec ruby examples/render_path.rb --ply FILE [options]"
  parser.on("--ply FILE") { |value| options[:ply] = value }
  parser.on("--output PATH") { |value| options[:output] = value }
  parser.on("--frames N", Integer) { |value| options[:frames] = value }
  parser.on("--width N", Integer) { |value| options[:width] = value }
  parser.on("--height N", Integer) { |value| options[:height] = value }
end.parse!
abort "error: --ply is required" unless options[:ply]

def normalize(vector)
  length = Math.sqrt(vector.sum { |value| value * value })
  vector.map { |value| value / length }
end

def cross(left, right)
  [
    (left[1] * right[2]) - (left[2] * right[1]),
    (left[2] * right[0]) - (left[0] * right[2]),
    (left[0] * right[1]) - (left[1] * right[0])
  ]
end

def look_at(position, target)
  forward = normalize(target.zip(position).map { |target_value, position_value| target_value - position_value })
  right = normalize(cross([0.0, 1.0, 0.0], forward))
  up = cross(forward, right)
  rotation = [right, up, forward]
  view = Numo::SFloat.eye(4)
  view[0...3, 0...3] = Numo::SFloat.cast(rotation)
  view[0...3, 3] = Numo::SFloat.cast(rotation.map { |row| -row.zip(position).sum { |a, b| a * b } })
  view
end

def sh_coefficients(params)
  direct = params.fetch(:sh0)
  rest = params.fetch(:shN)
  output = direct.class.zeros(direct.shape[0], direct.shape[1] + rest.shape[1], 3)
  output[true, 0...direct.shape[1], true] = direct
  output[true, direct.shape[1]...output.shape[1], true] = rest unless rest.shape[1].zero?
  output
end

params = Gsplat::IO::Ply.read(options.fetch(:ply))
center = params.fetch(:means).mean(axis: 0).to_a
distances = Numo::NMath.sqrt(((params.fetch(:means) - Numo::SFloat.cast(center))**2).sum(axis: 1))
radius = [distances.max.to_f * 2.0, 1.0].max
colors = sh_coefficients(params)
degree = [Math.sqrt(colors.shape[1]).to_i - 1, 4].min
focal = [options.fetch(:width), options.fetch(:height)].max.to_f
intrinsics = Numo::SFloat[
  [[focal, 0, options.fetch(:width) / 2.0],
   [0, focal, options.fetch(:height) / 2.0],
   [0, 0, 1]]
]
FileUtils.mkdir_p(options.fetch(:output))

options.fetch(:frames).times do |frame|
  angle = 2 * Math::PI * frame / options.fetch(:frames)
  position = [
    center[0] + (radius * Math.cos(angle)),
    center[1] + (radius * 0.25),
    center[2] + (radius * Math.sin(angle))
  ]
  view = look_at(position, center).reshape(1, 4, 4)
  rendered, = Gsplat.rasterization(
    means: params.fetch(:means),
    quats: params.fetch(:quats),
    scales: Numo::NMath.exp(params.fetch(:scales)),
    opacities: 1.0 / (1.0 + Numo::NMath.exp(-params.fetch(:opacities))),
    colors: colors,
    viewmats: view,
    ks: intrinsics,
    width: options.fetch(:width),
    height: options.fetch(:height),
    sh_degree: degree
  )
  path = File.join(options.fetch(:output), format("frame_%04d.png", frame))
  Gsplat::IO::Image.write(path, rendered[0, true, true, true])
end
