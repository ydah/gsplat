#!/usr/bin/env ruby
# frozen_string_literal: true

require "optparse"
require "gsplat"

options = {
  width: 64,
  height: 64,
  n_gaussians: 5_000,
  steps: 300,
  learning_rate: 20.0
}
OptionParser.new do |parser|
  parser.banner = "Usage: bundle exec ruby examples/fit_image.rb [options]"
  parser.on("--width N", Integer) { |value| options[:width] = value }
  parser.on("--height N", Integer) { |value| options[:height] = value }
  parser.on("--gaussians N", Integer) { |value| options[:n_gaussians] = value }
  parser.on("--steps N", Integer) { |value| options[:steps] = value }
  parser.on("--learning-rate X", Float) { |value| options[:learning_rate] = value }
end.parse!

steps = options.delete(:steps)
result = Gsplat::Training::ImageFitter.new(**options).fit(steps: steps)
warn format(
  "fit complete: PSNR %<initial>.2f dB -> %<final>.2f dB (%<steps>d steps)",
  initial: result.initial_psnr,
  final: result.final_psnr,
  steps: steps
)
