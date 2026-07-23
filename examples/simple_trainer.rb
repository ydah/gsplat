#!/usr/bin/env ruby
# frozen_string_literal: true

require "fileutils"
require "optparse"
require "gsplat"

options = {
  data: nil,
  output_dir: "results",
  max_steps: 30_000,
  data_factor: 1,
  strategy: "default"
}
OptionParser.new do |parser|
  parser.banner = "Usage: bundle exec ruby examples/simple_trainer.rb --data PATH [options]"
  parser.on("--data PATH", "COLMAP dataset root") { |value| options[:data] = value }
  parser.on("--output PATH") { |value| options[:output_dir] = value }
  parser.on("--steps N", Integer) { |value| options[:max_steps] = value }
  parser.on("--data-factor N", Integer) { |value| options[:data_factor] = value }
  parser.on("--strategy NAME", %w[default mcmc]) { |value| options[:strategy] = value }
end.parse!
abort "error: --data is required" unless options[:data]

scene = Gsplat::Training::Scene.from_colmap(
  options.fetch(:data),
  data_factor: options.fetch(:data_factor)
)
config_options = {
  max_steps: options.fetch(:max_steps),
  output_dir: options.fetch(:output_dir),
  eval_steps: [options.fetch(:max_steps)],
  save_steps: [options.fetch(:max_steps)]
}
config_options.merge!(opacity_reg: 0.01, scale_reg: 0.01) if options.fetch(:strategy) == "mcmc"
config = Gsplat::Training::Config.new(**config_options)
strategy = if options.fetch(:strategy) == "mcmc"
             Gsplat::Strategy::MCMC.new
           else
             Gsplat::Strategy::Default.new
           end
trainer = Gsplat::Training::Trainer.new(scene: scene, config: config, strategy: strategy)
result = trainer.train
FileUtils.mkdir_p(options.fetch(:output_dir))
Gsplat::IO::Ply.write(
  File.join(options.fetch(:output_dir), "splats.ply"),
  trainer.params
)
warn format(
  "training complete: step=%<step>d PSNR=%<psnr>.2f SSIM=%<ssim>.4f",
  step: result.step,
  psnr: result.final_metrics.fetch(:psnr),
  ssim: result.final_metrics.fetch(:ssim)
)
