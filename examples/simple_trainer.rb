#!/usr/bin/env ruby
# frozen_string_literal: true

require "fileutils"
require "optparse"
require "gsplat"

sample_data = File.expand_path("data/colmap", __dir__)
options = {
  data: sample_data,
  output_dir: nil,
  max_steps: nil,
  data_factor: 1,
  strategy: "default",
  dry_run: false
}
parser = OptionParser.new do |options_parser|
  options_parser.banner = "Usage: bundle exec ruby examples/simple_trainer.rb [options]"
  options_parser.on("--data PATH", "COLMAP dataset root (default: bundled sample)") { |value| options[:data] = value }
  options_parser.on("--output PATH") { |value| options[:output_dir] = value }
  options_parser.on("--steps N", Integer) { |value| options[:max_steps] = value }
  options_parser.on("--data-factor N", Integer) { |value| options[:data_factor] = value }
  options_parser.on("--strategy NAME", %w[default mcmc]) { |value| options[:strategy] = value }
  options_parser.on("--dry-run", "Validate the sparse model without loading images") { options[:dry_run] = true }
  options_parser.on_tail("-h", "--help", "Show this help") do
    puts options_parser
    exit
  end
end
parser.parse!

using_sample = File.expand_path(options.fetch(:data)) == sample_data
options[:output_dir] ||= using_sample ? "results/sample" : "results"
options[:max_steps] ||= using_sample ? 10 : 30_000

if options.fetch(:dry_run)
  dataset = Gsplat::IO::Colmap.read(options.fetch(:data), data_factor: options.fetch(:data_factor))
  warn format(
    "dataset valid: cameras=%<cameras>d images=%<images>d points=%<points>d",
    cameras: dataset.cameras.length,
    images: dataset.images.length,
    points: dataset.points3d.length
  )
  exit
end

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
