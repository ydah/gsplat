#!/usr/bin/env ruby
# frozen_string_literal: true

require "json"
require "optparse"
require "gsplat"

SCENARIOS = {
  "raster" => "100k Gaussians at 800x800: forward and forward+backward",
  "fit_image" => "50k Gaussians at 512x512 for 2,000 optimization steps",
  "colmap" => "COLMAP capture training for 30,000 steps (requires --data)"
}.freeze

options = {
  scenario: "raster",
  backend: ENV.fetch("GSPLAT_BACKEND", "native"),
  gaussians: nil,
  width: nil,
  height: nil,
  steps: nil,
  iterations: 1,
  data: nil,
  output: "benchmark-results",
  quick: false,
  json: nil,
  list: false
}
parser = OptionParser.new do |opts|
  opts.banner = "Usage: ruby -Ilib benchmarks/bench_rasterize.rb [options]"
  opts.on("--scenario NAME", SCENARIOS.keys) { |value| options[:scenario] = value }
  opts.on("--backend NAME", %w[ruby native auto]) { |value| options[:backend] = value }
  opts.on("--gaussians N", Integer) { |value| options[:gaussians] = value }
  opts.on("--width N", Integer) { |value| options[:width] = value }
  opts.on("--height N", Integer) { |value| options[:height] = value }
  opts.on("--steps N", Integer) { |value| options[:steps] = value }
  opts.on("--iterations N", Integer) { |value| options[:iterations] = value }
  opts.on("--data PATH", "COLMAP dataset root") { |value| options[:data] = value }
  opts.on("--output PATH") { |value| options[:output] = value }
  opts.on("--json PATH", "Write machine-readable results") { |value| options[:json] = value }
  opts.on("--quick", "Use a CI-sized workload") { options[:quick] = true }
  opts.on("--list", "List the design workloads") { options[:list] = true }
  opts.on_tail("-h", "--help") do
    puts opts
    exit
  end
end
parser.parse!

raise ArgumentError, "iterations must be positive" unless options[:iterations].positive?

%i[gaussians width height steps].each do |name|
  value = options[name]
  raise ArgumentError, "#{name} must be positive" if value && !value.positive?
end

if options[:list]
  SCENARIOS.each { |name, description| puts "#{name}: #{description}" }
  exit
end

def measure(iterations, warmup: true)
  yield if warmup
  samples = Array.new(iterations) do
    started = Process.clock_gettime(Process::CLOCK_MONOTONIC)
    yield
    (Process.clock_gettime(Process::CLOCK_MONOTONIC) - started) * 1_000
  end
  { mean_ms: samples.sum / samples.length, min_ms: samples.min, max_ms: samples.max }
end

# rubocop:disable Metrics/AbcSize
def raster_benchmark(options)
  count = options[:gaussians] || (options[:quick] ? 64 : 100_000)
  width = options[:width] || (options[:quick] ? 32 : 800)
  height = options[:height] || (options[:quick] ? 24 : 800)
  rng = Random.new(42)
  means = Numo::SFloat.cast(Array.new(count) { [rng.rand - 0.5, rng.rand - 0.5, 2 + rng.rand] })
  quats = Numo::SFloat.zeros(count, 4)
  quats[true, 0] = 1
  scales = Numo::SFloat.ones(count, 3) * 0.03
  views = Numo::SFloat.eye(4).reshape(1, 4, 4)
  intrinsics = Numo::SFloat[[[52, 0, width / 2.0], [0, 52, height / 2.0], [0, 0, 1]]]
  radii, means2d, depths, conics, = Gsplat.fully_fused_projection(
    means, quats: quats, scales: scales, viewmats: views, ks: intrinsics,
           width: width, height: height
  )
  colors = Numo::SFloat.ones(1, count, 3) * 0.5
  opacities = Numo::SFloat.ones(1, count) * 0.2
  tile_size = 16
  tile_width = (width.to_f / tile_size).ceil
  tile_height = (height.to_f / tile_size).ceil
  _, keys, ids = Gsplat.isect_tiles(means2d, radii, depths, tile_size, tile_width, tile_height)
  offsets = Gsplat.isect_offset_encode(keys, 1, tile_width, tile_height)
  forward_args = [means2d, conics, colors, opacities, nil, nil,
                  width, height, tile_size, offsets, ids]
  forward = -> { Gsplat::Backend.dispatch(:rasterize_to_pixels_forward, *forward_args) }
  forward_backward = lambda do
    rendered, alphas, last_ids = forward.call
    Gsplat::Backend.dispatch(
      :rasterize_to_pixels_backward,
      *forward_args,
      alphas,
      last_ids,
      Numo::SFloat.ones(*rendered.shape),
      Numo::SFloat.ones(*alphas.shape),
      absgrad: true
    )
  end
  {
    workload: { gaussians: count, width: width, height: height, iterations: options[:iterations] },
    forward: measure(options[:iterations], &forward),
    forward_backward: measure(options[:iterations], &forward_backward)
  }
end
# rubocop:enable Metrics/AbcSize

def fit_image_benchmark(options)
  count = options[:gaussians] || (options[:quick] ? 16 : 50_000)
  width = options[:width] || (options[:quick] ? 8 : 512)
  height = options[:height] || (options[:quick] ? 8 : 512)
  steps = options[:steps] || (options[:quick] ? 1 : 2_000)
  result = nil
  timing = measure(1, warmup: false) do
    result = Gsplat::Training::ImageFitter.new(
      width: width, height: height, n_gaussians: count, dtype: Numo::SFloat
    ).fit(steps: steps)
  end
  {
    workload: { gaussians: count, width: width, height: height, steps: steps },
    total: timing,
    initial_psnr: result.initial_psnr,
    final_psnr: result.final_psnr
  }
end

def colmap_benchmark(options)
  raise OptionParser::MissingArgument, "--data is required for the colmap scenario" unless options[:data]

  steps = options[:steps] || (options[:quick] ? 1 : 30_000)
  scene = Gsplat::Training::Scene.from_colmap(options[:data])
  config = Gsplat::Training::Config.new(
    max_steps: steps, output_dir: options[:output], eval_steps: [], save_steps: []
  )
  result = nil
  timing = measure(1, warmup: false) do
    result = Gsplat::Training::Trainer.new(scene: scene, config: config).train
  end
  {
    workload: { cameras: scene.camera_count, initial_gaussians: scene.points.shape[0], steps: steps },
    total: timing,
    final_metrics: result.final_metrics
  }
end

Gsplat.backend = options[:backend]
result = case options[:scenario]
         when "raster" then raster_benchmark(options)
         when "fit_image" then fit_image_benchmark(options)
         when "colmap" then colmap_benchmark(options)
         end
report = {
  scenario: options[:scenario],
  backend: Gsplat.backend,
  ruby: RUBY_VERSION,
  platform: RUBY_PLATFORM,
  omp_threads: ENV.fetch("OMP_NUM_THREADS", nil),
  result: result
}
puts JSON.pretty_generate(report)
File.write(options[:json], "#{JSON.pretty_generate(report)}\n") if options[:json]
