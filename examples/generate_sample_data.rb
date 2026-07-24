#!/usr/bin/env ruby
# frozen_string_literal: true

require "fileutils"
require "optparse"
require "gsplat"

# Builds the deterministic miniature scene used by the runnable examples.
module SampleData
  WIDTH = 16
  HEIGHT = 16
  CAMERA_TRANSLATIONS = [-0.12, 0.0, 0.12].freeze

  module_function

  def generate(root)
    sparse_directory = File.join(root, "colmap", "sparse", "0")
    image_directory = File.join(root, "colmap", "images")
    FileUtils.mkdir_p([sparse_directory, image_directory])

    parameters, colors = gaussian_parameters
    views, intrinsics = cameras
    images, = Gsplat.rasterization(
      means: parameters.fetch(:means),
      quats: parameters.fetch(:quats),
      scales: Numo::NMath.exp(parameters.fetch(:scales)),
      opacities: sigmoid(parameters.fetch(:opacities)),
      colors: parameters.fetch(:sh0),
      viewmats: views,
      ks: intrinsics,
      width: WIDTH,
      height: HEIGHT,
      sh_degree: 0
    )

    write_colmap_model(sparse_directory, colors, parameters.fetch(:means))
    write_images(image_directory, images)
    Gsplat::IO::Ply.write(File.join(root, "splats.ply"), parameters)
  end

  def gaussian_parameters
    means = Numo::SFloat.zeros(16, 3)
    colors = Numo::SFloat.zeros(16, 3)
    16.times do |index|
      x_coord = ((index % 4) * 2) + 0.5
      y_coord = ((index / 4) * 2) + 0.5
      fraction = index.fdiv(15)
      means[index, true] = [(x_coord - 4) / 4.0, (y_coord - 4) / 4.0, 2]
      colors[index, true] = [fraction, 0.2 + (0.5 * fraction), 1 - fraction]
    end

    parameters = {
      means: means,
      quats: identity_quaternions(16),
      scales: Numo::SFloat.ones(16, 3) * Math.log(0.2),
      opacities: Numo::SFloat.ones(16) * Math.log(0.85 / 0.15),
      sh0: Gsplat::Utils.rgb_to_sh(colors).reshape(16, 1, 3),
      shN: Numo::SFloat.zeros(16, 0, 3)
    }
    [parameters, colors]
  end
  private_class_method :gaussian_parameters

  def cameras
    count = CAMERA_TRANSLATIONS.length
    views = Numo::SFloat.zeros(count, 4, 4)
    intrinsics = Numo::SFloat.zeros(count, 3, 3)
    count.times do |index|
      views[index, true, true] = Numo::SFloat.eye(4)
      views[index, 0, 3] = CAMERA_TRANSLATIONS.fetch(index)
      intrinsics[index, true, true] = Numo::SFloat[[16, 0, 8], [0, 16, 8], [0, 0, 1]]
    end
    [views, intrinsics]
  end
  private_class_method :cameras

  def identity_quaternions(count)
    quaternions = Numo::SFloat.zeros(count, 4)
    quaternions[true, 0] = 1
    quaternions
  end
  private_class_method :identity_quaternions

  def sigmoid(values)
    1.0 / (1.0 + Numo::NMath.exp(-values))
  end
  private_class_method :sigmoid

  def write_colmap_model(directory, colors, means)
    File.write(
      File.join(directory, "cameras.txt"),
      "# One PINHOLE camera shared by all sample views.\n1 PINHOLE #{WIDTH} #{HEIGHT} 16 16 8 8\n"
    )
    image_records = CAMERA_TRANSLATIONS.each_index.map do |index|
      format(
        "%<id>d 1 0 0 0 %<tx>.6f 0 0 1 view_%<frame>03d.png\n",
        id: index + 1,
        tx: CAMERA_TRANSLATIONS[index],
        frame: index
      )
    end
    File.write(
      File.join(directory, "images.txt"),
      "# Each registered image is followed by an empty POINTS2D line.\n#{image_records.join("\n")}"
    )
    point_records = means.shape[0].times.map do |index|
      rgb = colors[index, true].to_a.map { |value| (value * 255).round }
      format(
        "%<id>d %<x>.9g %<y>.9g %<z>.9g %<r>d %<g>d %<b>d 0",
        id: index + 1,
        x: means[index, 0],
        y: means[index, 1],
        z: means[index, 2],
        r: rgb[0],
        g: rgb[1],
        b: rgb[2]
      )
    end
    File.write(File.join(directory, "points3D.txt"), "# Synthetic colored seed points.\n#{point_records.join("\n")}\n")
  end
  private_class_method :write_colmap_model

  def write_images(directory, images)
    CAMERA_TRANSLATIONS.each_index do |index|
      path = File.join(directory, format("view_%03d.png", index))
      Gsplat::IO::Image.write(path, images[index, true, true, true], backend: :chunky_png)
    end
  end
  private_class_method :write_images
end

options = { output: File.expand_path("data", __dir__) }
parser = OptionParser.new do |options_parser|
  options_parser.banner = "Usage: bundle exec ruby examples/generate_sample_data.rb [options]"
  options_parser.on("--output PATH", "Destination directory (default: examples/data)") do |value|
    options[:output] = value
  end
  options_parser.on_tail("-h", "--help", "Show this help") do
    puts options_parser
    exit
  end
end
parser.parse!

Gsplat.backend = :ruby
SampleData.generate(File.expand_path(options.fetch(:output)))
warn "sample data written to #{File.expand_path(options.fetch(:output))}"
