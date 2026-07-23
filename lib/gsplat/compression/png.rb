# frozen_string_literal: true

require "fileutils"
require "json"

require_relative "grid_sort"
require_relative "kmeans"
require_relative "png_codec"
require_relative "quantizer"

module Gsplat
  # Parameter codecs for compact Gaussian model storage.
  module Compression
    # Self-contained PNG and K-means Gaussian parameter compression.
    class Png
      # Parameter names encoded as 8-bit PNG images.
      PNG8_PARAMETERS = %w[scales quats opacities sh0].freeze

      attr_reader :use_sort, :kmeans_clusters, :kmeans_iterations, :sh_quantization

      def initialize(use_sort: true, kmeans_clusters: 256, kmeans_iterations: 8, sh_quantization: 6)
        @use_sort = use_sort
        @kmeans_clusters = Integer(kmeans_clusters)
        @kmeans_iterations = Integer(kmeans_iterations)
        @sh_quantization = Integer(sh_quantization)
        validate_options!
      end

      # Compresses Gaussian arrays into an upstream-compatible directory.
      #
      # @param directory [String] output directory
      # @param parameters [Hash{String, Symbol=>Numo::NArray}] arrays with a shared first dimension
      # @return [String] the output directory
      def compress(directory, parameters)
        FileUtils.mkdir_p(directory)
        values, side = GridSort.prepare(parameters, use_sort: use_sort)
        values["means"] = log_transform(values.fetch("means"))
        values["quats"] = Math::Quaternion.normalize(values.fetch("quats"))
        metadata = values.to_h do |name, value|
          validate_name!(name)
          [name, compress_parameter(directory, name, value, side)]
        end
        File.write(File.join(directory, "meta.json"), "#{JSON.pretty_generate(metadata)}\n")
        directory
      end

      # Restores Gaussian arrays from a compressed directory.
      #
      # @param directory [String] directory produced by {#compress}
      # @return [Hash{Symbol=>Numo::NArray}]
      def decompress(directory)
        metadata = JSON.parse(File.read(File.join(directory, "meta.json")))
        values = metadata.to_h do |name, entry|
          validate_name!(name)
          [name.to_sym, decompress_parameter(directory, name, entry)]
        end
        values[:means] = inverse_log_transform(values.fetch(:means))
        values
      end

      private

      def compress_parameter(directory, name, value, side)
        return compress_png16(directory, name, value, side) if name == "means"
        return compress_png8(directory, name, value, side) if PNG8_PARAMETERS.include?(name)
        if name == "shN"
          return KMeans.compress(
            File.join(directory, "shN.npz"),
            value,
            clusters: kmeans_clusters,
            iterations: kmeans_iterations,
            quantization: sh_quantization
          )
        end

        IO::Npy.write_npz(File.join(directory, "#{name}.npz"), { arr: value })
        { "shape" => value.shape, "dtype" => Quantizer.dtype_name(value.class), "codec" => "npz" }
      end

      def decompress_parameter(directory, name, metadata)
        return decompress_png16(directory, name, metadata) if name == "means"
        return decompress_png8(directory, name, metadata) if PNG8_PARAMETERS.include?(name)
        return KMeans.decompress(File.join(directory, "shN.npz"), metadata) if name == "shN"

        IO::Npy.read_npz(File.join(directory, "#{name}.npz")).fetch("arr")
      end

      def compress_png8(directory, name, value, side)
        pixels, metadata = Quantizer.encode(value, side, bits: 8)
        PngCodec.write(File.join(directory, "#{name}.png"), pixels)
        metadata
      end

      def decompress_png8(directory, name, metadata)
        Quantizer.decode(PngCodec.read(File.join(directory, "#{name}.png")), metadata)
      end

      def compress_png16(directory, name, value, side)
        pixels, metadata = Quantizer.encode(value, side, bits: 16)
        shape = pixels.shape
        values = pixels.to_a.flatten
        lower = Numo::UInt8.cast(values.map { |entry| entry & 0xff }).reshape(*shape)
        upper = Numo::UInt8.cast(values.map { |entry| entry >> 8 }).reshape(*shape)
        PngCodec.write(File.join(directory, "#{name}_l.png"), lower)
        PngCodec.write(File.join(directory, "#{name}_u.png"), upper)
        metadata
      end

      def decompress_png16(directory, name, metadata)
        lower = PngCodec.read(File.join(directory, "#{name}_l.png")).to_a.flatten
        upper = PngCodec.read(File.join(directory, "#{name}_u.png")).to_a.flatten
        side = ::Math.sqrt(metadata.fetch("shape").first).to_i
        shape = [side, side, metadata.fetch("shape").drop(1).inject(1, :*)]
        pixels = Numo::UInt16.cast(
          lower.zip(upper).map { |low, high| low | (high << 8) }
        ).reshape(*shape)
        Quantizer.decode(pixels, metadata)
      end

      def log_transform(values)
        output = Numo::NMath.log(1 + values.abs)
        negative = values.lt(0)
        output[negative] *= -1 if negative.any?
        output
      end

      def inverse_log_transform(values)
        output = Numo::NMath.exp(values.abs) - 1
        negative = values.lt(0)
        output[negative] *= -1 if negative.any?
        output
      end

      def validate_name!(name)
        return if name.match?(/\A[a-zA-Z][a-zA-Z0-9_]*\z/)

        raise ArgumentError, "invalid compression parameter name #{name.inspect}"
      end

      def validate_options!
        raise ArgumentError, "kmeans_clusters must be between 1 and 65536" unless (1..65_536).cover?(kmeans_clusters)
        raise ArgumentError, "kmeans_iterations must be positive" unless kmeans_iterations.positive?
        raise ArgumentError, "sh_quantization must be between 1 and 8" unless (1..8).cover?(sh_quantization)
      end
    end
  end
end
