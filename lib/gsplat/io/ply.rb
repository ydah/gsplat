# frozen_string_literal: true

require_relative "ply_reader"

module Gsplat
  module IO
    # Inria-compatible Gaussian-splat PLY input and output.
    module Ply
      # Leading vertex properties in the Inria layout.
      BASE_PROPERTIES = %w[x y z nx ny nz].freeze
      # Properties following spherical-harmonic coefficients.
      TRAILING_PROPERTIES = %w[opacity scale_0 scale_1 scale_2 rot_0 rot_1 rot_2 rot_3].freeze
      # Minimum property set required for a readable Gaussian model.
      REQUIRED_PROPERTIES = (%w[x y z] + %w[f_dc_0 f_dc_1 f_dc_2] + TRAILING_PROPERTIES).freeze

      module_function

      # Writes raw Gaussian parameters with the Inria property layout.
      #
      # @param target [String, #write] output path or binary IO
      # @param params [Hash] means, scales, quats, opacities, sh0 and shN
      # @param format [Symbol] :binary_little_endian or :ascii
      # @return [Integer] bytes written
      def write(target, params, format: :binary_little_endian)
        arrays = normalize_params(params)
        rows = parameter_rows(arrays).select { |row| row.all?(&:finite?) }
        properties = property_names(arrays.fetch(:shN).shape[1])
        header = build_header(format, rows.length, properties)
        payload = encode_rows(rows, format)
        write_bytes(target, header + payload)
      end

      # Reads Inria-layout Gaussian parameters from ASCII or little-endian PLY.
      #
      # @param source [String, #read] input path or IO
      # @return [Hash{Symbol=>Numo::SFloat}]
      def read(source)
        columns = PlyReader.decode(read_bytes(source))
        missing = REQUIRED_PROPERTIES - columns.keys
        raise Gsplat::Error, "missing PLY properties: #{missing.join(', ')}" unless missing.empty?

        build_params(columns)
      end

      def normalize_params(params)
        required = %i[means scales quats opacities sh0 shN]
        missing = required - params.keys
        raise ArgumentError, "missing PLY params: #{missing.join(', ')}" unless missing.empty?

        arrays = required.to_h { |key| [key, Ops::TensorOps.data(params.fetch(key))] }
        count = arrays.fetch(:means).shape[0]
        expected = {
          means: [count, 3],
          scales: [count, 3],
          quats: [count, 4],
          opacities: [count],
          sh0: [count, 1, 3]
        }
        invalid = expected.find { |key, shape| arrays.fetch(key).shape != shape }
        if invalid
          raise ShapeError,
                "expected #{invalid[0]} #{invalid[1].inspect}, got #{arrays.fetch(invalid[0]).shape.inspect}"
        end

        shn = arrays.fetch(:shN)
        unless shn.ndim == 3 && shn.shape[0] == count && shn.shape[2] == 3
          raise ShapeError, "expected shN [#{count},K,3], got #{shn.shape.inspect}"
        end

        arrays
      end
      private_class_method :normalize_params

      def parameter_rows(arrays)
        count = arrays.fetch(:means).shape[0]
        count.times.map do |index|
          [
            *arrays.fetch(:means)[index, true].to_a.map(&:to_f),
            0.0, 0.0, 0.0,
            *channel_major_sh(arrays.fetch(:sh0), index),
            *channel_major_sh(arrays.fetch(:shN), index),
            arrays.fetch(:opacities)[index].to_f,
            *arrays.fetch(:scales)[index, true].to_a.map(&:to_f),
            *arrays.fetch(:quats)[index, true].to_a.map(&:to_f)
          ]
        end
      end
      private_class_method :parameter_rows

      def channel_major_sh(array, index)
        3.times.flat_map do |channel|
          array.shape[1].times.map { |coefficient| array[index, coefficient, channel].to_f }
        end
      end
      private_class_method :channel_major_sh

      def property_names(rest_count)
        direct = 3.times.map { |index| "f_dc_#{index}" }
        rest = (rest_count * 3).times.map { |index| "f_rest_#{index}" }
        BASE_PROPERTIES + direct + rest + TRAILING_PROPERTIES
      end
      private_class_method :property_names

      def build_header(format, count, properties)
        unless %i[binary_little_endian ascii].include?(format)
          raise NotSupportedError, "unsupported PLY output format #{format.inspect}"
        end

        lines = ["ply", "format #{format} 1.0", "element vertex #{count}"]
        lines.concat(properties.map { |name| "property float #{name}" })
        lines << "end_header"
        "#{lines.join("\n")}\n".b
      end
      private_class_method :build_header

      def encode_rows(rows, format)
        return rows.flatten.pack("e*") if format == :binary_little_endian

        rows.map { |row| row.map { |value| format("%.9g", value) }.join(" ") }.join("\n").concat("\n").b
      end
      private_class_method :encode_rows

      def build_params(columns)
        count = columns.fetch("x").length
        validate_column_lengths!(columns, count)
        rest_names = indexed_names(columns, "f_rest")
        raise Gsplat::Error, "f_rest property count must be divisible by 3" unless (rest_names.length % 3).zero?

        rest_count = rest_names.length / 3
        {
          means: matrix(columns, %w[x y z]),
          scales: matrix(columns, %w[scale_0 scale_1 scale_2]),
          quats: matrix(columns, %w[rot_0 rot_1 rot_2 rot_3]),
          opacities: Numo::SFloat.cast(columns.fetch("opacity")),
          sh0: sh_array(columns, 1, %w[f_dc_0 f_dc_1 f_dc_2]),
          shN: sh_array(columns, rest_count, rest_names)
        }
      end
      private_class_method :build_params

      def matrix(columns, names)
        values = columns.fetch(names.first).length.times.flat_map do |index|
          names.map { |name| columns.fetch(name)[index] }
        end
        Numo::SFloat.cast(values).reshape(columns.fetch(names.first).length, names.length)
      end
      private_class_method :matrix

      def sh_array(columns, coefficient_count, names)
        count = columns.fetch("x").length
        output = Numo::SFloat.zeros(count, coefficient_count, 3)
        3.times do |channel|
          coefficient_count.times do |coefficient|
            name = names.fetch((channel * coefficient_count) + coefficient)
            output[true, coefficient, channel] = Numo::SFloat.cast(columns.fetch(name))
          end
        end
        output
      end
      private_class_method :sh_array

      def indexed_names(columns, prefix)
        names = columns.keys.grep(/\A#{Regexp.escape(prefix)}_\d+\z/)
        names.sort_by { |name| Integer(name.split("_").last, 10) }
      end
      private_class_method :indexed_names

      def validate_column_lengths!(columns, count)
        invalid = columns.find { |_name, values| values.length != count }
        raise Gsplat::Error, "inconsistent PLY column #{invalid[0]}" if invalid
      end
      private_class_method :validate_column_lengths!

      def read_bytes(source)
        source.respond_to?(:read) ? source.read.b : File.binread(source)
      end
      private_class_method :read_bytes

      def write_bytes(target, data)
        target.respond_to?(:write) ? target.write(data) : File.binwrite(target, data)
      end
      private_class_method :write_bytes
    end
  end
end
