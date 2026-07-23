# frozen_string_literal: true

require "stringio"

require_relative "zip_archive"

module Gsplat
  # File formats and dataset readers used by training and interchange.
  module IO
    # NumPy v1.0 NPY and NPZ interoperability.
    module Npy
      # Binary prefix defined by the NumPy NPY format.
      MAGIC = "\x93NUMPY".b
      # NPY format version emitted and accepted by this codec.
      VERSION = [1, 0].freeze
      # Header byte alignment used by NPY v1.0.
      HEADER_ALIGNMENT = 64
      # Mapping from supported Numo types to NumPy descriptors and pack formats.
      DESCRIPTORS = {
        Numo::SFloat => ["<f4", "e*", 4],
        Numo::DFloat => ["<f8", "E*", 8],
        Numo::UInt8 => ["|u1", "C*", 1],
        Numo::UInt16 => ["<u2", "S<*", 2],
        Numo::Int32 => ["<i4", "l<*", 4],
        Numo::Int64 => ["<i8", "q<*", 8],
        Numo::Bit => ["|b1", "C*", 1]
      }.freeze
      # Reverse descriptor-to-Numo type mapping.
      TYPES = DESCRIPTORS.to_h { |type, metadata| [metadata.first, type] }.freeze

      module_function

      # Reads an NPY v1.0 array.
      #
      # @param source [String, #read] path or binary IO
      # @return [Numo::NArray]
      def read(source)
        decode(read_bytes(source))
      end

      # Writes an NPY v1.0 array in C order.
      #
      # @param target [String, #write] path or binary IO
      # @param array [Numo::NArray]
      # @return [Integer] bytes written
      def write(target, array)
        write_bytes(target, encode(array))
      end

      # Reads all arrays from an NPZ archive.
      #
      # @param source [String, #read] path or binary IO
      # @return [Hash{String=>Numo::NArray}]
      def read_npz(source)
        ZipArchive.decode(read_bytes(source)).each_with_object({}) do |(name, data), arrays|
          next unless name.end_with?(".npy")

          arrays[name.delete_suffix(".npy")] = decode(data)
        end
      end

      # Writes arrays to an NPZ archive.
      #
      # @param target [String, #write] path or binary IO
      # @param arrays [Hash{String, Symbol=>Numo::NArray}]
      # @param compression [Symbol] :deflate or :stored
      # @return [Integer] bytes written
      def write_npz(target, arrays, compression: :deflate)
        entries = arrays.to_h do |name, array|
          normalized = normalize_entry_name(name)
          ["#{normalized}.npy", encode(array)]
        end
        write_bytes(target, ZipArchive.encode(entries, compression: compression))
      end

      # Returns the NumPy dtype descriptor for a Numo type.
      #
      # @param type [Class]
      # @return [String]
      def descriptor_for(type)
        DESCRIPTORS.fetch(type) do
          raise NotSupportedError, "unsupported Numo dtype #{type}"
        end.first
      end

      def encode(array)
        descriptor, pack_format, = DESCRIPTORS.fetch(array.class) do
          raise NotSupportedError, "unsupported Numo dtype #{array.class}"
        end
        header = build_header(descriptor, array.shape)
        values = array.to_a
        values = values.is_a?(Array) ? values.flatten : [values]

        MAGIC + VERSION.pack("C2") + [header.bytesize].pack("v") + header + values.pack(pack_format)
      end
      private_class_method :encode

      def decode(data)
        data = data.b
        raise Gsplat::Error, "invalid NPY magic" unless data.start_with?(MAGIC)

        version = byteslice!(data, 6, 2).unpack("C2")
        raise NotSupportedError, "unsupported NPY version #{version.join('.')}" unless version == VERSION

        header_length = byteslice!(data, 8, 2).unpack1("v")
        header = byteslice!(data, 10, header_length)
        descriptor, shape = parse_header(header)
        type, pack_format, byte_width = dtype_metadata(descriptor)
        count = shape.empty? ? 1 : shape.inject(:*)
        payload = byteslice!(data, 10 + header_length, count * byte_width)
        values = payload.unpack(pack_format)

        return type.cast(values.first) if shape.empty?

        type.cast(values).reshape(*shape)
      end
      private_class_method :decode

      def build_header(descriptor, shape)
        shape_text = if shape.empty?
                       ""
                     elsif shape.length == 1
                       "#{shape.first},"
                     else
                       shape.join(", ")
                     end
        dictionary = "{'descr': '#{descriptor}', 'fortran_order': False, 'shape': (#{shape_text}), }"
        padding = (HEADER_ALIGNMENT - ((10 + dictionary.bytesize + 1) % HEADER_ALIGNMENT)) % HEADER_ALIGNMENT
        header = "#{dictionary}#{' ' * padding}\n"
        raise NotSupportedError, "NPY v1.0 header exceeds 65535 bytes" if header.bytesize > 65_535

        header.b
      end
      private_class_method :build_header

      def parse_header(header)
        descriptor = header[/(?:'|")descr(?:'|"):\s*(?:'|")([^'"]+)(?:'|")/, 1]
        order = header[/(?:'|")fortran_order(?:'|"):\s*(True|False)/, 1]
        shape_text = header[/(?:'|")shape(?:'|"):\s*\(([^)]*)\)/, 1]
        raise Gsplat::Error, "invalid NPY header" unless descriptor && order && shape_text
        raise NotSupportedError, "Fortran order NPY arrays are unsupported" if order == "True"

        shape = shape_text.scan(/\d+/).map!(&:to_i)
        [descriptor, shape]
      end
      private_class_method :parse_header

      def dtype_metadata(descriptor)
        type = TYPES.fetch(descriptor) do
          raise NotSupportedError, "unsupported NumPy dtype #{descriptor.inspect}"
        end
        [type, *DESCRIPTORS.fetch(type).drop(1)]
      end
      private_class_method :dtype_metadata

      def normalize_entry_name(name)
        normalized = name.to_s.delete_suffix(".npy")
        if normalized.empty? || normalized.match?(%r{[/\\]}) || normalized.include?("\0")
          raise ArgumentError, "invalid NPZ entry name #{name.inspect}"
        end

        normalized
      end
      private_class_method :normalize_entry_name

      def read_bytes(source)
        return source.read.b if source.respond_to?(:read)

        File.binread(source)
      end
      private_class_method :read_bytes

      def write_bytes(target, data)
        return target.write(data) if target.respond_to?(:write)

        File.binwrite(target, data)
      end
      private_class_method :write_bytes

      def byteslice!(data, offset, length)
        slice = data.byteslice(offset, length)
        return slice if slice&.bytesize == length

        raise Gsplat::Error, "truncated NPY data"
      end
      private_class_method :byteslice!
    end
  end
end
