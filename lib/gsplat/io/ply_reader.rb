# frozen_string_literal: true

require "stringio"

module Gsplat
  module IO
    # Header and scalar payload decoder for PLY files.
    module PlyReader
      Element = Data.define(:name, :count, :properties)
      Property = Data.define(:name, :type, :count_type)
      TYPE_FORMATS = {
        "char" => ["c", 1], "int8" => ["c", 1],
        "uchar" => ["C", 1], "uint8" => ["C", 1],
        "short" => ["s<", 2], "int16" => ["s<", 2],
        "ushort" => ["S<", 2], "uint16" => ["S<", 2],
        "int" => ["l<", 4], "int32" => ["l<", 4],
        "uint" => ["L<", 4], "uint32" => ["L<", 4],
        "float" => ["e", 4], "float32" => ["e", 4],
        "double" => ["E", 8], "float64" => ["E", 8]
      }.freeze

      module_function

      def decode(data)
        header, payload = split_header(data.b)
        format, elements = parse_header(header)
        columns = vertex_columns(elements)
        reader = payload_reader(payload, format)
        decode_elements(reader, elements, columns)
        columns
      end

      def split_header(data)
        match = /\A.*?end_header\r?\n/m.match(data)
        raise Gsplat::Error, "PLY end_header is missing" unless match

        [match[0], data.byteslice(match[0].bytesize..)]
      end
      private_class_method :split_header

      def parse_header(header)
        lines = header.lines(chomp: true).map { |line| line.delete_suffix("\r") }
        raise Gsplat::Error, "invalid PLY magic" unless lines.shift == "ply"

        format = nil
        elements = []
        lines.each do |line|
          parts = line.split
          case parts.first
          when "format" then format = parse_format(parts)
          when "element" then elements << parse_element(parts)
          when "property" then elements.last.properties << parse_property(parts)
          end
        end
        raise Gsplat::Error, "PLY format declaration is missing" unless format
        raise Gsplat::Error, "PLY vertex element is missing" unless elements.any? { |item| item.name == "vertex" }

        [format, elements]
      end
      private_class_method :parse_header

      def parse_format(parts)
        raise NotSupportedError, "unsupported PLY version #{parts[2]}" unless parts[2] == "1.0"
        return parts[1].to_sym if %w[ascii binary_little_endian].include?(parts[1])

        raise NotSupportedError, "unsupported PLY format #{parts[1].inspect}"
      end
      private_class_method :parse_format

      def parse_element(parts)
        raise Gsplat::Error, "invalid PLY element declaration" unless parts.length == 3

        Element.new(name: parts[1], count: Integer(parts[2], 10), properties: [])
      end
      private_class_method :parse_element

      def parse_property(parts)
        if parts[1] == "list"
          validate_type!(parts[2])
          validate_type!(parts[3])
          return Property.new(name: parts[4], type: parts[3], count_type: parts[2])
        end
        validate_type!(parts[1])
        Property.new(name: parts[2], type: parts[1], count_type: nil)
      end
      private_class_method :parse_property

      def validate_type!(type)
        return if TYPE_FORMATS.key?(type)

        raise NotSupportedError, "unsupported PLY property type #{type.inspect}"
      end
      private_class_method :validate_type!

      def vertex_columns(elements)
        vertex = elements.find { |element| element.name == "vertex" }
        vertex.properties.reject(&:count_type).to_h { |property| [property.name, []] }
      end
      private_class_method :vertex_columns

      def payload_reader(payload, format)
        return payload.split.each if format == :ascii

        StringIO.new(payload)
      end
      private_class_method :payload_reader

      def decode_elements(reader, elements, columns)
        elements.each do |element|
          element.count.times do
            element.properties.each do |property|
              value = read_property(reader, property)
              columns[property.name] << value if element.name == "vertex" && !property.count_type
            end
          end
        end
      rescue EOFError, StopIteration
        raise Gsplat::Error, "truncated PLY payload"
      end
      private_class_method :decode_elements

      def read_property(reader, property)
        return read_scalar(reader, property.type) unless property.count_type

        count = read_scalar(reader, property.count_type).to_i
        Array.new(count) { read_scalar(reader, property.type) }
      end
      private_class_method :read_property

      def read_scalar(reader, type)
        return Float(reader.next) unless reader.is_a?(StringIO)

        format, width = TYPE_FORMATS.fetch(type)
        bytes = reader.read(width)
        raise EOFError unless bytes&.bytesize == width

        bytes.unpack1(format)
      end
      private_class_method :read_scalar
    end
  end
end
