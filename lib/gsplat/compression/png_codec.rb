# frozen_string_literal: true

require "zlib"

module Gsplat
  module Compression
    # Dependency-free PNG codec for 8-bit parameter grids.
    module PngCodec
      SIGNATURE = "\x89PNG\r\n\x1A\n".b
      COLOR_TYPES = { 1 => 0, 2 => 4, 3 => 2, 4 => 6 }.freeze
      CHANNELS = COLOR_TYPES.invert.freeze

      module_function

      def write(path, pixels)
        validate_pixels!(pixels)
        height, width, channels = pixels.shape
        rows = pixels.to_a.map { |row| "\0".b + row.flatten.pack("C*") }.join
        header = [width, height, 8, COLOR_TYPES.fetch(channels), 0, 0, 0].pack("NNC5")
        File.binwrite(
          path,
          SIGNATURE + chunk("IHDR", header) + chunk("IDAT", Zlib::Deflate.deflate(rows)) + chunk("IEND", "")
        )
        path
      end

      def read(path)
        data = File.binread(path)
        raise Gsplat::Error, "invalid PNG signature" unless data.start_with?(SIGNATURE)

        width, height, channels, compressed = parse_chunks(data.byteslice(SIGNATURE.bytesize..))
        bytes = unfilter(Zlib::Inflate.inflate(compressed), width, height, channels)
        Numo::UInt8.from_binary(bytes).reshape(height, width, channels)
      end

      def validate_pixels!(pixels)
        valid = pixels.is_a?(Numo::UInt8) && pixels.ndim == 3 && COLOR_TYPES.key?(pixels.shape[-1])
        return if valid

        actual = pixels.respond_to?(:shape) ? pixels.shape.inspect : pixels.class
        raise ShapeError, "expected UInt8 PNG pixels [H,W,C] with C=1..4, got #{actual}"
      end
      private_class_method :validate_pixels!

      def chunk(type, payload)
        type = type.b
        [payload.bytesize].pack("N") + type + payload + [Zlib.crc32(type + payload)].pack("N")
      end
      private_class_method :chunk

      def parse_chunks(data)
        width = height = channels = nil
        compressed = +"".b
        until data.empty?
          length = data.byteslice(0, 4)&.unpack1("N")
          raise Gsplat::Error, "truncated PNG chunk" unless length && data.bytesize >= length + 12

          type = data.byteslice(4, 4)
          payload = data.byteslice(8, length)
          expected_crc = data.byteslice(8 + length, 4).unpack1("N")
          raise Gsplat::Error, "invalid PNG CRC" unless Zlib.crc32(type + payload) == expected_crc

          width, height, channels = parse_header(payload) if type == "IHDR"
          compressed << payload if type == "IDAT"
          data = data.byteslice((length + 12)..) || +"".b
          break if type == "IEND"
        end
        raise Gsplat::Error, "PNG is missing IHDR or IDAT" unless width && !compressed.empty?

        [width, height, channels, compressed]
      end
      private_class_method :parse_chunks

      def parse_header(payload)
        width, height, depth, color_type, compression, filter, interlace = payload.unpack("NNC5")
        valid = depth == 8 && CHANNELS.key?(color_type) && compression.zero? && filter.zero? && interlace.zero?
        raise NotSupportedError, "only non-interlaced 8-bit PNG is supported" unless valid

        [width, height, CHANNELS.fetch(color_type)]
      end
      private_class_method :parse_header

      def unfilter(data, width, height, channels)
        stride = width * channels
        output = String.new(capacity: stride * height, encoding: Encoding::BINARY)
        previous = Array.new(stride, 0)
        offset = 0
        height.times do
          filter = data.getbyte(offset)
          encoded = data.byteslice(offset + 1, stride)&.bytes
          raise Gsplat::Error, "truncated PNG scanline" unless encoded&.length == stride

          row = reconstruct_row(encoded, previous, channels, filter)
          output << row.pack("C*")
          previous = row
          offset += stride + 1
        end
        output
      end
      private_class_method :unfilter

      def reconstruct_row(encoded, previous, channels, filter)
        raise NotSupportedError, "unsupported PNG filter #{filter}" unless (0..4).cover?(filter)

        encoded.each_with_index.map do |value, index|
          left = index >= channels ? encoded[index - channels] : 0
          upper = previous[index]
          upper_left = index >= channels ? previous[index - channels] : 0
          prediction = filter_prediction(filter, left, upper, upper_left)
          encoded[index] = (value + prediction) & 0xff
        end
      end
      private_class_method :reconstruct_row

      def filter_prediction(filter, left, upper, upper_left)
        return 0 if filter.zero?
        return left if filter == 1
        return upper if filter == 2
        return (left + upper) / 2 if filter == 3

        paeth(left, upper, upper_left)
      end
      private_class_method :filter_prediction

      def paeth(left, upper, upper_left)
        estimate = left + upper - upper_left
        distances = [(estimate - left).abs, (estimate - upper).abs, (estimate - upper_left).abs]
        [left, upper, upper_left][distances.each_with_index.min.last]
      end
      private_class_method :paeth
    end
  end
end
