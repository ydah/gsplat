# frozen_string_literal: true

module Gsplat
  module IO
    # Optional ruby-vips implementation.
    module VipsImageBackend
      module_function

      def available?
        require "vips"
        true
      rescue LoadError, StandardError
        false
      end

      def read(path)
        image = Vips::Image.new_from_file(path.to_s, access: :sequential)
        scale = { uchar: 255.0, ushort: 65_535.0 }.fetch(image.format.to_sym, 1.0)
        rgb = rgb_image(image).cast(:float)
        Numo::SFloat.from_binary(rgb.write_to_memory).reshape(rgb.height, rgb.width, 3) / scale
      end

      def write(path, array)
        bytes = quantized_bytes(array)
        image = Vips::Image.new_from_memory(bytes, array.shape[1], array.shape[0], 3, :uchar)
        image.write_to_file(path.to_s)
        path
      end

      def rgb_image(image)
        return image.extract_band(0, n: 3) if image.bands >= 3

        band = image.extract_band(0)
        band.bandjoin([band, band])
      end
      private_class_method :rgb_image

      def quantized_bytes(array)
        array.to_a.flatten.map { |value| (value.to_f.clamp(0.0, 1.0) * 255).round }.pack("C*")
      end
      private_class_method :quantized_bytes
    end

    # Portable chunky_png implementation.
    module ChunkyPngImageBackend
      module_function

      def available?
        require "chunky_png"
        true
      rescue LoadError
        false
      end

      def read(path)
        image = ChunkyPNG::Image.from_file(path.to_s)
        values = image.height.times.flat_map do |y_coord|
          image.width.times.flat_map do |x_coord|
            pixel = image[x_coord, y_coord]
            [ChunkyPNG::Color.r(pixel), ChunkyPNG::Color.g(pixel), ChunkyPNG::Color.b(pixel)]
          end
        end
        Numo::SFloat.cast(values).reshape(image.height, image.width, 3) / 255.0
      end

      def write(path, array)
        image = ChunkyPNG::Image.new(array.shape[1], array.shape[0])
        array.shape[0].times do |y_coord|
          array.shape[1].times do |x_coord|
            rgb = array[y_coord, x_coord, true].to_a.map do |value|
              (value.to_f.clamp(0.0, 1.0) * 255).round
            end
            image[x_coord, y_coord] = ChunkyPNG::Color.rgb(*rgb)
          end
        end
        image.save(path.to_s)
        path
      end
    end
  end
end
