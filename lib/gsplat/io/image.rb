# frozen_string_literal: true

require_relative "image_backends"

module Gsplat
  module IO
    # RGB image input/output with vips-first optional backends.
    module Image
      BACKENDS = {
        vips: VipsImageBackend,
        chunky_png: ChunkyPngImageBackend
      }.freeze

      module_function

      # Reads an image as float32 [H,W,3] in the range 0..1.
      def read(path, backend: :auto)
        resolve_backend(backend).read(path)
      end

      # Writes a float [H,W,3] image, clamping values to 0..1.
      def write(path, array, backend: :auto)
        validate_array!(array)
        resolve_backend(backend).write(path, array)
      end

      def available_backends
        BACKENDS.filter_map { |name, implementation| name if implementation.available? }
      end

      def resolve_backend(name)
        if name == :auto
          selected = BACKENDS.values.find(&:available?)
          return selected if selected

          raise NotSupportedError, "image IO requires ruby-vips or chunky_png"
        end
        implementation = BACKENDS.fetch(name.to_sym) do
          raise ArgumentError, "unknown image backend #{name.inspect}"
        end
        return implementation if implementation.available?

        raise NotSupportedError, "image backend #{name} is unavailable"
      end
      private_class_method :resolve_backend

      def validate_array!(array)
        valid = array.is_a?(Numo::NArray) &&
                [Numo::SFloat, Numo::DFloat].include?(array.class) &&
                array.ndim == 3 && array.shape[2] == 3
        return if valid

        actual = array.respond_to?(:shape) ? array.shape.inspect : array.class
        raise ShapeError, "expected image [H,W,3] floating array, got #{actual}"
      end
      private_class_method :validate_array!
    end
  end
end
