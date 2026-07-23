# frozen_string_literal: true

module Gsplat
  module Compression
    # Per-channel linear quantization used by PNG parameter grids.
    module Quantizer
      module_function

      def encode(array, side, bits:)
        feature_count = array.size / array.shape[0]
        raise NotSupportedError, "PNG parameter grids support at most four channels" if feature_count > 4

        flat = array.reshape(array.shape[0], feature_count)
        mins, maxs = channel_extrema(flat, feature_count)
        limit = (1 << bits) - 1
        values = quantized_values(flat, mins, maxs, limit)
        type = bits <= 8 ? Numo::UInt8 : Numo::UInt16
        pixels = type.cast(values).reshape(side, side, feature_count)
        [pixels, metadata(array, mins, maxs, bits)]
      end

      def decode(pixels, metadata)
        shape = metadata.fetch("shape")
        feature_count = shape.drop(1).inject(1, :*)
        mins = metadata.fetch("mins")
        maxs = metadata.fetch("maxs")
        limit = (1 << metadata.fetch("bits")) - 1
        values = pixels.reshape(pixels.shape[0] * pixels.shape[1], feature_count).to_a.flat_map do |row|
          row.each_with_index.map do |value, channel|
            mins[channel] + ((value.to_f / limit) * (maxs[channel] - mins[channel]))
          end
        end
        numeric_type(metadata.fetch("dtype")).cast(values).reshape(*shape)
      end

      def channel_extrema(flat, feature_count)
        [
          feature_count.times.map { |channel| flat[true, channel].min.to_f },
          feature_count.times.map { |channel| flat[true, channel].max.to_f }
        ]
      end
      private_class_method :channel_extrema

      def quantized_values(flat, mins, maxs, limit)
        flat.to_a.flat_map do |row|
          row.each_with_index.map do |value, channel|
            span = maxs[channel] - mins[channel]
            span.zero? ? 0 : (((value - mins[channel]) / span) * limit).round.clamp(0, limit)
          end
        end
      end
      private_class_method :quantized_values

      def metadata(array, mins, maxs, bits)
        {
          "shape" => array.shape,
          "dtype" => dtype_name(array.class),
          "mins" => mins,
          "maxs" => maxs,
          "bits" => bits
        }
      end
      private_class_method :metadata

      def dtype_name(type)
        return "float32" if type == Numo::SFloat
        return "float64" if type == Numo::DFloat

        raise NotSupportedError, "compression requires float32 or float64 parameters"
      end

      def numeric_type(name)
        { "float32" => Numo::SFloat, "float64" => Numo::DFloat }.fetch(name) do
          raise NotSupportedError, "unsupported compressed dtype #{name.inspect}"
        end
      end
    end
  end
end
