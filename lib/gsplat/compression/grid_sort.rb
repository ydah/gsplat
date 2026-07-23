# frozen_string_literal: true

module Gsplat
  module Compression
    # Deterministic Morton ordering for square Gaussian parameter grids.
    module GridSort
      QUANTIZATION_MAX = 1023

      module_function

      def prepare(parameters, use_sort: true)
        values = parameters.to_h { |name, value| [name.to_s, data(value).dup] }
        validate_lengths!(values)
        side = ::Math.sqrt(values.fetch("means").shape[0]).floor
        raise ArgumentError, "at least one Gaussian is required" if side.zero?

        values = crop_to_square(values, side * side)
        values = reorder(values, morton_order(values.fetch("means"))) if use_sort
        [values, side]
      end

      def crop_to_square(values, count)
        return values if values.fetch("means").shape[0] == count

        opacities = values.fetch("opacities").reshape(values.fetch("opacities").shape[0])
        keep = (0...opacities.size).sort_by { |index| [-opacities[index].to_f, index] }.first(count)
        reorder(values, keep)
      end
      private_class_method :crop_to_square

      def morton_order(means)
        coordinates = means.to_a
        mins = 3.times.map { |axis| coordinates.map { |point| point[axis] }.min }
        maxs = 3.times.map { |axis| coordinates.map { |point| point[axis] }.max }
        (0...coordinates.length).sort_by do |index|
          quantized = 3.times.map do |axis|
            span = maxs[axis] - mins[axis]
            span.zero? ? 0 : (((coordinates[index][axis] - mins[axis]) / span) * QUANTIZATION_MAX).round
          end
          [morton3(*quantized), index]
        end
      end

      def morton3(x_coord, y_coord, z_coord)
        code = 0
        10.times do |bit|
          code |= ((x_coord >> bit) & 1) << (3 * bit)
          code |= ((y_coord >> bit) & 1) << ((3 * bit) + 1)
          code |= ((z_coord >> bit) & 1) << ((3 * bit) + 2)
        end
        code
      end
      private_class_method :morton3

      def reorder(values, order)
        indices = Numo::Int32.cast(order)
        values.transform_values { |value| value[indices, *Array.new(value.ndim - 1, true)].dup }
      end
      private_class_method :reorder

      def validate_lengths!(values)
        required = %w[means scales quats opacities sh0 shN]
        missing = required - values.keys
        raise ArgumentError, "missing compression parameters: #{missing.join(', ')}" unless missing.empty?

        count = values.fetch("means").shape[0]
        invalid = values.find { |_name, value| !value.is_a?(Numo::NArray) || value.shape[0] != count }
        raise ShapeError, "all compression parameters must share first dimension #{count}" if invalid
        raise ShapeError, "expected means [N,3]" unless values.fetch("means").shape == [count, 3]
      end
      private_class_method :validate_lengths!

      def data(value)
        value.is_a?(Autograd::Variable) ? value.data : value
      end
      private_class_method :data
    end
  end
end
