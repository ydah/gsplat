# frozen_string_literal: true

module Gsplat
  module Strategy
    # Structural parameter edits synchronized with Adam moment buffers.
    module Ops
      module_function

      def duplicate!(params, optimizers, mask)
        indices = selected_indices(mask, parameter_count(params))
        return 0 if indices.empty?

        prepare_optimizers!(optimizers)
        params.each do |key, variable|
          appended = select_rows(variable.data, indices)
          variable.replace_data!(append_rows(variable.data, appended))
          optimizers.fetch(key).append!(indices.size)
        end
        indices.size
      end

      def remove!(params, optimizers, mask)
        validate_mask!(mask, parameter_count(params))
        keep = Numo::Bit.cast(mask).eq(0)
        removed = mask.count_true
        return 0 if removed.zero?

        prepare_optimizers!(optimizers)
        params.each do |key, variable|
          variable.replace_data!(select_rows(variable.data, keep))
          optimizers.fetch(key).select!(keep)
        end
        removed
      end

      def split!(params, optimizers, mask, rng: Gsplat.rng)
        count = parameter_count(params)
        indices = selected_indices(mask, count)
        return 0 if indices.empty?

        children = split_children(params, indices, rng)
        keep = Numo::Bit.cast(mask).eq(0)
        prepare_optimizers!(optimizers)
        params.each do |key, variable|
          retained = select_rows(variable.data, keep)
          variable.replace_data!(append_rows(retained, children.fetch(key)))
          optimizer = optimizers.fetch(key)
          optimizer.select!(keep)
          optimizer.append!(indices.size * 2)
        end
        indices.size
      end

      def reset_opacity!(params, optimizers, maximum:)
        raise ArgumentError, "maximum opacity must be in (0,1)" unless maximum.positive? && maximum < 1

        variable = params.fetch(:opacities)
        cap = ::Math.log(maximum / (1 - maximum))
        values = variable.data.dup
        above = values.gt(cap)
        values[above] = cap if above.any?
        variable.replace_data!(values)
        optimizer = optimizers.fetch(:opacities)
        optimizer.state
        optimizer.zero_state_at!(0...values.shape[0])
        above.count_true
      end

      def parameter_count(params)
        raise ArgumentError, "params must not be empty" if params.empty?

        params.values.first.data.shape[0]
      end
      private_class_method :parameter_count

      def selected_indices(mask, count)
        validate_mask!(mask, count)
        Numo::Bit.cast(mask).where.to_a
      end
      private_class_method :selected_indices

      def validate_mask!(mask, count)
        return if mask.is_a?(Numo::NArray) && mask.shape == [count]

        actual = mask.respond_to?(:shape) ? mask.shape.inspect : mask.class.to_s
        raise ShapeError, "expected mask [#{count}], got #{actual}"
      end
      private_class_method :validate_mask!

      def prepare_optimizers!(optimizers)
        optimizers.each_value(&:state)
      end
      private_class_method :prepare_optimizers!

      def select_rows(array, selection)
        array[*([selection] + Array.new(array.ndim - 1, true))].dup
      end
      private_class_method :select_rows

      def append_rows(array, rows)
        output = array.class.zeros(*([array.shape[0] + rows.shape[0]] + array.shape[1..]))
        output[*([0...array.shape[0]] + Array.new(array.ndim - 1, true))] = array
        output[*([array.shape[0]...output.shape[0]] + Array.new(array.ndim - 1, true))] = rows
        output
      end
      private_class_method :append_rows

      def split_children(params, indices, rng)
        children = params.to_h do |key, variable|
          selected = select_rows(variable.data, indices)
          doubled = repeat_rows(selected, 2)
          [key, doubled]
        end
        children[:scales] -= ::Math.log(1.6)
        children[:means] += split_offsets(params, indices, rng)
        children
      end
      private_class_method :split_children

      def repeat_rows(array, repeat)
        output = array.class.zeros(*([array.shape[0] * repeat] + array.shape[1..]))
        array.shape[0].times do |index|
          repeat.times do |copy|
            output[*([(index * repeat) + copy] + Array.new(array.ndim - 1, true))] =
              array[*([index] + Array.new(array.ndim - 1, true))]
          end
        end
        output
      end
      private_class_method :repeat_rows

      def split_offsets(params, indices, rng)
        quaternions = select_rows(params.fetch(:quats).data, indices)
        log_scales = select_rows(params.fetch(:scales).data, indices)
        rotations = Math::Quaternion.to_rotmat(quaternions)
        output = params.fetch(:means).data.class.zeros(indices.size * 2, 3)
        indices.size.times do |index|
          2.times do |child|
            noise = output.class.cast(Array.new(3) { normal_sample(rng) })
            scaled = noise * Numo::NMath.exp(log_scales[index, true])
            output[(index * 2) + child, true] = rotations[index, true, true].dot(scaled)
          end
        end
        output
      end
      private_class_method :split_offsets

      def normal_sample(rng)
        radius = ::Math.sqrt(-2 * ::Math.log([rng.rand, Float::MIN].max))
        radius * ::Math.cos(2 * ::Math::PI * rng.rand)
      end
      private_class_method :normal_sample
    end
  end
end
