# frozen_string_literal: true

module Gsplat
  module Strategy
    # MCMC-specific parameter edits synchronized with Adam state.
    module Ops
      module_function

      # rubocop:disable Metrics/AbcSize
      def relocate!(params, optimizers, mask, binoms:, **options)
        min_opacity = options.fetch(:min_opacity, 0.005)
        rng = options.fetch(:rng, Gsplat.rng)
        count = params.fetch(:means).data.shape[0]
        validate_mcmc_mask!(mask, count)
        dead = Numo::Bit.cast(mask).where.to_a
        return 0 if dead.empty?

        alive = Numo::Bit.cast(mask).eq(0).where.to_a
        raise Gsplat::Error, "cannot relocate when every Gaussian is dead" if alive.empty?

        weights = sigmoid_mcmc(params.fetch(:opacities).data[alive])
        sampled = weighted_indices(weights, dead.size, rng).map { |index| alive.fetch(index) }
        update_relocation_sources!(params, sampled, binoms, min_opacity)
        prepare_mcmc_optimizers!(optimizers)
        params.each do |key, variable|
          variable.data[*([dead] + Array.new(variable.data.ndim - 1, true))] =
            variable.data[*([sampled] + Array.new(variable.data.ndim - 1, true))]
          optimizers.fetch(key).zero_state_at!(sampled)
          optimizers.fetch(key).zero_state_at!(dead)
        end
        dead.size
      end
      # rubocop:enable Metrics/AbcSize

      def sample_add!(params, optimizers, count, binoms:, **options)
        min_opacity = options.fetch(:min_opacity, 0.005)
        rng = options.fetch(:rng, Gsplat.rng)
        raise ArgumentError, "count must be a non-negative integer" unless count.is_a?(Integer) && !count.negative?
        return 0 if count.zero?

        weights = sigmoid_mcmc(params.fetch(:opacities).data)
        sampled = weighted_indices(weights, count, rng)
        update_relocation_sources!(params, sampled, binoms, min_opacity)
        prepare_mcmc_optimizers!(optimizers)
        params.each do |key, variable|
          rows = variable.data[*([sampled] + Array.new(variable.data.ndim - 1, true))].dup
          variable.replace_data!(append_mcmc_rows(variable.data, rows))
          optimizers.fetch(key).append!(count)
        end
        count
      end

      # rubocop:disable Metrics/AbcSize
      def inject_position_noise!(params, scaler:, rng: Gsplat.rng)
        return params if scaler.zero?

        opacities = sigmoid_mcmc(params.fetch(:opacities).data)
        scales = Numo::NMath.exp(params.fetch(:scales).data)
        covariances, = Gsplat.quat_scale_to_covar_preci(
          params.fetch(:quats).data,
          scales,
          compute_preci: false
        )
        noise = params.fetch(:means).data.class.zeros(*params.fetch(:means).data.shape)
        noise.shape[0].times do |index|
          gate = 1.0 / (1.0 + ::Math.exp(-100.0 * ((1.0 - opacities[index].to_f) - 0.995)))
          standard = noise.class.cast(Array.new(3) { normal_mcmc_sample(rng) })
          noise[index, true] = covariances[index, true, true].dot(standard) * gate * scaler
        end
        params.fetch(:means).data[] = params.fetch(:means).data + noise
        params
      end
      # rubocop:enable Metrics/AbcSize

      def update_relocation_sources!(params, sampled, binoms, min_opacity)
        source_opacities = sigmoid_mcmc(params.fetch(:opacities).data[sampled])
        source_scales = Numo::NMath.exp(params.fetch(:scales).data[sampled, true])
        counts = sampled.tally
        ratios = Numo::Int32.cast(sampled.map { |index| counts.fetch(index) + 1 })
        opacities, scales = Gsplat.relocation(source_opacities, source_scales, ratios, binoms: binoms)
        epsilon = Numo::SFloat::EPSILON
        opacities = clamp_mcmc(opacities, min_opacity, 1.0 - epsilon)
        params.fetch(:opacities).data[sampled] = Numo::NMath.log(opacities / (1.0 - opacities))
        params.fetch(:scales).data[sampled, true] = Numo::NMath.log(scales)
      end
      private_class_method :update_relocation_sources!

      def weighted_indices(weights, count, rng)
        cumulative = weights.to_a.each_with_object([]) do |weight, sums|
          sums << (weight.to_f + (sums.last || 0.0))
        end
        valid_sum = cumulative.last&.positive? && cumulative.last.finite?
        raise Gsplat::Error, "sampling weights must have positive finite sum" unless valid_sum

        Array.new(count) do
          draw = rng.rand * cumulative.last
          cumulative.bsearch_index { |value| value > draw } || (cumulative.length - 1)
        end
      end
      private_class_method :weighted_indices

      def append_mcmc_rows(array, rows)
        output = array.class.zeros(*([array.shape[0] + rows.shape[0]] + array.shape[1..]))
        output[*([0...array.shape[0]] + Array.new(array.ndim - 1, true))] = array
        output[*([array.shape[0]...output.shape[0]] + Array.new(array.ndim - 1, true))] = rows
        output
      end
      private_class_method :append_mcmc_rows

      def prepare_mcmc_optimizers!(optimizers)
        optimizers.each_value(&:state)
      end
      private_class_method :prepare_mcmc_optimizers!

      def validate_mcmc_mask!(mask, count)
        return if mask.is_a?(Numo::NArray) && mask.shape == [count]

        actual = mask.respond_to?(:shape) ? mask.shape.inspect : mask.class.to_s
        raise ShapeError, "expected mask [#{count}], got #{actual}"
      end
      private_class_method :validate_mcmc_mask!

      def sigmoid_mcmc(values)
        1.0 / (1.0 + Numo::NMath.exp(-values))
      end
      private_class_method :sigmoid_mcmc

      def clamp_mcmc(values, minimum, maximum)
        output = values.dup
        output[output.lt(minimum)] = minimum
        output[output.gt(maximum)] = maximum
        output
      end
      private_class_method :clamp_mcmc

      def normal_mcmc_sample(rng)
        radius = ::Math.sqrt(-2.0 * ::Math.log([rng.rand, Float::MIN].max))
        radius * ::Math.cos(2.0 * ::Math::PI * rng.rand)
      end
      private_class_method :normal_mcmc_sample
    end
  end
end
