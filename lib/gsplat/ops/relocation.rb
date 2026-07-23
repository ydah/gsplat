# frozen_string_literal: true

# Closed-form relocation primitive and public wrapper.
module Gsplat
  module Ops
    # Closed-form opacity and scale update from 3DGS-MCMC equation 9.
    module Relocation
      # Default maximum split ratio represented by {.binomial_table}.
      DEFAULT_N_MAX = 51

      module_function

      # Builds the binomial coefficient lookup used by relocation.
      #
      # @param n_max [Integer] maximum split ratio, inclusive
      # @param dtype [Class] Numo floating-point class
      # @return [Numo::NArray] [n_max,n_max]
      def binomial_table(n_max: DEFAULT_N_MAX, dtype: Numo::SFloat)
        raise ArgumentError, "n_max must be a positive integer" unless n_max.is_a?(Integer) && n_max.positive?

        output = dtype.zeros(n_max, n_max)
        n_max.times do |n_value|
          (n_value + 1).times { |k_value| output[n_value, k_value] = binomial(n_value, k_value) }
        end
        output
      end

      # Recalculates activated opacities and scales for a requested split ratio.
      #
      # @param opacities [Numo::NArray] [N], activated values in [0,1]
      # @param scales [Numo::NArray] [N,3], activated positive scales
      # @param ratios [Numo::NArray] [N], clamped to 1..n_max
      # @param binoms [Numo::NArray] [n_max,n_max]
      # @return [Array<Numo::NArray>] new opacities and scales
      def compute(opacities, scales, ratios, binoms: binomial_table(dtype: opacities.class))
        validate_inputs!(opacities, scales, ratios, binoms)
        output_opacities = opacities.class.zeros(opacities.shape[0])
        output_scales = scales.class.zeros(*scales.shape)
        opacities.shape[0].times do |index|
          ratio = ratios[index].to_i.clamp(1, binoms.shape[0])
          opacity = opacities[index].to_f
          new_opacity = 1.0 - ((1.0 - opacity)**(1.0 / ratio))
          coefficient = opacity / denominator(new_opacity, ratio, binoms)
          output_opacities[index] = new_opacity
          output_scales[index, true] = scales[index, true] * coefficient
        end
        [output_opacities, output_scales]
      end

      def binomial(n_value, k_value)
        return 1 if k_value.zero? || k_value == n_value

        k_value = [k_value, n_value - k_value].min
        (1..k_value).reduce(1) { |value, index| (value * (n_value - k_value + index)) / index }
      end
      private_class_method :binomial

      def denominator(opacity, ratio, binoms)
        (1..ratio).sum do |i_value|
          (0...i_value).sum do |k_value|
            sign = k_value.even? ? 1.0 : -1.0
            binoms[i_value - 1, k_value].to_f * sign * (opacity**(k_value + 1)) / ::Math.sqrt(k_value + 1)
          end
        end
      end
      private_class_method :denominator

      def validate_inputs!(opacities, scales, ratios, binoms)
        unless opacities.is_a?(Numo::NArray) && [Numo::SFloat, Numo::DFloat].include?(opacities.class)
          raise ArgumentError, "opacities must be Numo::SFloat or Numo::DFloat"
        end

        count = opacities.shape[0] if opacities.ndim == 1
        valid = count && scales.shape == [count, 3] && ratios.shape == [count] &&
                binoms.ndim == 2 && binoms.shape[0] == binoms.shape[1]
        return if valid

        raise ShapeError,
              "expected opacities [N], scales [N,3], ratios [N], binoms [M,M]; " \
              "got #{opacities.shape.inspect}, #{scales.shape.inspect}, " \
              "#{ratios.shape.inspect}, #{binoms.shape.inspect}"
      end
      private_class_method :validate_inputs!
    end
  end

  class << self
    # Computes the deterministic 3DGS-MCMC relocation update.
    def relocation(opacities, scales, ratios, binoms: nil)
      options = binoms ? { binoms: binoms } : {}
      Ops::Relocation.compute(opacities, scales, ratios, **options)
    end
  end
end
