# frozen_string_literal: true

module Gsplat
  module Strategy
    # Shared strategy lifecycle and parameter/optimizer validation.
    class Base
      REQUIRED_KEYS = %i[means quats scales opacities sh0 shN].freeze

      def initialize_state(scene_scale:)
        raise ArgumentError, "scene_scale must be positive" unless scene_scale.positive?

        { scene_scale: scene_scale }
      end

      # rubocop:disable Metrics/AbcSize, Naming/PredicateMethod
      def check_sanity(params, optimizers)
        missing_params = REQUIRED_KEYS - params.keys
        missing_optimizers = REQUIRED_KEYS - optimizers.keys
        raise ArgumentError, "missing params: #{missing_params.join(', ')}" unless missing_params.empty?
        raise ArgumentError, "missing optimizers: #{missing_optimizers.join(', ')}" unless missing_optimizers.empty?

        count = nil
        REQUIRED_KEYS.each do |key|
          variable = params.fetch(key)
          raise ArgumentError, "params[:#{key}] must be an Autograd::Variable" unless variable.is_a?(Autograd::Variable)

          count ||= variable.data.shape[0]
          unless variable.data.ndim.positive? && variable.data.shape[0] == count
            raise ShapeError, "all params must share first-axis size #{count}; #{key}=#{variable.data.shape.inspect}"
          end

          optimizer = optimizers.fetch(key)
          unless optimizer.is_a?(Optim::Adam) &&
                 optimizer.groups.values.any? { |group| group.variable.equal?(variable) }
            raise ArgumentError, "optimizers[:#{key}] must optimize params[:#{key}]"
          end
        end
        true
      end
      # rubocop:enable Metrics/AbcSize, Naming/PredicateMethod

      def step_pre_backward(params:, optimizers:, state:, step:, info:)
        [params, optimizers, state, step, info]
        nil
      end
    end
  end
end
