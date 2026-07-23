# frozen_string_literal: true

module Gsplat
  module Optim
    # Closed-form exponential learning-rate schedule.
    class ExponentialLR
      attr_reader :max_steps, :step_count

      def initialize(optimizer, lr_final:, max_steps:)
        raise ArgumentError, "lr_final must be positive" unless lr_final.positive?
        raise ArgumentError, "max_steps must be positive" unless max_steps.is_a?(Integer) && max_steps.positive?

        @optimizer = optimizer
        @lr_final = lr_final
        @max_steps = max_steps
        @step_count = 0
        @initial_rates = optimizer.groups.transform_values(&:lr)
      end

      def step(step = nil)
        @step_count = step || (step_count + 1)
        progress = [step_count.to_f / max_steps, 1.0].min
        @optimizer.groups.each_key do |name|
          initial = @initial_rates.fetch(name)
          rate = initial * ((@lr_final / initial)**progress)
          @optimizer.set_learning_rate(name, rate)
        end
        @optimizer.learning_rate
      end
    end
  end
end
