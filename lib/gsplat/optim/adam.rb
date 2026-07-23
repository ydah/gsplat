# frozen_string_literal: true

module Gsplat
  module Optim
    # Dense Adam optimizer with editable first-axis state for densification.
    class Adam
      Group = Data.define(:name, :variable, :lr, :eps)
      State = Data.define(:step, :exp_avg, :exp_avg_sq)

      attr_reader :beta1, :beta2, :groups

      # rubocop:disable Naming/MethodParameterName
      def initialize(groups = nil, lr: 1e-3, betas: [0.9, 0.999], eps: 1e-15, **named_groups)
        # rubocop:enable Naming/MethodParameterName
        raise ArgumentError, "provide positional or named groups, not both" if groups && !named_groups.empty?

        groups ||= named_groups
        @beta1, @beta2 = betas
        validate_hyperparameters!(lr, eps)
        @groups = normalize_groups(groups, lr, eps)
        @states = {}
      end

      # rubocop:disable Metrics/AbcSize
      def step
        groups.each_value do |group|
          gradient = group.variable.grad
          next unless gradient

          state = state_for(group.name)
          next_step = state.step + 1
          exp_avg = (beta1 * state.exp_avg) + ((1 - beta1) * gradient)
          exp_avg_sq = (beta2 * state.exp_avg_sq) + ((1 - beta2) * (gradient**2))
          corrected_mean = exp_avg / (1 - (beta1**next_step))
          corrected_variance = exp_avg_sq / (1 - (beta2**next_step))
          update = group.lr * corrected_mean / ((corrected_variance**0.5) + group.eps)
          group.variable.data[] = group.variable.data - update
          @states[group.name] = State.new(
            step: next_step,
            exp_avg: exp_avg,
            exp_avg_sq: exp_avg_sq
          )
        end
        self
      end
      # rubocop:enable Metrics/AbcSize

      def zero_grad!
        groups.each_value { |group| group.variable.zero_grad! }
        self
      end

      def state(name = nil)
        return state_for(name.to_sym) if name
        return state_for(groups.keys.first) if groups.length == 1

        groups.keys.to_h { |key| [key, state_for(key)] }
      end

      def learning_rate(name = nil)
        return groups.fetch(name.to_sym).lr if name
        return groups.values.first.lr if groups.length == 1

        groups.transform_values(&:lr)
      end

      def learning_rate=(value)
        raise ArgumentError, "learning rate must be positive" unless value.positive?

        @groups = groups.transform_values do |group|
          Group.new(name: group.name, variable: group.variable, lr: value, eps: group.eps)
        end
      end

      def set_learning_rate(name, value)
        raise ArgumentError, "learning rate must be positive" unless value.positive?

        key = name.to_sym
        group = groups.fetch(key)
        @groups[key] = Group.new(name: key, variable: group.variable, lr: value, eps: group.eps)
      end

      def select!(selection)
        groups.each_key do |name|
          current = state_for(name)
          @states[name] = State.new(
            step: current.step,
            exp_avg: select_first_axis(current.exp_avg, selection),
            exp_avg_sq: select_first_axis(current.exp_avg_sq, selection)
          )
        end
        self
      end

      def append!(count)
        raise ArgumentError, "append count must be non-negative" unless count.is_a?(Integer) && !count.negative?
        return self if count.zero?

        groups.each_key do |name|
          current = state_for(name)
          @states[name] = State.new(
            step: current.step,
            exp_avg: append_zeros(current.exp_avg, count),
            exp_avg_sq: append_zeros(current.exp_avg_sq, count)
          )
        end
        self
      end

      def zero_state_at!(indices)
        groups.each_key do |name|
          current = state_for(name)
          index = [indices] + Array.new(current.exp_avg.ndim - 1, true)
          current.exp_avg[*index] = 0
          current.exp_avg_sq[*index] = 0
        end
        self
      end

      private

      def validate_hyperparameters!(learning_rate, epsilon)
        valid_betas = [beta1, beta2].all? { |value| value >= 0 && value < 1 }
        raise ArgumentError, "betas must be in [0,1)" unless valid_betas
        raise ArgumentError, "learning rate must be positive" unless learning_rate.positive?
        raise ArgumentError, "eps must be positive" unless epsilon.positive?
      end

      def normalize_groups(input, default_lr, default_eps)
        raw_groups = input.is_a?(Autograd::Variable) ? { default: { variable: input } } : input
        unless raw_groups.is_a?(Hash) && !raw_groups.empty?
          raise ArgumentError, "groups must be a Variable or non-empty Hash"
        end

        raw_groups.to_h do |name, specification|
          specification = { variable: specification } if specification.is_a?(Autograd::Variable)
          variable = specification.fetch(:variable)
          raise ArgumentError, "group variable must be an Autograd::Variable" unless variable.is_a?(Autograd::Variable)

          group_lr = specification.fetch(:lr, default_lr)
          group_eps = specification.fetch(:eps, default_eps)
          validate_hyperparameters!(group_lr, group_eps)
          key = name.to_sym
          [key, Group.new(name: key, variable: variable, lr: group_lr, eps: group_eps)]
        end
      end

      def state_for(name)
        group = groups.fetch(name)
        @states[name] ||= State.new(
          step: 0,
          exp_avg: group.variable.data.class.zeros(*group.variable.data.shape),
          exp_avg_sq: group.variable.data.class.zeros(*group.variable.data.shape)
        )
      end

      def select_first_axis(array, selection)
        array[*([selection] + Array.new(array.ndim - 1, true))].dup
      end

      def append_zeros(array, count)
        output = array.class.zeros(*([array.shape[0] + count] + array.shape[1..]))
        output[*([0...array.shape[0]] + Array.new(array.ndim - 1, true))] = array
        output
      end
    end
  end
end
