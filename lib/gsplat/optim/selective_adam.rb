# frozen_string_literal: true

module Gsplat
  module Optim
    # Adam variant that updates only visible first-axis parameter rows.
    class SelectiveAdam < Adam
      # Updates visible rows while leaving hidden parameters and moments intact.
      #
      # @param visibility [Numo::Bit, Numo::NArray, Array<Boolean>] mask shaped [N]
      # @return [SelectiveAdam] self
      def step(visibility)
        mask = validate_visibility(visibility)
        groups.each_value do |group|
          gradient = group.variable.grad
          next unless gradient

          update_group(group, gradient, mask)
        end
        self
      end

      private

      # rubocop:disable Metrics/AbcSize
      def update_group(group, gradient, mask)
        state = state_for(group.name)
        next_step = state.step + 1
        candidate_mean = (beta1 * state.exp_avg) + ((1 - beta1) * gradient)
        candidate_variance = (beta2 * state.exp_avg_sq) + ((1 - beta2) * (gradient**2))
        selection = first_axis_selection(mask, gradient.ndim)
        exp_avg = state.exp_avg.dup
        exp_avg_sq = state.exp_avg_sq.dup
        exp_avg[*selection] = candidate_mean[*selection]
        exp_avg_sq[*selection] = candidate_variance[*selection]
        corrected_mean = exp_avg / (1 - (beta1**next_step))
        corrected_variance = exp_avg_sq / (1 - (beta2**next_step))
        update = group.lr * corrected_mean / ((corrected_variance**0.5) + group.eps)
        group.variable.data[*selection] = group.variable.data[*selection] - update[*selection]
        @states[group.name] = State.new(
          step: next_step,
          exp_avg: exp_avg,
          exp_avg_sq: exp_avg_sq
        )
      end
      # rubocop:enable Metrics/AbcSize

      def validate_visibility(visibility)
        mask = Numo::Bit.cast(visibility)
        expected = groups.values.first.variable.data.shape[0]
        unless mask.ndim == 1 && mask.shape[0] == expected
          raise ShapeError, "expected visibility [#{expected}], got #{mask.shape.inspect}"
        end

        groups.each_value do |group|
          next if group.variable.data.shape[0] == expected

          raise ShapeError, "all SelectiveAdam groups must share first-axis size #{expected}"
        end

        mask
      end

      def first_axis_selection(mask, dimensions)
        [mask] + Array.new(dimensions - 1, true)
      end
    end
  end
end
