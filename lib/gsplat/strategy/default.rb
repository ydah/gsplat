# frozen_string_literal: true

module Gsplat
  module Strategy
    # Original 3DGS densification strategy with gsplat refinements.
    class Default < Base
      DEFAULTS = {
        prune_opa: 0.005,
        grow_grad2d: 0.0002,
        grow_scale3d: 0.01,
        grow_scale2d: 0.05,
        prune_scale3d: 0.1,
        prune_scale2d: 0.15,
        refine_scale2d_stop_iter: 0,
        refine_start_iter: 500,
        refine_stop_iter: 15_000,
        reset_every: 3_000,
        refine_every: 100,
        pause_refine_after_reset: 0,
        absgrad: false,
        revised_opacity: false,
        key_for_gradient: :means2d
      }.freeze

      attr_reader(*DEFAULTS.keys)

      def initialize(**options)
        super()
        unknown = options.keys - DEFAULTS.keys
        raise ArgumentError, "unknown options: #{unknown.join(', ')}" unless unknown.empty?

        DEFAULTS.merge(options).each { |name, value| instance_variable_set(:"@#{name}", value) }
        validate_options!
      end

      def initialize_state(scene_scale:)
        super.merge(grad2d: nil, count: nil, radii: nil)
      end

      # rubocop:disable Metrics/ParameterLists
      def step_post_backward(params:, optimizers:, state:, step:, info:, **_options)
        check_sanity(params, optimizers)
        collect_statistics!(params, state, info)
        refine!(params, optimizers, state, step) if should_refine?(step)
        reset_due = step.positive? && (step % reset_every).zero?
        Ops.reset_opacity!(params, optimizers, maximum: 2 * prune_opa) if reset_due
        state
      end
      # rubocop:enable Metrics/ParameterLists

      # rubocop:disable Metrics/AbcSize
      def refinement_masks(params, state)
        count = params.fetch(:means).data.shape[0]
        average = params.fetch(:means).data.class.zeros(count)
        observed = state[:count].gt(0)
        average[observed] = state[:grad2d][observed] / state[:count][observed] if observed.any?
        high = average.gt(grow_grad2d)
        max_scale = Numo::NMath.exp(params.fetch(:scales).data).max(axis: 1)
        small = max_scale.le(grow_scale3d * state.fetch(:scene_scale))
        small &= state[:radii].le(grow_scale2d) if refine_scale2d_stop_iter.positive? && state[:radii]
        [high & small, high & small.eq(0)]
      end
      # rubocop:enable Metrics/AbcSize

      private

      def validate_options!
        valid_opacity = prune_opa.positive? && prune_opa < 0.5
        raise ArgumentError, "prune_opa must be in (0,0.5)" unless valid_opacity

        intervals = [refine_every, reset_every]
        valid_intervals = intervals.all? { |value| value.is_a?(Integer) && value.positive? }
        raise ArgumentError, "refinement intervals must be positive" unless valid_intervals
        return if refine_start_iter <= refine_stop_iter

        raise ArgumentError, "refine_start_iter must not exceed refine_stop_iter"
      end

      # rubocop:disable Metrics/AbcSize
      def collect_statistics!(params, state, info)
        projected = info.fetch(key_for_gradient)
        gradient = if absgrad
                     projected.respond_to?(:absgrad) ? projected.absgrad : info[:means2d_absgrad]
                   else
                     projected.respond_to?(:grad) ? projected.grad : nil
                   end
        return unless gradient

        count = params.fetch(:means).data.shape[0]
        initialize_statistics!(state, count, gradient.class)
        width = info.fetch(:width)
        height = info.fetch(:height)
        normalized = gradient.dup
        normalized[true, true, 0] *= width / 2.0
        normalized[true, true, 1] *= height / 2.0
        norm = (normalized**2).sum(axis: 2)**0.5
        radii = Gsplat::Ops::TensorOps.data(info.fetch(:radii))
        radii = radii.max(axis: radii.ndim - 1) if radii.ndim == 3 && radii.shape[-1] == 2
        visible = radii.gt(0)
        norm[visible.eq(0)] = 0
        state[:grad2d] += norm.sum(axis: 0)
        state[:count] += gradient.class.cast(visible).sum(axis: 0)
        normalized_radii = radii / [width, height].max.to_f
        camera_max = normalized_radii.max(axis: 0)
        larger = camera_max.gt(state[:radii])
        state[:radii][larger] = camera_max[larger] if larger.any?
      end
      # rubocop:enable Metrics/AbcSize

      def initialize_statistics!(state, count, type)
        return if state[:grad2d]&.shape == [count]

        state[:grad2d] = type.zeros(count)
        state[:count] = type.zeros(count)
        state[:radii] = type.zeros(count)
      end

      def should_refine?(step)
        return false unless step.between?(refine_start_iter, refine_stop_iter)
        return false unless (step % refine_every).zero?
        return true if pause_refine_after_reset.zero?

        (step % reset_every) > pause_refine_after_reset
      end

      def refine!(params, optimizers, state, step)
        duplicate_mask, split_mask = refinement_masks(params, state)
        duplicate_count = Ops.duplicate!(params, optimizers, duplicate_mask)
        revise_duplicated_opacity!(params, duplicate_mask, duplicate_count) if revised_opacity
        extended_split = extend_mask(split_mask, duplicate_count)
        Ops.split!(params, optimizers, extended_split)
        prune!(params, optimizers, state, step)
        initialize_statistics!(
          state,
          params.fetch(:means).data.shape[0],
          params.fetch(:means).data.class
        )
      end

      def revise_duplicated_opacity!(params, original_mask, duplicate_count)
        return if duplicate_count.zero?

        variable = params.fetch(:opacities)
        indices = original_mask.where.to_a
        active = sigmoid(variable.data[indices])
        revised = 1 - Numo::NMath.sqrt(1 - active)
        logits = Numo::NMath.log(revised / (1 - revised))
        variable.data[indices] = logits
        variable.data[-duplicate_count..] = logits
      end

      def extend_mask(mask, count)
        return mask if count.zero?

        output = Numo::Bit.zeros(mask.size + count)
        output[0...mask.size] = mask
        output
      end

      def prune!(params, optimizers, state, step)
        opacity = sigmoid(params.fetch(:opacities).data)
        prune = opacity.lt(prune_opa)
        if step > reset_every
          max_scale = Numo::NMath.exp(params.fetch(:scales).data).max(axis: 1)
          prune |= max_scale.gt(prune_scale3d * state.fetch(:scene_scale))
          if refine_scale2d_stop_iter.positive? && step < refine_scale2d_stop_iter &&
             state[:radii]&.shape == prune.shape
            prune |= state[:radii].gt(prune_scale2d)
          end
        end
        Ops.remove!(params, optimizers, prune)
      end

      def sigmoid(values)
        1.0 / (1 + Numo::NMath.exp(-values))
      end
    end
  end
end
