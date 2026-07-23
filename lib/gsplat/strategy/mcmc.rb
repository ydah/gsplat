# frozen_string_literal: true

module Gsplat
  module Strategy
    # 3D Gaussian Splatting as Markov Chain Monte Carlo strategy.
    class MCMC < Base
      # Upstream-compatible relocation and noise defaults.
      DEFAULTS = {
        cap_max: 1_000_000,
        noise_lr: 5e5,
        refine_start_iter: 500,
        refine_stop_iter: 25_000,
        refine_every: 100,
        min_opacity: 0.005,
        verbose: false
      }.freeze

      attr_reader :cap_max, :min_opacity, :noise_lr, :refine_every,
                  :refine_start_iter, :refine_stop_iter, :verbose

      def initialize(**options)
        super()
        unknown = options.keys - DEFAULTS.keys
        raise ArgumentError, "unknown options: #{unknown.join(', ')}" unless unknown.empty?

        DEFAULTS.merge(options).each { |name, value| instance_variable_set(:"@#{name}", value) }
        validate_options!
      end

      # Creates strategy state including the relocation binomial table.
      #
      # @param scene_scale [Numeric]
      # @return [Hash]
      def initialize_state(scene_scale: 1.0)
        super.merge(binoms: Gsplat::Ops::Relocation.binomial_table(n_max: 51))
      end

      # rubocop:disable Metrics/ParameterLists
      def step_post_backward(params:, optimizers:, state:, step:, info:, **options)
        raise ArgumentError, "info must be a Hash" unless info.is_a?(Hash)

        learning_rate = options.fetch(:lr)
        check_sanity(params, optimizers)
        if should_refine?(step)
          relocated = relocate_dead!(params, optimizers, state)
          added = add_new!(params, optimizers, state)
          log_step(step, relocated, added) if verbose
        end
        Ops.inject_position_noise!(params, scaler: learning_rate * noise_lr)
        state
      end
      # rubocop:enable Metrics/ParameterLists

      private

      def validate_options!
        raise ArgumentError, "cap_max must be positive" unless cap_max.is_a?(Integer) && cap_max.positive?
        raise ArgumentError, "noise_lr must be non-negative" unless noise_lr >= 0
        unless refine_every.is_a?(Integer) && refine_every.positive?
          raise ArgumentError, "refine_every must be positive"
        end
        raise ArgumentError, "min_opacity must be in (0,1)" unless min_opacity.positive? && min_opacity < 1
        return if refine_start_iter <= refine_stop_iter

        raise ArgumentError, "refine_start_iter must not exceed refine_stop_iter"
      end

      def should_refine?(step)
        step < refine_stop_iter && step > refine_start_iter && (step % refine_every).zero?
      end

      def relocate_dead!(params, optimizers, state)
        opacity = 1.0 / (1.0 + Numo::NMath.exp(-params.fetch(:opacities).data))
        dead = opacity.le(min_opacity)
        Ops.relocate!(
          params,
          optimizers,
          dead,
          binoms: state.fetch(:binoms),
          min_opacity: min_opacity
        )
      end

      def add_new!(params, optimizers, state)
        current = params.fetch(:means).data.shape[0]
        target = [cap_max, (1.05 * current).to_i].min
        Ops.sample_add!(
          params,
          optimizers,
          [target - current, 0].max,
          binoms: state.fetch(:binoms),
          min_opacity: min_opacity
        )
      end

      def log_step(step, relocated, added)
        Gsplat.logger.info(
          "MCMC step #{step}: relocated #{relocated}, added #{added}"
        )
      end
    end
  end
end
