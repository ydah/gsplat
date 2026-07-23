# frozen_string_literal: true

module Gsplat
  module Training
    # Trainer hyperparameters matching the upstream simple trainer defaults.
    class Config
      # Default training, optimizer, evaluation, and output options.
      DEFAULTS = {
        max_steps: 30_000,
        batch_size: 1,
        sh_degree: 3,
        sh_degree_interval: 1_000,
        ssim_lambda: 0.2,
        init_opacity: 0.1,
        init_scale: 1.0,
        means_lr: 1.6e-4,
        means_lr_final: 1.6e-6,
        scales_lr: 5e-3,
        quats_lr: 1e-3,
        opacities_lr: 5e-2,
        sh0_lr: 2.5e-3,
        shN_lr: 1.25e-4,
        opacity_reg: 0.0,
        scale_reg: 0.0,
        eval_steps: [7_000, 30_000],
        save_steps: [7_000, 30_000],
        log_every: 100,
        random_background: false,
        near_plane: 0.01,
        far_plane: 1e10,
        tile_size: 16,
        rasterize_mode: "classic",
        model_type: :three_d,
        output_dir: "results",
        seed: 42
      }.freeze

      # rubocop:disable Naming/MethodName
      attr_reader :batch_size, :eval_steps, :far_plane, :init_opacity,
                  :init_scale, :log_every, :max_steps, :means_lr,
                  :means_lr_final, :model_type, :near_plane, :opacities_lr,
                  :opacity_reg, :output_dir, :quats_lr, :random_background,
                  :rasterize_mode, :save_steps, :scales_lr, :scale_reg, :seed,
                  :sh0_lr, :shN_lr, :sh_degree, :sh_degree_interval,
                  :ssim_lambda, :tile_size
      # rubocop:enable Naming/MethodName

      def initialize(**options)
        unknown = options.keys - DEFAULTS.keys
        raise ArgumentError, "unknown trainer options: #{unknown.join(', ')}" unless unknown.empty?

        DEFAULTS.merge(options).each do |name, value|
          value = value.dup if value.is_a?(Array)
          instance_variable_set(:"@#{name}", value)
        end
        validate!
      end

      # Returns an independent hash suitable for checkpoint metadata.
      #
      # @return [Hash{Symbol=>Object}]
      def to_h
        DEFAULTS.keys.to_h { |name| [name, public_send(name)] }
      end

      private

      def validate!
        positive_integers = %i[max_steps batch_size sh_degree_interval log_every tile_size]
        invalid = positive_integers.find do |name|
          value = public_send(name)
          !value.is_a?(Integer) || !value.positive?
        end
        raise ArgumentError, "#{invalid} must be a positive integer" if invalid
        raise ArgumentError, "sh_degree must be in 0..4" unless sh_degree.is_a?(Integer) && sh_degree.between?(0, 4)
        raise ArgumentError, "ssim_lambda must be between 0 and 1" unless ssim_lambda.between?(0.0, 1.0)
        unless %i[three_d two_d 3dgs 2dgs].include?(model_type.to_sym)
          raise ArgumentError, "model_type must be :three_d/:3dgs or :two_d/:2dgs"
        end

        learning_rates = %i[means_lr means_lr_final scales_lr quats_lr opacities_lr sh0_lr shN_lr]
        invalid_rate = learning_rates.find { |name| !public_send(name).positive? }
        raise ArgumentError, "#{invalid_rate} must be positive" if invalid_rate

        regularizers = %i[opacity_reg scale_reg]
        invalid_regularizer = regularizers.find { |name| public_send(name).negative? }
        raise ArgumentError, "#{invalid_regularizer} must be non-negative" if invalid_regularizer
      end
    end
  end
end
