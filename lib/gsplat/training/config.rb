# frozen_string_literal: true

module Gsplat
  module Training
    # Trainer hyperparameters matching the upstream simple trainer defaults.
    class Config
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
        output_dir: "results",
        seed: 42
      }.freeze

      attr_reader(*DEFAULTS.keys)

      def initialize(**options)
        unknown = options.keys - DEFAULTS.keys
        raise ArgumentError, "unknown trainer options: #{unknown.join(', ')}" unless unknown.empty?

        DEFAULTS.merge(options).each do |name, value|
          value = value.dup if value.is_a?(Array)
          instance_variable_set(:"@#{name}", value)
        end
        validate!
      end

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
