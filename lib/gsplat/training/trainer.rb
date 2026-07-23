# frozen_string_literal: true

require "fileutils"

module Gsplat
  module Training
    # End-to-end multi-view Gaussian training loop.
    class Trainer
      # Final step, before/after metrics, and periodic history.
      Result = Data.define(:step, :initial_metrics, :final_metrics, :history)
      # Mapping from parameter names to {Config} learning-rate attributes.
      LEARNING_RATES = {
        means: :means_lr,
        scales: :scales_lr,
        quats: :quats_lr,
        opacities: :opacities_lr,
        sh0: :sh0_lr,
        shN: :shN_lr
      }.freeze

      attr_reader :config, :optimizers, :params, :scene, :step, :strategy, :strategy_state

      def initialize(scene:, config: Config.new, params: nil, strategy: nil)
        @scene = scene
        @config = config
        @rng = Random.new(config.seed)
        @params = params || Utils.init_from_points(
          scene.points,
          scene.colors,
          sh_degree: config.sh_degree,
          init_opacity: config.init_opacity,
          init_scale: config.init_scale,
          rng: @rng
        )
        @optimizers = build_optimizers
        @strategy = strategy || Strategy::Default.new
        @strategy.check_sanity(@params, @optimizers)
        @strategy_state = @strategy.initialize_state(scene_scale: scene.scene_scale)
        @scheduler = Optim::ExponentialLR.new(
          @optimizers.fetch(:means),
          lr_final: config.means_lr_final * scene.scene_scale,
          max_steps: config.max_steps
        )
        @step = 0
      end

      # Optimizes until the configured maximum or for an additional step count.
      #
      # @param steps [Integer, nil]
      # @return [Result]
      def train(steps: nil)
        target_step = steps ? step + steps : config.max_steps
        initial_metrics = evaluate
        history = []
        while step < target_step
          @step += 1
          loss = training_step
          history << { step: step, loss: loss } if log_due?
          evaluate_and_log!(history) if config.eval_steps.include?(step)
          save_checkpoint! if config.save_steps.include?(step)
        end
        Result.new(
          step: step,
          initial_metrics: initial_metrics,
          final_metrics: evaluate,
          history: history.freeze
        )
      end

      # Renders every view without graph recording and reports PSNR/SSIM.
      #
      # @return [Hash{Symbol=>Float}]
      def evaluate
        Autograd.no_grad do
          rendered, = render((0...scene.camera_count).to_a, sh_degree_for(step))
          values = Ops::TensorOps.data(rendered)
          {
            psnr: Losses.psnr(values, scene.images),
            ssim: Losses.ssim(values, scene.images, layout: :nhwc)
          }
        end
      end

      # Saves current parameters, optimizer moments, step, and config.
      #
      # @param path [String, nil] generated from `config.output_dir` when nil
      # @return [String] written path
      def save_checkpoint!(path = nil)
        path ||= File.join(config.output_dir, "checkpoints", format("step_%06d.npz", step))
        FileUtils.mkdir_p(File.dirname(path))
        IO::Checkpoint.save(
          path,
          params: params,
          optimizers: optimizers,
          step: step,
          config: config.to_h
        )
        path
      end

      # Restores parameters, optimizer moments, step, and scheduler position.
      #
      # @param source [String, #read]
      # @return [IO::Checkpoint::Snapshot]
      def load_checkpoint!(source)
        snapshot = IO::Checkpoint.restore!(source, params: params, optimizers: optimizers)
        @step = snapshot.step
        @scheduler.step(step)
        snapshot
      end

      private

      # rubocop:disable Metrics/AbcSize
      def training_step
        indices = batch_indices
        rendered, _alphas, info = render(indices, sh_degree_for(step))
        target = scene.images[indices, true, true, true].dup
        strategy.step_pre_backward(
          params: params, optimizers: optimizers, state: strategy_state, step: step, info: info
        )
        activated = activated_params
        loss = Losses.regularized_reconstruction(
          rendered,
          target,
          activated.fetch(:opacities),
          activated.fetch(:scales),
          ssim_lambda: config.ssim_lambda,
          opacity_reg: config.opacity_reg,
          scale_reg: config.scale_reg,
          layout: :nhwc
        )
        loss.backward
        strategy.step_post_backward(
          params: params,
          optimizers: optimizers,
          state: strategy_state,
          step: step,
          info: info,
          lr: optimizers.fetch(:means).learning_rate
        )
        optimizers.each_value(&:step)
        optimizers.each_value(&:zero_grad!)
        @scheduler.step(step)
        loss.data.to_f
      end
      # rubocop:enable Metrics/AbcSize

      # rubocop:disable Metrics/AbcSize
      def render(indices, degree)
        activated = activated_params
        backgrounds = random_background(indices.length, scene.images.class)
        renderer = %i[two_d 2dgs].include?(config.model_type.to_sym) ? :rasterization_2dgs : :rasterization
        outputs = Gsplat.public_send(
          renderer,
          means: params.fetch(:means),
          quats: activated.fetch(:quats),
          scales: activated.fetch(:scales),
          opacities: activated.fetch(:opacities),
          colors: activated.fetch(:colors),
          viewmats: scene.viewmats[indices, true, true].dup,
          ks: scene.intrinsics[indices, true, true].dup,
          width: scene.width,
          height: scene.height,
          sh_degree: degree,
          backgrounds: backgrounds,
          near_plane: config.near_plane,
          far_plane: config.far_plane,
          tile_size: config.tile_size,
          rasterize_mode: config.rasterize_mode,
          absgrad: strategy.respond_to?(:absgrad) && strategy.absgrad
        )
        return outputs if renderer == :rasterization

        [outputs[0], outputs[1], outputs[6]]
      end
      # rubocop:enable Metrics/AbcSize

      def activated_params
        sh0 = params.fetch(:sh0)
        shn = params.fetch(:shN)
        colors = if shn.data.shape[1].zero?
                   sh0
                 else
                   Ops::TensorOps.apply(Ops::ConcatCoefficients, sh0, shn)
                 end
        {
          quats: Ops::TensorOps.apply(Ops::NormalizeQuaternion, params.fetch(:quats)),
          scales: Ops::TensorOps.apply(Ops::Exp, params.fetch(:scales)),
          opacities: Ops::TensorOps.apply(Ops::Sigmoid, params.fetch(:opacities)),
          colors: colors
        }
      end

      def build_optimizers
        LEARNING_RATES.to_h do |name, config_name|
          rate = config.public_send(config_name)
          rate *= scene.scene_scale if name == :means
          [name, Optim::Adam.new(params.fetch(name), lr: rate)]
        end
      end

      def batch_indices
        Array.new(config.batch_size) { @rng.rand(scene.camera_count) }
      end

      def sh_degree_for(current_step)
        [config.sh_degree, current_step / config.sh_degree_interval].min
      end

      def random_background(count, type)
        return nil unless config.random_background

        type.cast(Array.new(count * 3) { @rng.rand }).reshape(count, 3)
      end

      def log_due?
        (step % config.log_every).zero?
      end

      def evaluate_and_log!(history)
        metrics = evaluate
        history << { step: step, **metrics }
        Gsplat.logger.info(
          format(
            "step=%<step>d psnr=%<psnr>.3f ssim=%<ssim>.5f gaussians=%<gaussians>d",
            step: step,
            psnr: metrics[:psnr],
            ssim: metrics[:ssim],
            gaussians: params.fetch(:means).data.shape[0]
          )
        )
      end
    end
  end
end
