# frozen_string_literal: true

module Gsplat
  module Training
    # A small self-consistency image fitting loop used by examples and E2E tests.
    class ImageFitter
      # Image-fit metrics, optimized colors, and target image.
      Result = Data.define(:initial_psnr, :final_psnr, :history, :colors, :target)

      # Scalar image MSE used to drive the example's reverse pass.
      class MeanSquaredError < Autograd::Function
        class << self
          # Evaluates scalar mean squared error.
          # @api private
          def forward(context, rendered, target)
            difference = rendered - target
            context.save(difference, rendered.size)
            rendered.class.cast((difference**2).sum / rendered.size)
          end

          # Returns the rendered-image VJP; targets are constants.
          # @api private
          def backward(context, grad_output)
            difference, count = context.saved_values
            [(2.0 * difference / count) * grad_output.to_f, nil]
          end
        end
      end

      attr_reader :height, :learning_rate, :n_gaussians, :width

      def initialize(width: 64, height: 64, n_gaussians: 2_000, learning_rate: 20.0, seed: 42)
        @width = width
        @height = height
        @n_gaussians = n_gaussians
        @learning_rate = learning_rate
        @rng = Random.new(seed)
        validate_options!
        build_scene
      end

      def fit(steps: 300)
        raise ArgumentError, "steps must be a positive integer" unless steps.is_a?(Integer) && steps.positive?

        target = render(@teacher_colors)
        colors = initial_colors
        history = [psnr(render(colors), target)]
        steps.times do
          variable = Autograd::Variable.new(colors, requires_grad: true)
          rendered = render(variable)
          MeanSquaredError.apply(rendered, target).backward
          colors = clipped(colors - (learning_rate * variable.grad), 0.0, 1.0)
          history << psnr(render(colors), target)
        end
        Result.new(
          initial_psnr: history.first,
          final_psnr: history.last,
          history: history.freeze,
          colors: colors,
          target: target
        )
      end

      private

      def validate_options!
        values = { width: width, height: height, n_gaussians: n_gaussians }
        invalid = values.find { |_name, value| !value.is_a?(Integer) || !value.positive? }
        raise ArgumentError, "#{invalid[0]} must be a positive integer" if invalid
        raise ArgumentError, "learning_rate must be positive" unless learning_rate.positive?
      end

      # rubocop:disable Metrics/AbcSize
      def build_scene
        @means = Numo::DFloat.zeros(n_gaussians, 3)
        @quaternions = Numo::DFloat.zeros(n_gaussians, 4)
        @quaternions[true, 0] = 1
        @scales = Numo::DFloat.zeros(n_gaussians, 3)
        @opacities = Numo::DFloat.ones(n_gaussians) * 0.85
        @viewmats = Numo::DFloat.zeros(1, 4, 4)
        @viewmats[0, true, true] = Numo::DFloat.eye(4)
        focal = [width, height].max.to_f
        @intrinsics = Numo::DFloat[[[focal, 0, width / 2.0], [0, focal, height / 2.0], [0, 0, 1]]]
        side = ::Math.sqrt(n_gaussians).ceil
        n_gaussians.times do |index|
          grid_x = index % side
          grid_y = index / side
          pixel_x = ((grid_x + 0.5) * width / side) - 0.5
          pixel_y = ((grid_y + 0.5) * height / side) - 0.5
          depth = 2.0 + (0.2 * (index % 3))
          @means[index, true] = [
            (pixel_x - (width / 2.0)) * depth / focal,
            (pixel_y - (height / 2.0)) * depth / focal,
            depth
          ]
          world_scale = 0.8 * depth / focal
          @scales[index, true] = [world_scale, world_scale, world_scale]
        end
        @teacher_colors = teacher_colors(side)
      end
      # rubocop:enable Metrics/AbcSize

      def teacher_colors(side)
        output = Numo::DFloat.zeros(n_gaussians, 3)
        n_gaussians.times do |index|
          x_coord = (index % side).to_f / [side - 1, 1].max
          y_coord = (index / side).to_f / [side - 1, 1].max
          output[index, true] = [x_coord, y_coord, 0.25 + (0.5 * x_coord * y_coord)]
        end
        output
      end

      def initial_colors
        values = Array.new(n_gaussians * 3) { 0.35 + (@rng.rand * 0.3) }
        Numo::DFloat.cast(values).reshape(n_gaussians, 3)
      end

      def render(colors)
        rendered, = Gsplat.rasterization(
          means: @means,
          quats: @quaternions,
          scales: @scales,
          opacities: @opacities,
          colors: colors,
          viewmats: @viewmats,
          ks: @intrinsics,
          width: width,
          height: height,
          tile_size: 16
        )
        rendered
      end

      def psnr(rendered, target)
        mean_squared_error = (((Ops::TensorOps.data(rendered) - target)**2).sum / target.size).to_f
        return Float::INFINITY if mean_squared_error.zero?

        -10.0 * ::Math.log10(mean_squared_error)
      end

      def clipped(values, minimum, maximum)
        output = values.dup
        below = output.lt(minimum)
        above = output.gt(maximum)
        output[below] = minimum if below.any?
        output[above] = maximum if above.any?
        output
      end
    end
  end
end
