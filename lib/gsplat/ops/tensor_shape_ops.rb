# frozen_string_literal: true

module Gsplat
  module Ops
    # Internal helpers for composing coarse differentiable operations.
    module TensorOps
      module_function

      def apply(function, *inputs, **)
        return function.apply(*inputs, **) if inputs.any?(Autograd::Variable)

        needs_grad = Array.new(inputs.length, false)
        function.forward(Autograd::Context.new(needs_grad), *inputs, **)
      end

      def data(value)
        value.is_a?(Autograd::Variable) ? value.data : value
      end
    end

    # Adds a leading camera dimension to a tensor.
    class CameraBroadcast < Autograd::Function
      class << self
        def forward(context, value, camera_count)
          valid_count = camera_count.is_a?(Integer) && camera_count.positive?
          raise ArgumentError, "camera_count must be positive" unless valid_count

          context.save(value.shape)
          output = value.class.zeros(*([camera_count] + value.shape))
          camera_count.times do |camera_index|
            output[*([camera_index] + Array.new(value.ndim, true))] = value
          end
          output
        end

        def backward(_context, gradient)
          [gradient.sum(axis: 0), nil]
        end
      end
    end

    # Computes camera-to-Gaussian viewing directions from world-to-camera matrices.
    class CameraDirections < Autograd::Function
      class << self
        def forward(context, means, viewmats)
          unless means.ndim == 2 && means.shape[-1] == 3 &&
                 viewmats.ndim == 3 && viewmats.shape[1..] == [4, 4]
            raise ShapeError,
                  "expected means [N,3] and viewmats [C,4,4], " \
                  "got #{means.shape.inspect} and #{viewmats.shape.inspect}"
          end

          output = means.class.zeros(viewmats.shape[0], means.shape[0], 3)
          viewmats.shape[0].times do |camera_index|
            rotation = means.class.cast(viewmats[camera_index, 0...3, 0...3])
            translation = means.class.cast(viewmats[camera_index, 0...3, 3])
            camera_center = -translation.dot(rotation)
            output[camera_index, true, true] = means - camera_center
          end
          context.save(means.shape)
          output
        end

        def backward(_context, gradient)
          [gradient.sum(axis: 0), nil]
        end
      end
    end

    # Adds a singleton feature axis to depths.
    class DepthFeatures < Autograd::Function
      class << self
        def forward(_context, depths)
          depths.reshape(*(depths.shape + [1]))
        end

        def backward(_context, gradient)
          [gradient.reshape(*gradient.shape[0...-1])]
        end
      end
    end

    # Concatenates per-Gaussian features and depth.
    class ConcatDepth < Autograd::Function
      class << self
        def forward(context, features, depth_features)
          unless features.shape[0...-1] == depth_features.shape[0...-1] && depth_features.shape[-1] == 1
            raise ShapeError,
                  "expected features [...,D] and depths [...,1], " \
                  "got #{features.shape.inspect} and #{depth_features.shape.inspect}"
          end

          feature_count = features.shape[-1]
          output = features.class.zeros(*(features.shape[0...-1] + [feature_count + 1]))
          output[*Array.new(features.ndim - 1, true), 0...feature_count] = features
          output[*Array.new(features.ndim - 1, true), feature_count] = depth_features[
            *Array.new(depth_features.ndim - 1, true), 0
          ]
          context.save(feature_count)
          output
        end

        def backward(context, gradient)
          feature_count = context.saved_values.fetch(0)
          leading = Array.new(gradient.ndim - 1, true)
          [
            gradient[*leading, 0...feature_count].dup,
            gradient[*leading, feature_count].reshape(*(gradient.shape[0...-1] + [1]))
          ]
        end
      end
    end
  end
end
