# frozen_string_literal: true

module Gsplat
  module Backend
    # Analytic Numo VJP for dense pinhole projection.
    module RubyProjectionBackward
      module_function

      # rubocop:disable Metrics/ParameterLists
      def backward(means, covars, quaternions, scales, viewmats, intrinsics, width, height,
                   grad_means2d, grad_depths, grad_conics, grad_compensations, **options)
        # rubocop:enable Metrics/ParameterLists
        prepared = RubyProjection.prepare_inputs(
          means, covars, quaternions, scales, viewmats, intrinsics, width, height,
          options.fetch(:eps2d), options.fetch(:near_plane), options.fetch(:far_plane),
          options.fetch(:radius_clip), options.fetch(:camera_model)
        )
        means, expanded_covars, viewmats, intrinsics = prepared
        camera_means, camera_covars = Math::CameraProjection.world_to_cam(means, expanded_covars, viewmats)
        projection_means = safe_means(
          camera_means,
          options.fetch(:near_plane),
          options.fetch(:far_plane)
        )
        means2d, covars2d = Math::CameraProjection.persp_proj(
          projection_means,
          camera_covars,
          intrinsics,
          width,
          height
        )
        jacobians = projection_jacobians(projection_means, intrinsics, width, height)
        output_gradients = validate_output_gradients(
          means.class,
          means2d,
          grad_means2d,
          grad_depths,
          grad_conics,
          grad_compensations
        )
        grad_camera_means, grad_camera_covars = projection_vjp(
          projection_means,
          camera_covars,
          covars2d,
          jacobians,
          intrinsics,
          output_gradients,
          options.fetch(:eps2d),
          width,
          height
        )
        grad_means, grad_expanded_covars = world_to_cam_vjp(
          grad_camera_means,
          grad_camera_covars,
          viewmats
        )
        input_covariance_gradients(
          means,
          covars,
          quaternions,
          scales,
          grad_means,
          grad_expanded_covars
        )
      end

      def safe_means(camera_means, near_plane, far_plane)
        output = camera_means.dup
        depths = output[true, true, 2]
        culled = depths.le(near_plane) | depths.ge(far_plane)
        depths[culled] = 1 if culled.any?
        output[true, true, 2] = depths
        output
      end
      private_class_method :safe_means

      def projection_jacobians(means, intrinsics, width, height)
        output = means.class.zeros(means.shape[0], means.shape[1], 2, 3)
        means.shape[0].times do |camera_index|
          _, jacobians = Math::CameraProjection.pinhole_camera(
            means[camera_index, true, true],
            intrinsics[camera_index, true, true],
            width,
            height
          )
          output[camera_index, true, true, true] = jacobians
        end
        output
      end
      private_class_method :projection_jacobians

      # rubocop:disable Metrics/ParameterLists
      def validate_output_gradients(type, means2d, grad_means2d, grad_depths, grad_conics, grad_compensations)
        # rubocop:enable Metrics/ParameterLists
        leading_shape = means2d.shape[0...-1]
        [
          cast_gradient(type, grad_means2d, means2d.shape),
          cast_gradient(type, grad_depths, leading_shape),
          cast_gradient(type, grad_conics, leading_shape + [3]),
          grad_compensations && cast_gradient(type, grad_compensations, leading_shape)
        ]
      end
      private_class_method :validate_output_gradients

      def cast_gradient(type, gradient, expected_shape)
        value = type.cast(gradient)
        unless value.shape == expected_shape
          raise ShapeError, "expected output gradient #{expected_shape.inspect}, got #{value.shape.inspect}"
        end

        value
      end
      private_class_method :cast_gradient

      # rubocop:disable Metrics/ParameterLists
      def projection_vjp(means, covars, covars2d, jacobians, intrinsics, gradients, eps2d, width, height)
        # rubocop:enable Metrics/ParameterLists
        grad_means2d, grad_depths, grad_conics, grad_compensations = gradients
        grad_covars2d = covariance_output_vjp(
          covars2d,
          grad_conics,
          grad_compensations,
          eps2d
        )
        count = means.shape[0] * means.shape[1]
        jacobian_flat = jacobians.reshape(count, 2, 3)
        covariance_flat = covars.reshape(count, 3, 3)
        gradient_flat = grad_covars2d.reshape(count, 2, 2)
        grad_camera_covars = Math::Mat.matmul_batch(
          Math::Mat.matmul_batch(jacobian_flat.transpose(0, 2, 1), gradient_flat),
          jacobian_flat
        ).reshape(*covars.shape)
        grad_jacobians = jacobian_vjp(jacobian_flat, covariance_flat, gradient_flat)
        grad_camera_means = camera_mean_vjp(
          means,
          intrinsics,
          grad_means2d,
          grad_depths,
          grad_jacobians.reshape(*jacobians.shape),
          width,
          height
        )
        [grad_camera_means, grad_camera_covars]
      end
      private_class_method :projection_vjp

      def jacobian_vjp(jacobians, covars, gradient)
        first = Math::Mat.matmul_batch(
          Math::Mat.matmul_batch(gradient, jacobians),
          covars.transpose(0, 2, 1)
        )
        second = Math::Mat.matmul_batch(
          Math::Mat.matmul_batch(gradient.transpose(0, 2, 1), jacobians),
          covars
        )
        first + second
      end
      private_class_method :jacobian_vjp
    end
  end
end
