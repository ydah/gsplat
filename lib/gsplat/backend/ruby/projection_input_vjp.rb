# frozen_string_literal: true

module Gsplat
  module Backend
    # Input-side VJP helpers for RubyProjectionBackward.
    module RubyProjectionBackward
      module_function

      # rubocop:disable Metrics/ParameterLists
      def camera_mean_vjp(means, intrinsics, grad_means2d, grad_depths, grad_jacobians, width, height,
                          options)
        # rubocop:enable Metrics/ParameterLists
        output = means.class.zeros(*means.shape)
        # rubocop:disable Metrics/BlockLength
        means.shape[0].times do |camera_index|
          camera_means = means[camera_index, true, true]
          camera_intrinsics = intrinsics[camera_index, true, true]
          output_gradient = grad_means2d[camera_index, true, true]
          camera_model = options.fetch(:camera_model)
          gradient = if camera_model.to_s == "ortho"
                       ortho_mean_vjp(camera_means, camera_intrinsics, output_gradient)
                     elsif extended_camera?(options)
                       distorted_mean_vjp(
                         camera_means,
                         camera_intrinsics,
                         output_gradient,
                         grad_jacobians[camera_index, true, true, true],
                         options,
                         camera_index
                       )
                     else
                       projected_mean_vjp(camera_means, camera_intrinsics, output_gradient) +
                         pinhole_jacobian_vjp(
                           camera_means,
                           camera_intrinsics,
                           grad_jacobians[camera_index, true, true, true],
                           width,
                           height
                         )
                     end
          gradient[true, 2] += grad_depths[camera_index, true]
          output[camera_index, true, true] = gradient
        end
        # rubocop:enable Metrics/BlockLength
        output
      end
      private_class_method :camera_mean_vjp

      def extended_camera?(options)
        options.fetch(:camera_model).to_s == "fisheye" ||
          %i[radial_coeffs tangential_coeffs thin_prism_coeffs].any? { |key| options[key] }
      end
      private_class_method :extended_camera?

      # rubocop:disable Metrics/AbcSize, Metrics/ParameterLists
      def distorted_mean_vjp(means, intrinsics, grad_projected, grad_jacobians, options, camera_index)
        # rubocop:enable Metrics/AbcSize, Metrics/ParameterLists
        output = means.class.zeros(*means.shape)
        epsilon = means.instance_of?(Numo::DFloat) ? 1e-4 : 2e-3
        means.shape[0].times do |gaussian_index|
          point = means[gaussian_index, true].to_a
          3.times do |axis|
            positive = point.dup
            negative = point.dup
            positive[axis] += epsilon
            negative[axis] -= epsilon
            positive_outputs = distorted_point_outputs(positive, intrinsics, options, camera_index, means.class)
            negative_outputs = distorted_point_outputs(negative, intrinsics, options, camera_index, means.class)
            projected_derivative = (positive_outputs[0] - negative_outputs[0]) / (2 * epsilon)
            jacobian_derivative = (positive_outputs[1] - negative_outputs[1]) / (2 * epsilon)
            output[gaussian_index, axis] =
              (projected_derivative * grad_projected[gaussian_index, true]).sum +
              (jacobian_derivative * grad_jacobians[gaussian_index, true, true]).sum
          end
        end
        output
      end
      private_class_method :distorted_mean_vjp

      def distorted_point_outputs(point, intrinsics, options, camera_index, type)
        projected, jacobians = Math::CameraDistortion.project_camera(
          type.cast([point]),
          intrinsics,
          options.fetch(:camera_model),
          radial: coefficient_row(options[:radial_coeffs], camera_index),
          tangential: coefficient_row(options[:tangential_coeffs], camera_index),
          thin_prism: coefficient_row(options[:thin_prism_coeffs], camera_index)
        )
        [projected.flatten, jacobians.reshape(2, 3)]
      end
      private_class_method :distorted_point_outputs

      # rubocop:disable Metrics/AbcSize
      def projected_mean_vjp(means, intrinsics, gradient)
        x_coord = means[true, 0]
        y_coord = means[true, 1]
        z_coord = means[true, 2]
        homogeneous_x = (intrinsics[0, 0] * x_coord) + (intrinsics[0, 1] * y_coord) +
                        (intrinsics[0, 2] * z_coord)
        homogeneous_y = (intrinsics[1, 0] * x_coord) + (intrinsics[1, 1] * y_coord) +
                        (intrinsics[1, 2] * z_coord)
        grad_x = gradient[true, 0]
        grad_y = gradient[true, 1]
        output = means.class.zeros(*means.shape)
        output[true, 0] = (
          (grad_x * intrinsics[0, 0]) + (grad_y * intrinsics[1, 0])
        ) / z_coord
        output[true, 1] = (
          (grad_x * intrinsics[0, 1]) + (grad_y * intrinsics[1, 1])
        ) / z_coord
        output[true, 2] = (
          (grad_x * ((intrinsics[0, 2] * z_coord) - homogeneous_x)) +
          (grad_y * ((intrinsics[1, 2] * z_coord) - homogeneous_y))
        ) / (z_coord**2)
        output
      end
      # rubocop:enable Metrics/AbcSize
      private_class_method :projected_mean_vjp

      def ortho_mean_vjp(means, intrinsics, gradient)
        output = means.class.zeros(*means.shape)
        output[true, 0] = gradient[true, 0] * intrinsics[0, 0]
        output[true, 1] = gradient[true, 1] * intrinsics[1, 1]
        output
      end
      private_class_method :ortho_mean_vjp

      # rubocop:disable Metrics/AbcSize
      def pinhole_jacobian_vjp(means, intrinsics, gradient, width, height)
        x_coord = means[true, 0]
        y_coord = means[true, 1]
        z_coord = means[true, 2]
        focal_x = intrinsics[0, 0]
        focal_y = intrinsics[1, 1]
        ratio_x = x_coord / z_coord
        ratio_y = y_coord / z_coord
        min_x = -((intrinsics[0, 2] / focal_x) + (0.15 * width / focal_x))
        max_x = ((width - intrinsics[0, 2]) / focal_x) + (0.15 * width / focal_x)
        min_y = -((intrinsics[1, 2] / focal_y) + (0.15 * height / focal_y))
        max_y = ((height - intrinsics[1, 2]) / focal_y) + (0.15 * height / focal_y)
        clipped_x, inside_x = clipped_ratio_and_mask(ratio_x, min_x, max_x)
        clipped_y, inside_y = clipped_ratio_and_mask(ratio_y, min_y, max_y)
        inside_x = means.class.cast(inside_x)
        inside_y = means.class.cast(inside_y)
        output = means.class.zeros(*means.shape)
        output[true, 0] = gradient[true, 0, 2] * (-focal_x / (z_coord**2)) * inside_x
        output[true, 1] = gradient[true, 1, 2] * (-focal_y / (z_coord**2)) * inside_y
        derivative_x_z = (
          (inside_x * 2 * focal_x * x_coord / (z_coord**3)) +
          ((1 - inside_x) * focal_x * clipped_x / (z_coord**2))
        )
        derivative_y_z = (
          (inside_y * 2 * focal_y * y_coord / (z_coord**3)) +
          ((1 - inside_y) * focal_y * clipped_y / (z_coord**2))
        )
        output[true, 2] = (
          (gradient[true, 0, 0] * (-focal_x / (z_coord**2))) +
          (gradient[true, 1, 1] * (-focal_y / (z_coord**2))) +
          (gradient[true, 0, 2] * derivative_x_z) +
          (gradient[true, 1, 2] * derivative_y_z)
        )
        output
      end
      # rubocop:enable Metrics/AbcSize
      private_class_method :pinhole_jacobian_vjp

      def clipped_ratio_and_mask(ratio, minimum, maximum)
        clipped = ratio.dup
        below = ratio.lt(minimum)
        above = ratio.gt(maximum)
        clipped[below] = minimum if below.any?
        clipped[above] = maximum if above.any?
        [clipped, below.eq(0) & above.eq(0)]
      end
      private_class_method :clipped_ratio_and_mask

      def world_to_cam_vjp(grad_means, grad_covars, viewmats)
        gaussian_count = grad_means.shape[1]
        world_means = grad_means.class.zeros(gaussian_count, 3)
        world_covars = grad_means.class.zeros(gaussian_count, 3, 3)
        grad_means.shape[0].times do |camera_index|
          rotation = viewmats[camera_index, 0...3, 0...3]
          rotation_batch = grad_means.class.zeros(gaussian_count, 3, 3)
          rotation_batch[true, true, true] = rotation
          world_means += grad_means[camera_index, true, true].dot(rotation)
          world_covars += Math::Mat.matmul_batch(
            Math::Mat.matmul_batch(rotation_batch.transpose(0, 2, 1),
                                   grad_covars[camera_index, true, true, true]),
            rotation_batch
          )
        end
        [world_means, world_covars]
      end
      private_class_method :world_to_cam_vjp

      # rubocop:disable Metrics/ParameterLists
      def input_covariance_gradients(means, covars, quaternions, scales, grad_means, grad_covars)
        # rubocop:enable Metrics/ParameterLists
        if covars
          return [
            grad_means,
            format_covariance_gradient(grad_covars, covars.shape, means.shape[0]),
            nil,
            nil
          ]
        end

        grad_quaternions, grad_scales = RubyQuatScaleToCovarPreci.backward(
          means.class.cast(quaternions),
          means.class.cast(scales),
          grad_covars,
          nil,
          triu: false
        )
        [grad_means, nil, grad_quaternions, grad_scales]
      end
      private_class_method :input_covariance_gradients

      def format_covariance_gradient(gradient, input_shape, gaussian_count)
        return gradient if input_shape == [gaussian_count, 3, 3]

        output = gradient.class.zeros(gaussian_count, 6)
        output[true, 0] = gradient[true, 0, 0]
        output[true, 1] = gradient[true, 0, 1] + gradient[true, 1, 0]
        output[true, 2] = gradient[true, 0, 2] + gradient[true, 2, 0]
        output[true, 3] = gradient[true, 1, 1]
        output[true, 4] = gradient[true, 1, 2] + gradient[true, 2, 1]
        output[true, 5] = gradient[true, 2, 2]
        output
      end
      private_class_method :format_covariance_gradient
    end
  end
end
