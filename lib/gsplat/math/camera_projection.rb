# frozen_string_literal: true

module Gsplat
  module Math
    # Camera-space transformation and pinhole Gaussian projection primitives.
    module CameraProjection
      module_function

      def world_to_cam(means, covars, viewmats)
        means, covars, viewmats = validate_world_inputs(means, covars, viewmats)
        camera_count = viewmats.shape[0]
        gaussian_count = means.shape[0]
        camera_means = means.class.zeros(camera_count, gaussian_count, 3)
        camera_covars = means.class.zeros(camera_count, gaussian_count, 3, 3)
        camera_count.times do |camera_index|
          rotation = viewmats[camera_index, 0...3, 0...3]
          translation = viewmats[camera_index, 0...3, 3]
          rotation_batch = means.class.zeros(gaussian_count, 3, 3)
          rotation_batch[true, true, true] = rotation
          camera_means[camera_index, true, true] = means.dot(rotation.transpose) + translation
          rotated = Mat.matmul_batch(rotation_batch, covars)
          camera_covars[camera_index, true, true, true] = Mat.matmul_batch(
            rotated,
            rotation_batch.transpose(0, 2, 1)
          )
        end
        [camera_means, camera_covars]
      end

      def persp_proj(means, covars, intrinsics, width, height)
        means, covars, intrinsics = validate_projection_inputs(means, covars, intrinsics, width, height)
        camera_count = means.shape[0]
        gaussian_count = means.shape[1]
        means2d = means.class.zeros(camera_count, gaussian_count, 2)
        covars2d = means.class.zeros(camera_count, gaussian_count, 2, 2)
        camera_count.times do |camera_index|
          projected_means, jacobians = pinhole_camera(
            means[camera_index, true, true],
            intrinsics[camera_index, true, true],
            width,
            height
          )
          means2d[camera_index, true, true] = projected_means
          covariance = covars[camera_index, true, true, true]
          transformed = Mat.matmul_batch(jacobians, covariance)
          covars2d[camera_index, true, true, true] = Mat.matmul_batch(
            transformed,
            jacobians.transpose(0, 2, 1)
          )
        end
        [means2d, covars2d]
      end

      def ortho_proj(means, covars, intrinsics, width, height)
        means, covars, intrinsics = validate_projection_inputs(means, covars, intrinsics, width, height)
        camera_count = means.shape[0]
        gaussian_count = means.shape[1]
        means2d = means.class.zeros(camera_count, gaussian_count, 2)
        covars2d = means.class.zeros(camera_count, gaussian_count, 2, 2)
        camera_count.times do |camera_index|
          projected_means, jacobians = ortho_camera(
            means[camera_index, true, true],
            intrinsics[camera_index, true, true]
          )
          means2d[camera_index, true, true] = projected_means
          transformed = Mat.matmul_batch(jacobians, covars[camera_index, true, true, true])
          covars2d[camera_index, true, true, true] = Mat.matmul_batch(
            transformed,
            jacobians.transpose(0, 2, 1)
          )
        end
        [means2d, covars2d]
      end

      # rubocop:disable Metrics/ParameterLists
      def distorted_proj(means, covars, intrinsics, camera_model, radial_coeffs: nil,
                         tangential_coeffs: nil, thin_prism_coeffs: nil)
        # rubocop:enable Metrics/ParameterLists
        camera_count = means.shape[0]
        gaussian_count = means.shape[1]
        means2d = means.class.zeros(camera_count, gaussian_count, 2)
        covars2d = means.class.zeros(camera_count, gaussian_count, 2, 2)
        camera_count.times do |camera_index|
          projected, jacobians = CameraDistortion.project_camera(
            means[camera_index, true, true],
            intrinsics[camera_index, true, true],
            camera_model,
            radial: radial_coeffs && radial_coeffs[camera_index, true].to_a,
            tangential: tangential_coeffs && tangential_coeffs[camera_index, true].to_a,
            thin_prism: thin_prism_coeffs && thin_prism_coeffs[camera_index, true].to_a
          )
          means2d[camera_index, true, true] = projected
          transformed = Mat.matmul_batch(jacobians, covars[camera_index, true, true, true])
          covars2d[camera_index, true, true, true] = Mat.matmul_batch(
            transformed,
            jacobians.transpose(0, 2, 1)
          )
        end
        [means2d, covars2d]
      end

      def validate_world_inputs(means, covars, viewmats)
        validate_float_array!(means, "means")
        covars = means.class.cast(covars)
        viewmats = means.class.cast(viewmats)
        unless means.ndim == 2 && means.shape[-1] == 3 && covars.shape == [means.shape[0], 3, 3]
          raise ShapeError,
                "expected means [N,3] and covars [N,3,3], " \
                "got #{means.shape.inspect} and #{covars.shape.inspect}"
        end
        unless viewmats.ndim == 3 && viewmats.shape[1..] == [4, 4]
          raise ShapeError, "expected viewmats [C,4,4], got #{viewmats.shape.inspect}"
        end

        [means, covars, viewmats]
      end
      private_class_method :validate_world_inputs

      def validate_projection_inputs(means, covars, intrinsics, width, height)
        validate_float_array!(means, "means")
        covars = means.class.cast(covars)
        intrinsics = means.class.cast(intrinsics)
        valid = means.ndim == 3 && means.shape[-1] == 3 &&
                covars.shape == means.shape[0...-1] + [3, 3] &&
                intrinsics.shape == [means.shape[0], 3, 3]
        unless valid
          raise ShapeError,
                "expected means [C,N,3], covars [C,N,3,3], and intrinsics [C,3,3], " \
                "got #{means.shape.inspect}, #{covars.shape.inspect}, and #{intrinsics.shape.inspect}"
        end
        unless width.is_a?(Integer) && width.positive? && height.is_a?(Integer) && height.positive?
          raise ArgumentError, "width and height must be positive integers"
        end

        [means, covars, intrinsics]
      end
      private_class_method :validate_projection_inputs

      def validate_float_array!(array, name)
        return if array.is_a?(Numo::NArray) && [Numo::SFloat, Numo::DFloat].include?(array.class)

        raise ArgumentError, "#{name} must be Numo::SFloat or Numo::DFloat"
      end
      private_class_method :validate_float_array!

      # rubocop:disable Metrics/AbcSize
      def pinhole_camera(means, intrinsics, width, height)
        x_coord = means[true, 0]
        y_coord = means[true, 1]
        z_coord = means[true, 2]
        focal_x = intrinsics[0, 0]
        focal_y = intrinsics[1, 1]
        center_x = intrinsics[0, 2]
        center_y = intrinsics[1, 2]
        clipped_x = clip_ratio(
          x_coord / z_coord,
          -((center_x / focal_x) + (0.15 * width / focal_x)),
          ((width - center_x) / focal_x) + (0.15 * width / focal_x)
        ) * z_coord
        clipped_y = clip_ratio(
          y_coord / z_coord,
          -((center_y / focal_y) + (0.15 * height / focal_y)),
          ((height - center_y) / focal_y) + (0.15 * height / focal_y)
        ) * z_coord
        jacobians = means.class.zeros(means.shape[0], 2, 3)
        jacobians[true, 0, 0] = focal_x / z_coord
        jacobians[true, 0, 2] = -focal_x * clipped_x / (z_coord**2)
        jacobians[true, 1, 1] = focal_y / z_coord
        jacobians[true, 1, 2] = -focal_y * clipped_y / (z_coord**2)
        homogeneous_x = (intrinsics[0, 0] * x_coord) + (intrinsics[0, 1] * y_coord) +
                        (intrinsics[0, 2] * z_coord)
        homogeneous_y = (intrinsics[1, 0] * x_coord) + (intrinsics[1, 1] * y_coord) +
                        (intrinsics[1, 2] * z_coord)
        means2d = means.class.zeros(means.shape[0], 2)
        means2d[true, 0] = homogeneous_x / z_coord
        means2d[true, 1] = homogeneous_y / z_coord
        [means2d, jacobians]
      end
      # rubocop:enable Metrics/AbcSize

      def clip_ratio(values, minimum, maximum)
        output = values.dup
        below = output.lt(minimum)
        above = output.gt(maximum)
        output[below] = minimum if below.any?
        output[above] = maximum if above.any?
        output
      end
      private_class_method :clip_ratio

      def ortho_camera(means, intrinsics)
        jacobians = means.class.zeros(means.shape[0], 2, 3)
        jacobians[true, 0, 0] = intrinsics[0, 0]
        jacobians[true, 1, 1] = intrinsics[1, 1]
        means2d = means.class.zeros(means.shape[0], 2)
        means2d[true, 0] = (means[true, 0] * intrinsics[0, 0]) + intrinsics[0, 2]
        means2d[true, 1] = (means[true, 1] * intrinsics[1, 1]) + intrinsics[1, 2]
        [means2d, jacobians]
      end
    end
  end
end
