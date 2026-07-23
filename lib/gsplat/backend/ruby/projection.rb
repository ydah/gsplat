# frozen_string_literal: true

module Gsplat
  module Backend
    # Numo implementation of dense fused Gaussian projection.
    module RubyProjection
      module_function

      # rubocop:disable Metrics/ParameterLists
      def forward(means, covars, quaternions, scales, viewmats, intrinsics, width, height,
                  eps2d:, near_plane:, far_plane:, radius_clip:, calc_compensations:, camera_model:)
        # rubocop:enable Metrics/ParameterLists
        inputs = prepare_inputs(
          means, covars, quaternions, scales, viewmats, intrinsics, width, height,
          eps2d, near_plane, far_plane, radius_clip, camera_model
        )
        means, covars, viewmats, intrinsics = inputs
        camera_means, camera_covars = Math::CameraProjection.world_to_cam(means, covars, viewmats)
        projection_means = projection_safe_means(camera_means, near_plane, far_plane)
        means2d, covars2d = project(
          projection_means, camera_covars, intrinsics, width, height, camera_model
        )
        finish_projection(
          camera_means,
          means2d,
          covars2d,
          width,
          height,
          eps2d,
          near_plane,
          far_plane,
          radius_clip,
          calc_compensations
        )
      end

      # rubocop:disable Metrics/ParameterLists
      def prepare_inputs(means, covars, quaternions, scales, viewmats, intrinsics, width, height,
                         eps2d, near_plane, far_plane, radius_clip, camera_model)
        # rubocop:enable Metrics/ParameterLists
        unless means.is_a?(Numo::NArray) && [Numo::SFloat, Numo::DFloat].include?(means.class)
          raise ArgumentError, "means must be Numo::SFloat or Numo::DFloat"
        end

        valid_means = means.ndim == 2 && means.shape[-1] == 3
        raise ShapeError, "expected means [N,3], got #{means.shape.inspect}" unless valid_means

        covars = prepare_covariances(means, covars, quaternions, scales)
        viewmats = means.class.cast(viewmats)
        intrinsics = means.class.cast(intrinsics)
        unless viewmats.ndim == 3 && viewmats.shape[1..] == [4, 4]
          raise ShapeError, "expected viewmats [C,4,4], got #{viewmats.shape.inspect}"
        end
        unless intrinsics.shape == [viewmats.shape[0], 3, 3]
          raise ShapeError, "expected intrinsics [#{viewmats.shape[0]},3,3], got #{intrinsics.shape.inspect}"
        end

        validate_options!(
          width: width,
          height: height,
          eps2d: eps2d,
          near_plane: near_plane,
          far_plane: far_plane,
          radius_clip: radius_clip,
          camera_model: camera_model
        )
        [means, covars, viewmats, intrinsics]
      end

      def prepare_covariances(means, covars, quaternions, scales)
        if covars
          raise ArgumentError, "provide covars or quaternions/scales, not both" if quaternions || scales

          return expand_covariances(means.class.cast(covars), means.shape[0])
        end
        raise ArgumentError, "quaternions and scales are required when covars is nil" unless quaternions && scales

        covariance, = RubyQuatScaleToCovarPreci.forward(
          means.class.cast(quaternions),
          means.class.cast(scales),
          compute_covar: true,
          compute_preci: false,
          triu: false
        )
        covariance
      end
      private_class_method :prepare_covariances

      def expand_covariances(covars, gaussian_count)
        return covars if covars.shape == [gaussian_count, 3, 3]
        unless covars.shape == [gaussian_count, 6]
          raise ShapeError, "expected covars [N,3,3] or [N,6], got #{covars.shape.inspect}"
        end

        output = covars.class.zeros(gaussian_count, 3, 3)
        [[0, 0, 0], [0, 1, 1], [0, 2, 2], [1, 1, 3], [1, 2, 4], [2, 2, 5]].each do |row, column, index|
          output[true, row, column] = covars[true, index]
          output[true, column, row] = covars[true, index]
        end
        output
      end
      private_class_method :expand_covariances

      # rubocop:disable Metrics/ParameterLists
      def validate_options!(width:, height:, eps2d:, near_plane:, far_plane:, radius_clip:, camera_model:)
        # rubocop:enable Metrics/ParameterLists
        unless width.is_a?(Integer) && width.positive? && height.is_a?(Integer) && height.positive?
          raise ArgumentError, "width and height must be positive integers"
        end
        raise ArgumentError, "eps2d and radius_clip must be non-negative" if eps2d.negative? || radius_clip.negative?
        raise ArgumentError, "near_plane must be less than far_plane" unless near_plane < far_plane
        return if %w[pinhole ortho].include?(camera_model.to_s)

        raise ArgumentError, "camera_model must be \"pinhole\" or \"ortho\""
      end
      private_class_method :validate_options!

      # rubocop:disable Metrics/ParameterLists
      def project(means, covars, intrinsics, width, height, camera_model)
        # rubocop:enable Metrics/ParameterLists
        if camera_model.to_s == "ortho"
          return Math::CameraProjection.ortho_proj(means, covars, intrinsics, width, height)
        end

        Math::CameraProjection.persp_proj(means, covars, intrinsics, width, height)
      end

      def projection_safe_means(camera_means, near_plane, far_plane)
        output = camera_means.dup
        depths = output[true, true, 2]
        culled = depths.le(near_plane) | depths.ge(far_plane)
        depths[culled] = 1 if culled.any?
        output[true, true, 2] = depths
        output
      end
      private_class_method :projection_safe_means

      # rubocop:disable Metrics/AbcSize, Metrics/ParameterLists
      def finish_projection(camera_means, means2d, covars2d, width, height, eps2d,
                            near_plane, far_plane, radius_clip, calc_compensations)
        # rubocop:enable Metrics/ParameterLists
        determinant_original = Math::Mat.det2x2(covars2d)
        regularized = covars2d.dup
        regularized[true, true, 0, 0] += eps2d
        regularized[true, true, 1, 1] += eps2d
        determinant = Math::Mat.det2x2(regularized)
        safe_determinant = determinant.dup
        invalid_determinant = safe_determinant.le(0)
        safe_determinant[invalid_determinant] = 1e-10 if invalid_determinant.any?
        conics = conics_from_covariance(regularized, safe_determinant)
        compensations = compensation(determinant_original, safe_determinant) if calc_compensations
        largest_eigenvalue = Math::Mat.eigvals2x2(regularized)[true, true, 1]
        largest_eigenvalue[largest_eigenvalue.lt(0)] = 0 if largest_eigenvalue.lt(0).any?
        radii = Numo::Int32.cast((3.33 * (largest_eigenvalue**0.5)).ceil)
        depths = camera_means[true, true, 2].dup
        visible = determinant.gt(0) & depths.gt(near_plane) & depths.lt(far_plane) & radii.gt(radius_clip)
        visible &= (means2d[true, true, 0] + radii).gt(0)
        visible &= means2d[true, true, 0] - radii < width
        visible &= (means2d[true, true, 1] + radii).gt(0)
        visible &= means2d[true, true, 1] - radii < height
        radii[visible.eq(0)] = 0
        [radii, means2d, depths, conics, compensations]
      end
      # rubocop:enable Metrics/AbcSize
      private_class_method :finish_projection

      def conics_from_covariance(covariance, determinant)
        output = covariance.class.zeros(*(covariance.shape[0...-2] + [3]))
        output[true, true, 0] = covariance[true, true, 1, 1] / determinant
        output[true, true, 1] = -(
          covariance[true, true, 0, 1] + covariance[true, true, 1, 0]
        ) / (2.0 * determinant)
        output[true, true, 2] = covariance[true, true, 0, 0] / determinant
        output
      end
      private_class_method :conics_from_covariance

      def compensation(original, regularized)
        ratio = original / regularized
        negative = ratio.lt(0)
        ratio[negative] = 0 if negative.any?
        ratio**0.5
      end
      private_class_method :compensation
    end
  end
end
