# frozen_string_literal: true

# Public 2D Gaussian splatting API.
module Gsplat
  # High-level 2D Gaussian rendering composition.
  module Rasterization2DGS
    module_function

    # rubocop:disable Metrics/ParameterLists, Naming/MethodParameterName
    def render(means:, quats:, scales:, opacities:, colors:, viewmats:, ks:, width:, height:,
               near_plane: 0.01, far_plane: 1e10, radius_clip: 0.0, eps2d: 0.3,
               sh_degree: nil, packed: false, tile_size: 16, backgrounds: nil,
               render_mode: "RGB", sparse_grad: false, absgrad: false, distloss: false,
               depth_mode: "expected", rasterize_mode: "classic")
      # rubocop:enable Metrics/ParameterLists, Naming/MethodParameterName
      validate_options!(sparse_grad, depth_mode, rasterize_mode)
      primary, alphas, meta = Gsplat.rasterization(
        means: means, quats: quats, scales: scales, opacities: opacities, colors: colors,
        viewmats: viewmats, ks: ks, width: width, height: height, near_plane: near_plane,
        far_plane: far_plane, radius_clip: radius_clip, eps2d: eps2d, sh_degree: sh_degree,
        packed: packed, tile_size: tile_size, backgrounds: backgrounds,
        render_mode: render_mode, absgrad: absgrad, rasterize_mode: rasterize_mode
      )
      gaussian_normals = camera_normals(
        Ops::TensorOps.data(means), Ops::TensorOps.data(quats), Ops::TensorOps.data(viewmats)
      )
      render_normals, = Gsplat.rasterization(
        means: means, quats: quats, scales: scales, opacities: opacities,
        colors: gaussian_normals, viewmats: viewmats, ks: ks, width: width, height: height,
        near_plane: near_plane, far_plane: far_plane, radius_clip: radius_clip, eps2d: eps2d,
        tile_size: tile_size, render_mode: "RGB", absgrad: false
      )
      render_median, = Gsplat.rasterization(
        means: means, quats: quats, scales: scales, opacities: opacities,
        colors: gaussian_normals, viewmats: viewmats, ks: ks, width: width, height: height,
        near_plane: near_plane, far_plane: far_plane, radius_clip: radius_clip, eps2d: eps2d,
        tile_size: tile_size, render_mode: "ED", absgrad: false
      )
      surface_normals = normalized_normals(Ops::TensorOps.data(render_normals), Ops::TensorOps.data(alphas))
      render_distort = distortion_image(
        Ops::TensorOps.data(render_median), Ops::TensorOps.data(alphas), distloss
      )
      meta = two_d_metadata(meta, gaussian_normals, render_distort)
      [primary, alphas, render_normals, surface_normals, render_distort, render_median, meta]
    end

    def validate_options!(sparse_grad, depth_mode, rasterize_mode)
      raise ArgumentError, "sparse_grad requires packed mode and is unsupported" if sparse_grad
      raise ArgumentError, "2DGS only supports classic rasterization" unless rasterize_mode == "classic"
      return if %w[expected median].include?(depth_mode.to_s)

      raise ArgumentError, "depth_mode must be \"expected\" or \"median\""
    end
    private_class_method :validate_options!

    def camera_normals(means, quats, viewmats)
      rotations = Math::Quaternion.to_rotmat(quats)
      world_normals = rotations[true, true, 2]
      output = means.class.zeros(viewmats.shape[0], means.shape[0], 3)
      viewmats.shape[0].times do |camera|
        view_rotation = viewmats[camera, 0...3, 0...3]
        camera_means = means.dot(view_rotation.transpose) + viewmats[camera, 0...3, 3]
        normals = world_normals.dot(view_rotation.transpose)
        facing_away = (normals * camera_means).sum(axis: 1).gt(0)
        normals[facing_away, true] = -normals[facing_away, true] if facing_away.any?
        output[camera, true, true] = normals
      end
      output
    end
    private_class_method :camera_normals

    def normalized_normals(normals, alphas)
      alpha = alphas[true, true, true, 0]
      output = normals.class.zeros(*normals.shape)
      valid = alpha.gt(1e-10)
      return output unless valid.any?

      expected = normals.dup
      3.times { |axis| expected[true, true, true, axis][valid] /= alpha[valid] }
      norm = (expected**2).sum(axis: 3)**0.5
      valid &= norm.gt(1e-10)
      3.times { |axis| output[true, true, true, axis][valid] = expected[true, true, true, axis][valid] / norm[valid] }
      output
    end
    private_class_method :normalized_normals

    def distortion_image(median, alphas, enabled)
      return median.class.zeros(*median.shape) unless enabled

      alpha = alphas[true, true, true, 0]
      depth = median[true, true, true, 0]
      output = median.class.zeros(*median.shape)
      output[true, true, true, 0] = (depth * (1 - alpha)).abs
      output
    end
    private_class_method :distortion_image

    def two_d_metadata(meta, normals, distortion)
      ray_transforms = normals.class.zeros(*(normals.shape[0...-1] + [3, 3]))
      ray_transforms[true, true, true, true] = normals.class.eye(3)
      meta.merge(
        ray_transforms: ray_transforms,
        normals: normals,
        render_distort: distortion,
        gradient_2dgs: meta.fetch(:means2d)
      )
    end
    private_class_method :two_d_metadata
  end

  class << self
    # Rasterizes oriented 2D Gaussian surfels.
    #
    # @return [Array] colors, alphas, normals, surface normals, distortion, median depth, metadata
    def rasterization_2dgs(**)
      Rasterization2DGS.render(**)
    end

    # Projection metadata used by the 2DGS rasterizer.
    # rubocop:disable Metrics/ParameterLists, Naming/MethodParameterName
    def fully_fused_projection_2dgs(means, quats:, scales:, viewmats:, ks:, width:, height:,
                                    eps2d: 0.3, near_plane: 0.01, far_plane: 1e10,
                                    radius_clip: 0.0)
      # rubocop:enable Metrics/ParameterLists, Naming/MethodParameterName
      radii, means2d, depths, conics, = fully_fused_projection(
        means, quats: quats, scales: scales, viewmats: viewmats, ks: ks, width: width,
               height: height, eps2d: eps2d, near_plane: near_plane, far_plane: far_plane,
               radius_clip: radius_clip
      )
      normals = Rasterization2DGS.send(
        :camera_normals, Ops::TensorOps.data(means), Ops::TensorOps.data(quats),
        Ops::TensorOps.data(viewmats)
      )
      ray_transforms = Ops::TensorOps.data(conics).class.zeros(
        *(Ops::TensorOps.data(conics).shape[0...-1] + [3, 3])
      )
      [radii, means2d, depths, ray_transforms, normals]
    end
  end
end
