# frozen_string_literal: true

# High-level differentiable rendering API.
module Gsplat
  # Rendering pipeline composed from low-level operations.
  module Rasterization
    module_function

    # rubocop:disable Metrics/AbcSize, Metrics/MethodLength, Metrics/ParameterLists, Naming/MethodParameterName
    def render(means:, opacities:, colors:, viewmats:, ks:, width:, height:, quats: nil, scales: nil,
               covars: nil, near_plane: 0.01, far_plane: 1e10, radius_clip: 0.0, eps2d: 0.3,
               sh_degree: nil, tile_size: 16, backgrounds: nil, render_mode: "RGB",
               rasterize_mode: "classic", camera_model: "pinhole", absgrad: false,
               channel_chunk: 32, packed: false, global_z_order: true,
               radial_coeffs: nil, tangential_coeffs: nil, thin_prism_coeffs: nil)
      # rubocop:enable Metrics/ParameterLists, Naming/MethodParameterName
      RasterizationValidation.validate_options!(
        render_mode, rasterize_mode, sh_degree, tile_size, channel_chunk, width, height
      )
      RasterizationValidation.warn_unsupported_options(packed, global_z_order)
      RasterizationValidation.validate_scene!(
        means, quats, scales, covars, opacities, colors, viewmats, ks, sh_degree
      )
      camera_count = Ops::TensorOps.data(viewmats).shape[0]
      calculate_compensations = rasterize_mode == "antialiased"
      radii, means2d, depths, conics, compensations = Gsplat.fully_fused_projection(
        means,
        viewmats: viewmats,
        ks: ks,
        width: width,
        height: height,
        covars: covars,
        quats: quats,
        scales: scales,
        eps2d: eps2d,
        near_plane: near_plane,
        far_plane: far_plane,
        radius_clip: radius_clip,
        calc_compensations: calculate_compensations,
        camera_model: camera_model,
        radial_coeffs: radial_coeffs,
        tangential_coeffs: tangential_coeffs,
        thin_prism_coeffs: thin_prism_coeffs,
        global_z_order: global_z_order
      )
      projected_colors = RasterizationHelpers.prepare_colors(
        means, colors, viewmats, radii, camera_count, sh_degree
      )
      projected_opacities = RasterizationHelpers.camera_broadcast(opacities, camera_count)
      if calculate_compensations
        projected_opacities = Ops::TensorOps.apply(
          Ops::Multiply,
          projected_opacities,
          compensations
        )
      end
      raster_features, depth_index = RasterizationHelpers.render_features(
        projected_colors, depths, render_mode
      )
      tile_width = (width.to_f / tile_size).ceil
      tile_height = (height.to_f / tile_size).ceil
      tiles_per_gauss, isect_ids, flatten_ids = Gsplat.isect_tiles(
        Ops::TensorOps.data(means2d),
        Ops::TensorOps.data(radii),
        Ops::TensorOps.data(depths),
        tile_size,
        tile_width,
        tile_height
      )
      isect_offsets = Gsplat.isect_offset_encode(
        isect_ids, camera_count, tile_width, tile_height
      )
      render_colors, render_alphas = if %w[d Ed RGB-d RGB-Ed].include?(render_mode)
                                       RasterizationHelpers.rasterize_hit_features(
                                         means, quats, scales, raster_features, projected_opacities,
                                         viewmats, ks, width, height, tile_size, isect_offsets,
                                         flatten_ids, backgrounds: backgrounds,
                                                      camera_model: camera_model
                                       )
                                     else
                                       RasterizationHelpers.rasterize_features(
                                         means2d, conics, raster_features, projected_opacities,
                                         width, height, tile_size, isect_offsets, flatten_ids,
                                         backgrounds: backgrounds, channel_chunk: channel_chunk,
                                         absgrad: absgrad
                                       )
                                     end
      if %w[Ed ED RGB-Ed RGB+ED].include?(render_mode)
        render_colors = Ops::TensorOps.apply(
          Ops::NormalizeDepth,
          render_colors,
          render_alphas,
          depth_index: depth_index
        )
      end
      meta = RasterizationHelpers.metadata(
        radii, means2d, depths, conics, projected_opacities, tile_width, tile_height,
        tiles_per_gauss, isect_ids, flatten_ids, isect_offsets, width, height, tile_size,
        camera_count
      )
      means2d.bind_absgrad(meta, :means2d_absgrad) if absgrad && means2d.is_a?(Autograd::Variable)
      [render_colors, render_alphas, meta]
    end
    # rubocop:enable Metrics/AbcSize, Metrics/MethodLength
  end

  class << self
    def rasterization(**)
      Rasterization.render(**)
    end
  end
end
