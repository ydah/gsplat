# frozen_string_literal: true

module Gsplat
  # Input and option validation for the high-level renderer.
  module RasterizationValidation
    RENDER_MODES = %w[RGB d Ed D ED RGB-d RGB-Ed RGB+D RGB+ED].freeze
    RASTERIZE_MODES = %w[classic antialiased].freeze

    module_function

    # rubocop:disable Metrics/ParameterLists
    def validate_options!(render_mode, rasterize_mode, sh_degree, tile_size, channel_chunk, width, height)
      # rubocop:enable Metrics/ParameterLists
      raise ArgumentError, "unsupported render_mode #{render_mode.inspect}" unless RENDER_MODES.include?(render_mode)
      unless RASTERIZE_MODES.include?(rasterize_mode)
        raise ArgumentError, "unsupported rasterize_mode #{rasterize_mode.inspect}"
      end

      unless sh_degree.nil? || (0..4).cover?(sh_degree)
        raise ArgumentError, "sh_degree must be nil or an integer from 0 through 4"
      end

      values = { width: width, height: height, tile_size: tile_size, channel_chunk: channel_chunk }
      invalid = values.find { |_name, value| !value.is_a?(Integer) || !value.positive? }
      raise ArgumentError, "#{invalid[0]} must be a positive integer" if invalid
    end

    def warn_unsupported_options(packed, _global_z_order)
      Gsplat.logger.warn("packed mode is unsupported; using dense mode") if packed
    end

    def validate_eval3d!(with_eval3d, return_normals, covars, rasterize_mode)
      raise ArgumentError, "return_normals requires with_eval3d: true" if return_normals && !with_eval3d
      raise ArgumentError, "with_eval3d requires quats and scales instead of covars" if with_eval3d && covars
      return unless with_eval3d && rasterize_mode != "classic"

      raise ArgumentError, "with_eval3d requires rasterize_mode: \"classic\""
    end

    # rubocop:disable Metrics/ParameterLists
    def validate_scene!(means, quats, scales, covars, opacities, colors, viewmats, intrinsics, sh_degree)
      # rubocop:enable Metrics/ParameterLists
      means_shape = data(means).shape
      unless means_shape.length == 2 && means_shape[-1] == 3
        raise ShapeError, "expected means [N,3], got #{means_shape.inspect}"
      end

      gaussian_count = means_shape[0]
      opacity_shape = data(opacities).shape
      unless [[gaussian_count], [data(viewmats).shape[0], gaussian_count]].include?(opacity_shape)
        raise ShapeError, "expected opacities [N] or [C,N], got #{opacity_shape.inspect}"
      end

      validate_geometry!(quats, scales, covars, gaussian_count)
      camera_count = data(viewmats).shape[0]
      unless data(viewmats).shape == [camera_count, 4, 4] && data(intrinsics).shape == [camera_count, 3, 3]
        raise ShapeError,
              "expected viewmats [C,4,4] and ks [C,3,3], " \
              "got #{data(viewmats).shape.inspect} and #{data(intrinsics).shape.inspect}"
      end

      validate_color_shape!(data(colors).shape, gaussian_count, camera_count, sh_degree)
    end

    def validate_geometry!(quats, scales, covars, gaussian_count)
      if covars
        covar_shape = data(covars).shape
        return if [[gaussian_count, 3, 3], [gaussian_count, 6]].include?(covar_shape)

        raise ShapeError, "expected covars [N,3,3] or [N,6], got #{covar_shape.inspect}"
      end
      return if data(quats).shape == [gaussian_count, 4] && data(scales).shape == [gaussian_count, 3]

      raise ShapeError,
            "expected quats [N,4] and scales [N,3], " \
            "got #{data(quats).shape.inspect} and #{data(scales).shape.inspect}"
    end
    private_class_method :validate_geometry!

    def validate_color_shape!(shape, gaussian_count, camera_count, sh_degree)
      valid = if sh_degree
                shape.length == 3 && shape[0] == gaussian_count && shape[1] >= ((sh_degree + 1)**2)
              else
                (shape.length == 2 && shape[0] == gaussian_count) ||
                  (shape.length == 3 && shape[0...2] == [camera_count, gaussian_count])
              end
      return if valid

      expected = sh_degree ? "[N,K,D] with K >= #{(sh_degree + 1)**2}" : "[N,D] or [C,N,D]"
      raise ShapeError, "expected colors #{expected}, got #{shape.inspect}"
    end
    private_class_method :validate_color_shape!

    def data(value)
      Ops::TensorOps.data(value)
    end
    private_class_method :data
  end
end
