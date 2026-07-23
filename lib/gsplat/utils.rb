# frozen_string_literal: true

module Gsplat
  # Point-cloud initialization and scene geometry utilities.
  module Utils
    # Degree-zero real spherical-harmonic basis constant.
    SH_C0 = 0.28209479177387814

    module_function

    # Brute-force Euclidean nearest-neighbor distances, including self at zero.
    # rubocop:disable Naming/MethodParameterName
    def knn(points, k: 4)
      validate_points!(points)
      unless k.is_a?(Integer) && k.between?(1, points.shape[0])
        raise ArgumentError, "k must be in 1..#{points.shape[0]}"
      end

      output = points.class.zeros(points.shape[0], k)
      points.shape[0].times do |index|
        distances = Numo::NMath.sqrt(((points - points[index, true])**2).sum(axis: 1))
        output[index, true] = distances.sort[0...k]
      end
      output
    end
    # rubocop:enable Naming/MethodParameterName

    def rgb_to_sh(rgb)
      (rgb - 0.5) / SH_C0
    end

    # Converts degree-zero SH coefficients back to RGB values.
    #
    # @param coefficients [Numo::NArray] [...,3]
    # @return [Numo::NArray] [...,3]
    def sh_to_rgb(coefficients)
      (coefficients * SH_C0) + 0.5
    end

    # Initializes raw trainable Gaussian parameters from colored 3D points.
    # rubocop:disable Metrics/AbcSize
    def init_from_points(points, colors, **options)
      sh_degree = options.fetch(:sh_degree, 3)
      init_opacity = options.fetch(:init_opacity, 0.1)
      init_scale = options.fetch(:init_scale, 1.0)
      rng = options.fetch(:rng, Gsplat.rng)
      validate_initialization!(points, colors, sh_degree, init_opacity, init_scale)
      count = points.shape[0]
      neighbor_count = [4, count].min
      squared = knn(points, k: neighbor_count)[true, 1...neighbor_count]**2
      distance = Numo::NMath.sqrt(squared.mean(axis: 1))
      epsilon = points.is_a?(Numo::DFloat) ? 1e-12 : 1e-6
      distance[distance.lt(epsilon)] = epsilon
      log_scales = Numo::NMath.log(distance * init_scale).reshape(count, 1)
      scales = points.class.zeros(count, 3)
      scales[true, true] = log_scales
      quaternions = points.class.cast(Array.new(count * 4) { rng.rand }).reshape(count, 4)
      opacity = ::Math.log(init_opacity / (1 - init_opacity))
      sh0 = rgb_to_sh(points.class.cast(colors)).reshape(count, 1, 3)
      shn = points.class.zeros(count, ((sh_degree + 1)**2) - 1, 3)
      {
        means: variable(points.dup),
        scales: variable(scales),
        quats: variable(quaternions),
        opacities: variable(points.class.ones(count) * opacity),
        sh0: variable(sh0),
        shN: variable(shn)
      }
    end
    # rubocop:enable Metrics/AbcSize

    # Maximum camera-center distance from the mean center.
    def scene_scale(camera_to_worlds)
      unless camera_to_worlds.is_a?(Numo::NArray) &&
             camera_to_worlds.ndim == 3 && camera_to_worlds.shape[1..] == [4, 4]
        actual = camera_to_worlds.respond_to?(:shape) ? camera_to_worlds.shape.inspect : camera_to_worlds.class
        raise ShapeError, "expected camera_to_worlds [C,4,4], got #{actual}"
      end

      locations = camera_to_worlds[true, 0...3, 3]
      center = locations.mean(axis: 0)
      Numo::NMath.sqrt(((locations - center)**2).sum(axis: 1)).max.to_f
    end

    def validate_points!(points)
      valid = points.is_a?(Numo::NArray) &&
              [Numo::SFloat, Numo::DFloat].include?(points.class) &&
              points.ndim == 2 && points.shape[1] == 3 && points.shape[0].positive?
      raise ShapeError, "expected points [N,3] floating array" unless valid
    end
    private_class_method :validate_points!

    def validate_initialization!(points, colors, degree, opacity, scale)
      validate_points!(points)
      raise ArgumentError, "at least two points are required" if points.shape[0] < 2
      unless colors.is_a?(Numo::NArray) && colors.shape == [points.shape[0], 3]
        raise ShapeError, "expected colors [#{points.shape[0]},3], got #{colors.shape.inspect}"
      end
      raise ArgumentError, "sh_degree must be in 0..4" unless degree.is_a?(Integer) && degree.between?(0, 4)
      raise ArgumentError, "init_opacity must be in (0,1)" unless opacity.positive? && opacity < 1
      raise ArgumentError, "init_scale must be positive" unless scale.positive?
    end
    private_class_method :validate_initialization!

    def variable(data)
      Autograd::Variable.new(data, requires_grad: true)
    end
    private_class_method :variable
  end
end
