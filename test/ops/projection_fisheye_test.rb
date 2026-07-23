# frozen_string_literal: true

require "test_helper"

class ProjectionFisheyeTest < Minitest::Test
  class WeightedProjection < Gsplat::Autograd::Function
    def self.forward(context, means2d, conics, weights:)
      context.save(*weights)
      means2d.class.cast((means2d * weights[0]).sum + (conics * weights[1]).sum)
    end

    def self.backward(context, gradient)
      context.saved_values.map { |weight| weight * gradient.to_f }
    end
  end

  def setup
    @previous_backend = Gsplat.backend
    Gsplat.backend = :ruby
    @viewmats = Numo::DFloat.eye(4).reshape(1, 4, 4)
    @intrinsics = Numo::DFloat[[[100, 0, 50], [0, 100, 40], [0, 0, 1]]]
    @covariance = Numo::DFloat[[[0.01, 0, 0], [0, 0.01, 0], [0, 0, 0.01]]]
  end

  def teardown
    Gsplat.backend = @previous_backend
  end

  def test_equidistant_fisheye_projection
    _, means2d, = project(Numo::DFloat[[1, 0, 1]], camera_model: "fisheye")

    assert_allclose means2d, Numo::DFloat[[[50 + (25 * ::Math::PI), 40]]],
                    atol: 1e-9, rtol: 0.0
  end

  def test_opencv_radial_tangential_and_thin_prism_distortion
    options = {
      radial_coeffs: Numo::DFloat[[0.1, 0, 0, 0, 0, 0]],
      tangential_coeffs: Numo::DFloat[[0.01, 0.02]],
      thin_prism_coeffs: Numo::DFloat[[0.03, 0, 0.04, 0]]
    }
    _, means2d, = project(Numo::DFloat[[0.5, 0, 1]], **options)

    assert_allclose means2d, Numo::DFloat[[[103.5, 41.25]]], atol: 1e-9, rtol: 0.0
  end

  def test_fisheye_backward_matches_float64_central_difference
    means = Numo::DFloat[[0.2, -0.1, 1.5]]
    mean_var = Gsplat::Autograd::Variable.new(means, requires_grad: true)
    weights = [Numo::DFloat[[[0.3, -0.2]]], Numo::DFloat[[[0.1, -0.05, 0.2]]]]
    options = {
      camera_model: "fisheye",
      radial_coeffs: Numo::DFloat[[0.01, -0.002, 0.0003, 0]]
    }
    _, projected, _, conics, = project(mean_var, **options)
    WeightedProjection.apply(projected, conics, weights: weights).backward

    numeric = central_difference(means) do |value|
      _, value_projected, _, value_conics, = project(value, **options)
      (value_projected * weights[0]).sum + (value_conics * weights[1]).sum
    end
    assert_allclose mean_var.grad, numeric, atol: 2e-3, rtol: 2e-3
  end

  def test_native_backend_falls_back_with_identical_distortion
    means = Numo::SFloat[[0.2, -0.1, 1.5]]
    options = { camera_model: "fisheye", radial_coeffs: Numo::SFloat[[0.01, 0, 0, 0]] }
    ruby = with_backend(:ruby) { project(means, **options) }
    native = with_backend(:native) { project(means, **options) }

    ruby.zip(native).each do |expected, actual|
      next unless expected

      assert_allclose actual, expected, atol: 0.0, rtol: 0.0
    end
  end

  def test_matches_python_fisheye_golden
    fixture = golden("proj_fisheye_c1_n1000")
    outputs = Gsplat.fully_fused_projection(
      fixture.fetch("means"),
      quats: fixture.fetch("quats"),
      scales: fixture.fetch("scales"),
      viewmats: fixture.fetch("viewmats"),
      ks: fixture.fetch("ks"),
      width: fixture.fetch("width").to_i,
      height: fixture.fetch("height").to_i,
      calc_compensations: true,
      camera_model: "fisheye"
    )
    %w[radii means2d depths conics compensations].zip(outputs).each do |name, actual|
      assert_allclose actual, fixture.fetch(name), atol: 2e-3, rtol: 2e-3
    end
  end

  private

  def project(means, **)
    Gsplat.fully_fused_projection(
      means,
      covars: @covariance,
      viewmats: @viewmats,
      ks: @intrinsics,
      width: 100,
      height: 80,
      calc_compensations: true,
      **
    )
  end

  def central_difference(input, epsilon: 1e-5)
    gradient = input.class.zeros(*input.shape)
    input.size.times do |index|
      positive = input.flatten.dup
      negative = input.flatten.dup
      positive[index] += epsilon
      negative[index] -= epsilon
      gradient[index] = (
        yield(positive.reshape(*input.shape)) - yield(negative.reshape(*input.shape))
      ) / (2 * epsilon)
    end
    gradient
  end
end
