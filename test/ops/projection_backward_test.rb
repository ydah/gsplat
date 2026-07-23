# frozen_string_literal: true

require "test_helper"

class ProjectionBackwardTest < Minitest::Test
  class ProjectionLoss < Gsplat::Autograd::Function
    # rubocop:disable Metrics/ParameterLists
    def self.forward(context, means2d, depths, conics, compensations, weights:)
      # rubocop:enable Metrics/ParameterLists
      context.save(*weights)
      value = (means2d * weights[0]).sum + (depths * weights[1]).sum + (conics * weights[2]).sum
      value += (compensations * weights[3]).sum
      means2d.class.cast(value)
    end

    def self.backward(context, grad_output)
      context.saved_values.map { |weight| weight * grad_output.to_f }
    end
  end

  def setup
    @previous_backend = Gsplat.backend
    Gsplat.backend = :ruby
    @means = Numo::DFloat[[0.1, -0.05, 2.0]]
    @quaternions = Numo::DFloat[[0.9, 0.1, -0.2, 0.3]]
    @scales = Numo::DFloat[[0.12, 0.08, 0.1]]
    @viewmats = Numo::DFloat.zeros(1, 4, 4)
    @viewmats[0, true, true] = Numo::DFloat.eye(4)
    @intrinsics = Numo::DFloat[[[80, 0, 32], [0, 82, 24], [0, 0, 1]]]
    @weights = [
      Numo::DFloat[[[0.3, -0.7]]],
      Numo::DFloat[[0.2]],
      Numo::DFloat[[[-0.4, 0.6, 0.1]]],
      Numo::DFloat[[0.5]]
    ]
  end

  def teardown
    Gsplat.backend = @previous_backend
  end

  def test_quaternion_scale_backward_matches_float64_central_difference
    mean_var = Gsplat::Autograd::Variable.new(@means, requires_grad: true)
    quaternion_var = Gsplat::Autograd::Variable.new(@quaternions, requires_grad: true)
    scale_var = Gsplat::Autograd::Variable.new(@scales, requires_grad: true)
    _, means2d, depths, conics, compensations = Gsplat.fully_fused_projection(
      mean_var,
      quats: quaternion_var,
      scales: scale_var,
      **projection_options
    )

    ProjectionLoss.apply(means2d, depths, conics, compensations, weights: @weights).backward

    numeric_means = central_difference(@means) { |value| weighted_projection(value, @quaternions, @scales) }
    numeric_quaternions = central_difference(@quaternions) { |value| weighted_projection(@means, value, @scales) }
    numeric_scales = central_difference(@scales) { |value| weighted_projection(@means, @quaternions, value) }
    assert_allclose mean_var.grad, numeric_means, atol: 2e-6, rtol: 2e-5
    assert_allclose quaternion_var.grad, numeric_quaternions, atol: 2e-6, rtol: 2e-5
    assert_allclose scale_var.grad, numeric_scales, atol: 2e-6, rtol: 2e-5
  end

  def test_direct_covariance_backward_matches_float64_central_difference
    covariance, = Gsplat.quat_scale_to_covar_preci(
      @quaternions,
      @scales,
      compute_preci: false
    )
    covariance_var = Gsplat::Autograd::Variable.new(covariance, requires_grad: true)
    _, means2d, depths, conics, compensations = Gsplat.fully_fused_projection(
      @means,
      covars: covariance_var,
      **projection_options
    )

    ProjectionLoss.apply(means2d, depths, conics, compensations, weights: @weights).backward

    numeric_covariance = central_difference(covariance) do |value|
      weighted_projection_with_covariance(value)
    end
    assert_allclose covariance_var.grad, numeric_covariance, atol: 2e-6, rtol: 2e-5
  end

  def test_matches_python_projection_gradients
    fixture = golden("proj_pinhole_c1_n1000")
    mean_var = Gsplat::Autograd::Variable.new(fixture.fetch("means"), requires_grad: true)
    quaternion_var = Gsplat::Autograd::Variable.new(fixture.fetch("quats"), requires_grad: true)
    scale_var = Gsplat::Autograd::Variable.new(fixture.fetch("scales"), requires_grad: true)
    _, means2d, depths, conics, compensations = Gsplat.fully_fused_projection(
      mean_var,
      quats: quaternion_var,
      scales: scale_var,
      viewmats: fixture.fetch("viewmats"),
      ks: fixture.fetch("ks"),
      width: fixture.fetch("width").to_i,
      height: fixture.fetch("height").to_i,
      calc_compensations: true
    )
    weights = %w[weight_means2d weight_depths weight_conics weight_compensations].map do |name|
      fixture.fetch(name)
    end

    ProjectionLoss.apply(means2d, depths, conics, compensations, weights: weights).backward

    assert_allclose mean_var.grad, fixture.fetch("grad_means"), atol: 1e-4, rtol: 1e-4
    assert_allclose quaternion_var.grad, fixture.fetch("grad_quats"), atol: 1e-4, rtol: 1e-4
    assert_allclose scale_var.grad, fixture.fetch("grad_scales"), atol: 1e-4, rtol: 1e-4
  end

  def test_matches_python_direct_covariance_gradients
    fixture = golden("proj_covars_c1_n1000")
    mean_var = Gsplat::Autograd::Variable.new(fixture.fetch("means"), requires_grad: true)
    covariance_var = Gsplat::Autograd::Variable.new(fixture.fetch("covars"), requires_grad: true)
    _, means2d, depths, conics, compensations = Gsplat.fully_fused_projection(
      mean_var,
      covars: covariance_var,
      viewmats: fixture.fetch("viewmats"),
      ks: fixture.fetch("ks"),
      width: fixture.fetch("width").to_i,
      height: fixture.fetch("height").to_i,
      calc_compensations: true
    )
    weights = %w[weight_means2d weight_depths weight_conics weight_compensations].map do |name|
      fixture.fetch(name)
    end

    ProjectionLoss.apply(means2d, depths, conics, compensations, weights: weights).backward

    assert_allclose mean_var.grad, fixture.fetch("grad_means"), atol: 1e-4, rtol: 1e-4
    assert_allclose covariance_var.grad, fixture.fetch("grad_covars"), atol: 1e-4, rtol: 1e-4
  end

  private

  def projection_options
    {
      viewmats: @viewmats,
      ks: @intrinsics,
      width: 64,
      height: 48,
      calc_compensations: true
    }
  end

  def weighted_projection(means, quaternions, scales)
    _, means2d, depths, conics, compensations = Gsplat.fully_fused_projection(
      means,
      quats: quaternions,
      scales: scales,
      **projection_options
    )
    weighted_outputs(means2d, depths, conics, compensations)
  end

  def weighted_projection_with_covariance(covariance)
    _, means2d, depths, conics, compensations = Gsplat.fully_fused_projection(
      @means,
      covars: covariance,
      **projection_options
    )
    weighted_outputs(means2d, depths, conics, compensations)
  end

  def weighted_outputs(means2d, depths, conics, compensations)
    (means2d * @weights[0]).sum + (depths * @weights[1]).sum +
      (conics * @weights[2]).sum + (compensations * @weights[3]).sum
  end

  def central_difference(input, epsilon: 1e-6)
    gradient = Numo::DFloat.zeros(*input.shape)
    input.size.times do |index|
      positive = input.flatten.dup
      negative = input.flatten.dup
      positive[index] += epsilon
      negative[index] -= epsilon
      gradient[index] = (
        yield(positive.reshape(*input.shape)) - yield(negative.reshape(*input.shape))
      ) / (2.0 * epsilon)
    end
    gradient
  end
end
