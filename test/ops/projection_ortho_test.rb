# frozen_string_literal: true

require "test_helper"

class ProjectionOrthoTest < Minitest::Test
  class OrthoLoss < Gsplat::Autograd::Function
    # rubocop:disable Metrics/ParameterLists
    def self.forward(context, means2d, depths, conics, compensations, weights:)
      # rubocop:enable Metrics/ParameterLists
      context.save(*weights)
      value = (means2d * weights[0]).sum + (depths * weights[1]).sum +
              (conics * weights[2]).sum + (compensations * weights[3]).sum
      means2d.class.cast(value)
    end

    def self.backward(context, grad_output)
      context.saved_values.map { |weight| weight * grad_output.to_f }
    end
  end

  def setup
    @previous_backend = Gsplat.backend
    Gsplat.backend = :ruby
    @means = Numo::DFloat[[0.1, -0.2, 2]]
    @quaternions = Numo::DFloat[[1, 0, 0, 0]]
    @scales = Numo::DFloat[[0.1, 0.1, 0.1]]
    @viewmats = Numo::DFloat.zeros(1, 4, 4)
    @viewmats[0, true, true] = Numo::DFloat.eye(4)
    @intrinsics = Numo::DFloat[[[100, 0, 50], [0, 100, 40], [0, 0, 1]]]
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

  def test_ortho_projection_matches_hand_calculation
    covariance = Numo::DFloat[[[0.01, 0, 0], [0, 0.01, 0], [0, 0, 0.01]]]

    radii, means2d, depths, conics, compensations = Gsplat.fully_fused_projection(
      @means,
      covars: covariance,
      **projection_options
    )

    assert_equal [34, 34], radii[0, 0, true].to_a
    assert_allclose means2d, Numo::DFloat[[[60, 20]]], atol: 1e-12, rtol: 0.0
    assert_allclose depths, Numo::DFloat[[2]], atol: 1e-12, rtol: 0.0
    assert_allclose conics, Numo::DFloat[[[1.0 / 100.3, 0, 1.0 / 100.3]]], atol: 1e-12, rtol: 0.0
    assert_allclose compensations, Numo::DFloat[[100.0 / 100.3]], atol: 1e-12, rtol: 0.0
  end

  def test_ortho_backward_matches_float64_central_difference
    mean_var = Gsplat::Autograd::Variable.new(@means, requires_grad: true)
    quaternion_var = Gsplat::Autograd::Variable.new(@quaternions, requires_grad: true)
    scale_var = Gsplat::Autograd::Variable.new(@scales, requires_grad: true)
    _, means2d, depths, conics, compensations = Gsplat.fully_fused_projection(
      mean_var,
      quats: quaternion_var,
      scales: scale_var,
      **projection_options
    )

    OrthoLoss.apply(means2d, depths, conics, compensations, weights: @weights).backward

    assert_allclose mean_var.grad,
                    central_difference(@means) { |value| weighted_projection(value, @quaternions, @scales) },
                    atol: 2e-6, rtol: 2e-5
    assert_allclose quaternion_var.grad,
                    central_difference(@quaternions) { |value| weighted_projection(@means, value, @scales) },
                    atol: 2e-6, rtol: 2e-5
    assert_allclose scale_var.grad,
                    central_difference(@scales) { |value| weighted_projection(@means, @quaternions, value) },
                    atol: 2e-6, rtol: 2e-5
  end

  def test_matches_python_ortho_golden_data
    fixture = golden("proj_ortho_c1_n1000")
    radii, means2d, depths, conics, compensations = Gsplat.fully_fused_projection(
      fixture.fetch("means"),
      quats: fixture.fetch("quats"),
      scales: fixture.fetch("scales"),
      viewmats: fixture.fetch("viewmats"),
      ks: fixture.fetch("ks"),
      width: fixture.fetch("width").to_i,
      height: fixture.fetch("height").to_i,
      calc_compensations: true,
      camera_model: "ortho"
    )
    assert_equal fixture.fetch("radii").to_a, radii.to_a
    assert_allclose means2d, fixture.fetch("means2d"), atol: 1e-4, rtol: 1e-5
    assert_allclose depths, fixture.fetch("depths"), atol: 1e-5, rtol: 1e-5
    assert_allclose conics, fixture.fetch("conics"), atol: 1e-4, rtol: 1e-4
    assert_allclose compensations, fixture.fetch("compensations"), atol: 1e-5, rtol: 1e-5
  end

  private

  def projection_options
    {
      viewmats: @viewmats,
      ks: @intrinsics,
      width: 100,
      height: 80,
      calc_compensations: true,
      camera_model: "ortho"
    }
  end

  def weighted_projection(means, quaternions, scales)
    _, means2d, depths, conics, compensations = Gsplat.fully_fused_projection(
      means,
      quats: quaternions,
      scales: scales,
      **projection_options
    )
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
