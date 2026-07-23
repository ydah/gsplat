# frozen_string_literal: true

require "test_helper"

class QuatScaleToCovarPreciTest < Minitest::Test
  class PairLoss < Gsplat::Autograd::Function
    def self.forward(context, covariance, precision, covariance_weight:, precision_weight:)
      context.save(covariance_weight, precision_weight)
      covariance.class.cast((covariance * covariance_weight).sum + (precision * precision_weight).sum)
    end

    def self.backward(context, _grad_output)
      context.saved_values
    end
  end

  def setup
    @previous_backend = Gsplat.backend
    Gsplat.backend = :ruby
  end

  def teardown
    Gsplat.backend = @previous_backend
  end

  def test_identity_rotation_produces_diagonal_covariance_and_precision
    quaternions = Numo::SFloat[[1, 0, 0, 0]]
    scales = Numo::SFloat[[2, 3, 4]]

    covariance, precision = Gsplat.quat_scale_to_covar_preci(quaternions, scales)

    assert_allclose covariance[0, true, true], Numo::SFloat[[4, 0, 0], [0, 9, 0], [0, 0, 16]], atol: 1e-6, rtol: 0.0
    assert_allclose(
      precision[0, true, true],
      Numo::SFloat[[0.25, 0, 0], [0, 1.0 / 9.0, 0], [0, 0, 1.0 / 16.0]],
      atol: 1e-6,
      rtol: 0.0
    )
  end

  def test_triu_compresses_in_upstream_order
    quaternions = Numo::SFloat[[1, 0, 0, 0]]
    scales = Numo::SFloat[[2, 3, 4]]

    covariance, = Gsplat.quat_scale_to_covar_preci(quaternions, scales, compute_preci: false, triu: true)

    assert_equal [1, 6], covariance.shape
    assert_allclose covariance, Numo::SFloat[[4, 0, 0, 9, 0, 16]], atol: 0.0, rtol: 0.0
  end

  def test_output_selection_retains_tuple_positions
    quaternions = Numo::SFloat[[1, 0, 0, 0]]
    scales = Numo::SFloat[[1, 1, 1]]

    covariance, precision = Gsplat.quat_scale_to_covar_preci(
      quaternions,
      scales,
      compute_covar: false,
      compute_preci: true
    )

    assert_nil covariance
    assert_instance_of Numo::SFloat, precision
  end

  def test_backward_matches_float64_central_difference
    quaternions = Numo::DFloat[[0.8, -0.2, 0.3, 0.5]]
    scales = Numo::DFloat[[0.7, 1.2, 0.9]]
    covariance_weight = Numo::DFloat[[[0.2, -0.1, 0.4], [0.3, 0.7, -0.2], [0.8, 0.1, -0.5]]]
    precision_weight = Numo::DFloat[[[-0.2, 0.5, 0.1], [0.4, -0.3, 0.6], [0.2, -0.7, 0.9]]]
    quaternion_var = Gsplat::Autograd::Variable.new(quaternions, requires_grad: true)
    scale_var = Gsplat::Autograd::Variable.new(scales, requires_grad: true)
    covariance, precision = Gsplat.quat_scale_to_covar_preci(quaternion_var, scale_var)
    loss = PairLoss.apply(
      covariance,
      precision,
      covariance_weight: covariance_weight,
      precision_weight: precision_weight
    )

    loss.backward

    numeric_quaternions = central_difference(quaternions) do |value|
      weighted_value(value, scales, covariance_weight, precision_weight)
    end
    numeric_scales = central_difference(scales) do |value|
      weighted_value(quaternions, value, covariance_weight, precision_weight)
    end
    assert_allclose quaternion_var.grad, numeric_quaternions, atol: 1e-6, rtol: 1e-5
    assert_allclose scale_var.grad, numeric_scales, atol: 1e-6, rtol: 1e-5
  end

  def test_matches_python_golden_data
    fixture = golden("quat_covar_full")
    covariance, precision = Gsplat.quat_scale_to_covar_preci(fixture.fetch("quats"), fixture.fetch("scales"))

    assert_allclose covariance, fixture.fetch("covars"), atol: 1e-5, rtol: 1e-5
    assert_allclose precision, fixture.fetch("precis"), atol: 1e-5, rtol: 1e-5
  end

  private

  def weighted_value(quaternions, scales, covariance_weight, precision_weight)
    covariance, precision = Gsplat.quat_scale_to_covar_preci(quaternions, scales)
    (covariance * covariance_weight).sum + (precision * precision_weight).sum
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
