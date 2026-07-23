# frozen_string_literal: true

require "test_helper"

class SphericalHarmonicsTest < Minitest::Test
  C0 = 0.2820947917738781
  C1 = 0.48860251190292

  class WeightedSum < Gsplat::Autograd::Function
    def self.forward(context, colors, weight:)
      context.save(weight)
      colors.class.cast((colors * weight).sum)
    end

    def self.backward(context, grad_output)
      [context.saved_values.fetch(0) * grad_output.to_f]
    end
  end

  def setup
    @previous_backend = Gsplat.backend
    Gsplat.backend = :ruby
  end

  def teardown
    Gsplat.backend = @previous_backend
  end

  def test_degree_zero_is_constant_basis_times_dc_coefficient
    directions = Numo::SFloat[[1, 0, 0], [0, 1, 0]]
    coefficients = Numo::SFloat.zeros(2, 1, 2)
    coefficients[true, 0, true] = Numo::SFloat[[2, 3], [4, 5]]

    colors = Gsplat.spherical_harmonics(0, directions, coefficients)

    assert_allclose colors, Numo::SFloat[[2 * C0, 3 * C0], [4 * C0, 5 * C0]], atol: 1e-6, rtol: 1e-6
  end

  def test_degree_one_uses_upstream_basis_order
    directions = Numo::DFloat[[1, 0, 0]]
    coefficients = Numo::DFloat.ones(1, 4, 1)

    color = Gsplat.spherical_harmonics(1, directions, coefficients)

    assert_allclose color, Numo::DFloat[[C0 - C1]], atol: 1e-12, rtol: 1e-12
  end

  def test_masks_zero_output
    directions = Numo::SFloat[[1, 0, 0], [0, 1, 0]]
    coefficients = Numo::SFloat.ones(2, 4, 3)

    colors = Gsplat.spherical_harmonics(1, directions, coefficients, masks: Numo::Bit[1, 0])

    assert_allclose colors[1, true], Numo::SFloat.zeros(3), atol: 0.0, rtol: 0.0
  end

  def test_degree_four_backward_matches_float64_central_difference
    directions = Numo::DFloat[[0.8, -0.2, 0.5]]
    coefficients = Numo::DFloat.new(1, 25, 2).rand(-0.5, 0.5)
    weight = Numo::DFloat[[0.3, -0.7]]
    direction_var = Gsplat::Autograd::Variable.new(directions, requires_grad: true)
    coefficient_var = Gsplat::Autograd::Variable.new(coefficients, requires_grad: true)
    colors = Gsplat.spherical_harmonics(4, direction_var, coefficient_var)

    WeightedSum.apply(colors, weight: weight).backward

    numeric_directions = central_difference(directions) do |value|
      (Gsplat.spherical_harmonics(4, value, coefficients) * weight).sum
    end
    numeric_coefficients = central_difference(coefficients) do |value|
      (Gsplat.spherical_harmonics(4, directions, value) * weight).sum
    end
    assert_allclose direction_var.grad, numeric_directions, atol: 1e-6, rtol: 1e-5
    assert_allclose coefficient_var.grad, numeric_coefficients, atol: 1e-7, rtol: 1e-6
  end

  def test_matches_python_golden_data
    fixture = golden("sh_deg3")
    colors = Gsplat.spherical_harmonics(3, fixture.fetch("dirs"), fixture.fetch("coeffs"))

    assert_allclose colors, fixture.fetch("colors"), atol: 1e-5, rtol: 1e-5
  end

  private

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
