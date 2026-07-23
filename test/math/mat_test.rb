# frozen_string_literal: true

require "test_helper"

class MatTest < Minitest::Test
  def test_inv2x2_and_determinant
    matrices = Numo::DFloat[[[4, 7], [2, 6]], [[3, 1], [5, 2]]]
    inverses = Gsplat::Math::Mat.inv2x2(matrices)
    products = Gsplat::Math::Mat.matmul_batch(inverses, matrices)
    identity = Numo::DFloat.eye(2).reshape(1, 2, 2).tile(2, 1, 1)

    assert_allclose products, identity, atol: 1e-12, rtol: 1e-12
    assert_allclose Gsplat::Math::Mat.det2x2(matrices), Numo::DFloat[10, 1], atol: 0.0, rtol: 0.0
  end

  def test_eigvals2x2_returns_ordered_closed_form_values
    matrices = Numo::DFloat[[[3, 1], [1, 3]], [[2, 0], [0, 5]]]

    eigenvalues = Gsplat::Math::Mat.eigvals2x2(matrices)

    assert_allclose eigenvalues, Numo::DFloat[[2, 4], [2, 5]], atol: 1e-12, rtol: 0.0
  end

  def test_inv3x3_and_determinant
    matrices = Numo::DFloat[[[1, 2, 3], [0, 1, 4], [5, 6, 0]]]
    inverse = Gsplat::Math::Mat.inv3x3(matrices)
    product = Gsplat::Math::Mat.matmul_batch(inverse, matrices)

    assert_allclose product[0, true, true], Numo::DFloat.eye(3), atol: 1e-12, rtol: 1e-12
    assert_allclose Gsplat::Math::Mat.det3x3(matrices), Numo::DFloat[1], atol: 1e-12, rtol: 0.0
  end

  def test_matmul_batch_multiplies_rectangular_small_matrices
    left = Numo::SFloat[[[1, 2, 3], [4, 5, 6]]]
    right = Numo::SFloat[[[7, 8], [9, 10], [11, 12]]]

    result = Gsplat::Math::Mat.matmul_batch(left, right)

    assert_equal [[[58.0, 64.0], [139.0, 154.0]]], result.to_a
  end

  def test_dtype_option_promotes_calculation
    matrix = Numo::SFloat[[[2, 0], [0, 4]]]

    inverse = Gsplat::Math::Mat.inv2x2(matrix, dtype: Numo::DFloat)

    assert_instance_of Numo::DFloat, inverse
  end

  def test_singular_inverse_fails_fast
    matrix = Numo::DFloat[[[1, 2], [2, 4]]]

    error = assert_raises(Gsplat::Error) { Gsplat::Math::Mat.inv2x2(matrix) }

    assert_match(/singular/, error.message)
  end
end
