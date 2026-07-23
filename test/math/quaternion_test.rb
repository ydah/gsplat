# frozen_string_literal: true

require "test_helper"

class QuaternionTest < Minitest::Test
  def test_normalize_produces_unit_quaternions
    quaternions = Numo::SFloat[[2, 0, 0, 0], [1, 2, 3, 4]]

    normalized = Gsplat::Math::Quaternion.normalize(quaternions)
    norms = (normalized**2).sum(axis: 1)**0.5

    assert_allclose norms, Numo::SFloat.ones(2), atol: 1e-6, rtol: 1e-6
  end

  def test_rotation_matrices_are_orthonormal_with_unit_determinant
    quaternions = Numo::DFloat[[1, 2, 3, 4], [0.5, -1, 0.25, 2]]
    rotations = Gsplat::Math::Quaternion.to_rotmat(quaternions)
    transposed = rotations.transpose(0, 2, 1)
    products = Gsplat::Math::Mat.matmul_batch(rotations, transposed)
    identity = Numo::DFloat.eye(3).reshape(1, 3, 3).tile(2, 1, 1)

    assert_allclose products, identity, atol: 1e-12, rtol: 1e-12
    assert_allclose Gsplat::Math::Mat.det3x3(rotations), Numo::DFloat.ones(2), atol: 1e-12, rtol: 1e-12
  end

  def test_identity_quaternion_maps_to_identity_matrix
    rotation = Gsplat::Math::Quaternion.to_rotmat(Numo::SFloat[[1, 0, 0, 0]])

    assert_allclose rotation[0, true, true], Numo::SFloat.eye(3), atol: 0.0, rtol: 0.0
  end

  def test_to_rotmat_vjp_matches_central_difference
    quaternion = Numo::DFloat[[0.8, -0.2, 0.3, 0.5]]
    weight = Numo::DFloat[[[0.2, -0.5, 0.7], [1.1, -0.3, 0.4], [-0.8, 0.6, 0.9]]]
    analytic = Gsplat::Math::Quaternion.to_rotmat_vjp(quaternion, weight)
    numeric = central_difference(quaternion) do |value|
      (Gsplat::Math::Quaternion.to_rotmat(value) * weight).sum
    end

    assert_allclose analytic, numeric, atol: 1e-6, rtol: 1e-5
  end

  def test_normalize_vjp_matches_central_difference
    quaternion = Numo::DFloat[[0.8, -0.2, 0.3, 0.5]]
    weight = Numo::DFloat[[0.4, -0.7, 0.2, 1.1]]
    analytic = Gsplat::Math::Quaternion.normalize_vjp(quaternion, weight)
    numeric = central_difference(quaternion) do |value|
      (Gsplat::Math::Quaternion.normalize(value) * weight).sum
    end

    assert_allclose analytic, numeric, atol: 1e-7, rtol: 1e-6
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
