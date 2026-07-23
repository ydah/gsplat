# frozen_string_literal: true

require "test_helper"

class AccumulateTest < Minitest::Test
  def setup
    @previous_backend = Gsplat.backend
    Gsplat.backend = :ruby
  end

  def teardown
    Gsplat.backend = @previous_backend
  end

  def test_single_gaussian_at_pixel_center
    colors, alphas = Gsplat.accumulate(
      Numo::DFloat[[[0.5, 0.5]]],
      Numo::DFloat[[[1, 0, 1]]],
      Numo::DFloat[[0.8]],
      Numo::DFloat[[[0.2, 0.4, 0.6]]],
      width: 1,
      height: 1
    )

    assert_allclose colors, Numo::DFloat[[[[0.16, 0.32, 0.48]]]], atol: 1e-12, rtol: 0.0
    assert_allclose alphas, Numo::DFloat[[[[0.8]]]], atol: 1e-12, rtol: 0.0
  end

  def test_front_to_back_compositing_and_background
    colors, alphas = Gsplat.accumulate(
      Numo::DFloat[[[0.5, 0.5], [0.5, 0.5]]],
      Numo::DFloat[[[1, 0, 1], [1, 0, 1]]],
      Numo::DFloat[[0.5, 0.5]],
      Numo::DFloat[[[1, 0], [0, 1]]],
      width: 1,
      height: 1,
      backgrounds: Numo::DFloat[[0.2, 0.4]]
    )

    assert_allclose colors, Numo::DFloat[[[[0.55, 0.35]]]], atol: 1e-12, rtol: 0.0
    assert_allclose alphas, Numo::DFloat[[[[0.75]]]], atol: 1e-12, rtol: 0.0
  end

  def test_alpha_clamp_and_skip_threshold
    colors, alphas = Gsplat.accumulate(
      Numo::SFloat[[[0.5, 0.5], [0.5, 0.5]]],
      Numo::SFloat[[[1, 0, 1], [1, 0, 1]]],
      Numo::SFloat[[2.0, 0.001]],
      Numo::SFloat[[[1], [100]]],
      width: 1,
      height: 1
    )

    assert_allclose colors, Numo::SFloat[[[[0.999]]]], atol: 1e-6, rtol: 0.0
    assert_allclose alphas, Numo::SFloat[[[[0.999]]]], atol: 1e-6, rtol: 0.0
  end
end
