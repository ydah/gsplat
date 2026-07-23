# frozen_string_literal: true

require "test_helper"

class LossesTest < Minitest::Test
  def test_identical_images_have_unit_ssim_and_infinite_psnr
    image = Numo::DFloat.new(1, 3, 8, 9).rand

    assert_in_delta 1.0, Gsplat::Training::Losses.ssim(image, image, layout: :nchw), 1e-12
    assert_equal Float::INFINITY, Gsplat::Training::Losses.psnr(image, image)
  end

  def test_l1_has_expected_value_and_gradients
    prediction = variable(Numo::DFloat[1, -2, 3])
    target = variable(Numo::DFloat[0, -2, 5])

    loss = Gsplat::Training::Losses.l1(prediction, target)
    loss.backward

    assert_in_delta 1.0, loss.data.to_f, 1e-12
    assert_allclose prediction.grad, Numo::DFloat[1.0 / 3, 0, -1.0 / 3], atol: 1e-12, rtol: 0.0
    assert_allclose target.grad, -prediction.grad, atol: 1e-12, rtol: 0.0
  end

  def test_ssim_gradients_match_central_differences
    image_a = Numo::DFloat.new(1, 1, 5, 6).rand
    image_b = Numo::DFloat.new(1, 1, 5, 6).rand
    variable_a = variable(image_a.dup)
    variable_b = variable(image_b.dup)
    Gsplat::Training::Losses.ssim(variable_a, variable_b, layout: :nchw).backward

    [[0, 0, 2, 3], [0, 0, 0, 0]].each do |index|
      expected_a = central_difference(image_a, image_b, index, :first)
      expected_b = central_difference(image_a, image_b, index, :second)
      assert_in_delta expected_a, variable_a.grad[*index], 1e-5
      assert_in_delta expected_b, variable_b.grad[*index], 1e-5
    end
  end

  def test_ssim_matches_golden_data
    fixture = golden("ssim_rgb")
    image_a = variable(fixture.fetch("image_a"))
    image_b = variable(fixture.fetch("image_b"))

    score = Gsplat::Training::Losses.ssim(image_a, image_b, layout: :nchw)
    score.backward

    assert_in_delta fixture.fetch("ssim").to_f, score.data.to_f, 1e-4
    assert_allclose image_a.grad, fixture.fetch("grad_image_a"), atol: 1e-3, rtol: 1e-3
    assert_allclose image_b.grad, fixture.fetch("grad_image_b"), atol: 1e-3, rtol: 1e-3
  end

  private

  def variable(data)
    Gsplat::Autograd::Variable.new(data, requires_grad: true)
  end

  def central_difference(first, second, index, selected)
    epsilon = 1e-6
    array = selected == :first ? first : second
    original = array[*index]
    array[*index] = original + epsilon
    plus = Gsplat::Training::Losses.ssim(first, second, layout: :nchw)
    array[*index] = original - epsilon
    minus = Gsplat::Training::Losses.ssim(first, second, layout: :nchw)
    array[*index] = original
    (plus - minus) / (2 * epsilon)
  end
end
