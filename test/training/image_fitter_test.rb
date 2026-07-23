# frozen_string_literal: true

require "test_helper"

class ImageFitterTest < Minitest::Test
  def setup
    @previous_backend = Gsplat.backend
    Gsplat.backend = :ruby
  end

  def teardown
    Gsplat.backend = @previous_backend
  end

  def test_self_consistency_fit_improves_psnr
    result = Gsplat::Training::ImageFitter.new(
      width: 8,
      height: 8,
      n_gaussians: 64,
      learning_rate: 20.0,
      seed: 7
    ).fit(steps: 12)

    assert_operator result.final_psnr, :>, result.initial_psnr
    assert_operator result.final_psnr, :>, 20.0
    result.history.each_cons(2) do |previous, current|
      assert_operator current, :>=, previous - 1e-10
    end
  end

  def test_128_pixel_fixture_is_present
    fixture = File.binread(File.expand_path("../fixtures/fit_image.ppm", __dir__), 32)

    assert fixture.start_with?("P6\n128 128\n255\n")
  end
end
