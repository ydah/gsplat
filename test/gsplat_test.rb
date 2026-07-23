# frozen_string_literal: true

require "test_helper"

class GsplatTest < Minitest::Test
  def test_has_a_version
    refute_nil Gsplat::VERSION
  end

  def test_error_hierarchy
    assert_operator Gsplat::ShapeError, :<, Gsplat::Error
    assert_operator Gsplat::NotSupportedError, :<, Gsplat::Error
  end

  def test_exposes_a_logger
    assert_respond_to Gsplat.logger, :warn
  end

  def test_exposes_a_replaceable_random_number_generator
    original_rng = Gsplat.rng
    Gsplat.rng = Random.new(42)

    assert_equal Random.new(42).rand, Gsplat.rng.rand
  ensure
    Gsplat.rng = original_rng
  end
end
