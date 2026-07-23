# frozen_string_literal: true

require "test_helper"

class TestHelperTest < Minitest::Test
  def test_assert_allclose_accepts_absolute_and_relative_tolerance
    actual = Numo::SFloat[1.0001, 100.01]
    expected = Numo::SFloat[1.0, 100.0]

    assert_allclose(actual, expected, atol: 0.001, rtol: 0.001)
  end

  def test_assert_allclose_rejects_values_outside_tolerance
    error = assert_raises(Minitest::Assertion) do
      assert_allclose(Numo::SFloat[1.0, 2.0], Numo::SFloat[1.0, 3.0], atol: 0.01, rtol: 0.0)
    end

    assert_match(/1 mismatch/, error.message)
  end

  def test_assert_allclose_allows_a_bounded_mismatch_ratio
    actual = Numo::SFloat[1.0, 2.0, 30.0, 4.0]
    expected = Numo::SFloat[1.0, 2.0, 3.0, 4.0]

    assert_allclose(actual, expected, atol: 0.0, rtol: 0.0, mismatch_ratio: 0.25)
  end

  def test_with_backend_restores_the_previous_backend
    previous = Gsplat.backend

    with_backend(:ruby) { assert_equal :ruby, Gsplat.backend }

    assert_equal previous, Gsplat.backend
  end

  def test_golden_skips_when_fixture_is_unavailable
    assert_raises(Minitest::Skip) { golden("not-generated") }
  end
end
