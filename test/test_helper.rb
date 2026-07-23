# frozen_string_literal: true

require "minitest/autorun"
require "gsplat"

module GsplatTestHelpers
  GOLDEN_DIR = File.expand_path("golden", __dir__)

  def assert_allclose(actual, expected, atol:, rtol:, mismatch_ratio: 0.0)
    assert_equal expected.shape, actual.shape, "array shapes differ"
    raise ArgumentError, "mismatch_ratio must be between 0.0 and 1.0" unless mismatch_ratio.between?(0.0, 1.0)

    actual_values = flattened_values(actual)
    expected_values = flattened_values(expected)
    mismatches = actual_values.zip(expected_values).filter_map.with_index do |(actual_value, expected_value), index|
      next if close_value?(actual_value, expected_value, atol, rtol)

      [index, actual_value, expected_value, (actual_value - expected_value).abs]
    end
    actual_ratio = actual_values.empty? ? 0.0 : mismatches.length.fdiv(actual_values.length)
    return if actual_ratio <= mismatch_ratio

    worst = mismatches.max_by(&:last)
    flunk(
      "#{mismatches.length} mismatch(es) of #{actual_values.length} " \
      "(ratio #{actual_ratio}); worst at flat index #{worst[0]}: " \
      "actual=#{worst[1]}, expected=#{worst[2]}, abs_error=#{worst[3]}"
    )
  end

  def golden(name)
    raise ArgumentError, "invalid golden fixture name #{name.inspect}" unless name.match?(/\A[\w.-]+\z/)

    path = File.join(GOLDEN_DIR, "#{name}.npz")
    skip "golden fixture #{name} is unavailable; see tools/README.md" unless File.file?(path)

    Gsplat::IO::Npy.read_npz(path)
  end

  def with_backend(backend)
    previous = Gsplat.backend
    Gsplat.backend = backend
    yield
  ensure
    Gsplat.backend = previous
  end

  private

  def flattened_values(array)
    values = array.to_a
    values.is_a?(Array) ? values.flatten : [values]
  end

  def close_value?(actual, expected, atol, rtol)
    return true if actual == expected

    (actual - expected).abs <= atol + (rtol * expected.abs)
  end
end

Minitest::Test.include(GsplatTestHelpers)
