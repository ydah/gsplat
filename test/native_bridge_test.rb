# frozen_string_literal: true

require "test_helper"

class NativeBridgeTest < Minitest::Test
  def test_native_add_uses_numo_storage
    skip "run bundle exec rake compile to build the native extension" unless Gsplat::Native.available?

    left = Numo::SFloat[[1, 2], [3, 4]]
    right = Numo::SFloat[[0.5, 1.5], [2.5, 3.5]]

    assert_allclose Gsplat::Native.add(left, right), left + right, atol: 0.0, rtol: 0.0
  end

  def test_unbuilt_extension_keeps_ruby_backend_available
    assert_includes [true, false], Gsplat::Native.available?
    assert Gsplat::Backend
  end
end
