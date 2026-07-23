# frozen_string_literal: true

require "tmpdir"
require "test_helper"

class ImageTest < Minitest::Test
  def test_each_available_backend_round_trips_png
    backends = Gsplat::IO::Image.available_backends
    skip "install ruby-vips or chunky_png to exercise image IO" if backends.empty?

    Dir.mktmpdir do |directory|
      backends.each do |backend|
        path = File.join(directory, "#{backend}.png")
        Gsplat::IO::Image.write(path, image, backend: backend)
        restored = Gsplat::IO::Image.read(path, backend: backend)

        assert_instance_of Numo::SFloat, restored
        assert_allclose restored, image, atol: (1.0 / 255) + 1e-7, rtol: 0.0
      end
    end
  end

  def test_auto_backend_prefers_vips_when_available
    available = Gsplat::IO::Image.available_backends
    skip "ruby-vips is unavailable" unless available.include?(:vips)

    assert_equal :vips, available.first
  end

  def test_rejects_non_rgb_array
    error = assert_raises(Gsplat::ShapeError) do
      Gsplat::IO::Image.write("unused.png", Numo::SFloat.zeros(3, 4))
    end

    assert_includes error.message, "[H,W,3]"
  end

  private

  def image
    Numo::SFloat[
      [[0, 0.25, 1], [1, 0.5, 0]],
      [[0.1, 0.2, 0.3], [0.7, 0.8, 0.9]]
    ]
  end
end
