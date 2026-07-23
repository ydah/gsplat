# frozen_string_literal: true

require "stringio"
require "test_helper"

class PlyTest < Minitest::Test
  def test_binary_round_trip_preserves_inria_parameters
    output = StringIO.new("".b)

    Gsplat::IO::Ply.write(output, params)
    restored = Gsplat::IO::Ply.read(StringIO.new(output.string))

    params.each do |name, expected|
      assert_allclose restored.fetch(name), expected, atol: 1e-6, rtol: 1e-6
    end
  end

  def test_writer_emits_canonical_inria_header
    output = StringIO.new("".b)

    Gsplat::IO::Ply.write(output, params)
    header = output.string.split("end_header\n", 2).first

    assert_includes header, "format binary_little_endian 1.0"
    assert_includes header, "element vertex 2"
    expected_order = %w[x y z nx ny nz f_dc_0 f_dc_1 f_dc_2
                        f_rest_0 f_rest_1 f_rest_2 opacity
                        scale_0 scale_1 scale_2 rot_0 rot_1 rot_2 rot_3]
    actual_order = header.lines.grep(/\Aproperty/).map { |line| line.split.last }
    assert_equal expected_order, actual_order
  end

  def test_reads_ascii_with_arbitrary_property_order
    path = File.expand_path("../fixtures/inria_ascii.ply", __dir__)

    restored = Gsplat::IO::Ply.read(path)

    assert_allclose restored[:means], Numo::SFloat[[1, 2, 3], [-1, -2, 4]],
                    atol: 0.0, rtol: 0.0
    assert_allclose restored[:quats], Numo::SFloat[[1, 0, 0, 0], [0.5, 0.5, 0.5, 0.5]],
                    atol: 0.0, rtol: 0.0
    assert_allclose restored[:sh0][0, 0, true], Numo::SFloat[0.1, 0.2, 0.3],
                    atol: 1e-7, rtol: 0.0
    assert_allclose restored[:shN][1, 0, true], Numo::SFloat[0.7, 0.8, 0.9],
                    atol: 1e-7, rtol: 0.0
  end

  private

  def params
    {
      means: Numo::SFloat[[1, 2, 3], [-1, -2, 4]],
      scales: Numo::SFloat[[0.1, 0.2, 0.3], [-0.1, -0.2, -0.3]],
      quats: Numo::SFloat[[1, 0, 0, 0], [0.5, 0.5, 0.5, 0.5]],
      opacities: Numo::SFloat[-2, 1],
      sh0: Numo::SFloat[[[0.1, 0.2, 0.3]], [[0.4, 0.5, 0.6]]],
      shN: Numo::SFloat[[[0.01, 0.02, 0.03]], [[0.7, 0.8, 0.9]]]
    }
  end
end
