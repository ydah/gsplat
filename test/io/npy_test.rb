# frozen_string_literal: true

require "stringio"
require "test_helper"

class NpyTest < Minitest::Test
  DTYPES = {
    Numo::SFloat => [[1.25, -2.5], [3.75, 4.0]],
    Numo::DFloat => [[1.25, -2.5], [3.75, 4.0]],
    Numo::Int32 => [[1, -2], [3, 4]],
    Numo::Int64 => [[1, -2], [3, 4]],
    Numo::Bit => [[1, 0], [0, 1]]
  }.freeze
  FIXTURE_NAMES = {
    Numo::SFloat => "float4",
    Numo::DFloat => "float8",
    Numo::Int32 => "int4",
    Numo::Int64 => "int8",
    Numo::Bit => "b1"
  }.freeze

  def test_round_trips_supported_dtypes_and_shapes
    DTYPES.each do |type, values|
      input = type.cast(values)
      io = StringIO.new("".b)

      Gsplat::IO::Npy.write(io, input)
      io.rewind
      output = Gsplat::IO::Npy.read(io)

      assert_instance_of type, output
      assert_equal input.shape, output.shape
      assert_equal input.to_a, output.to_a
    end
  end

  def test_round_trips_a_scalar
    input = Numo::DFloat.cast(3.5)
    io = StringIO.new("".b)

    Gsplat::IO::Npy.write(io, input)
    io.rewind
    output = Gsplat::IO::Npy.read(io)

    assert_empty output.shape
    assert_in_delta 3.5, output.to_f
  end

  def test_rejects_fortran_order
    valid = StringIO.new("".b)
    Gsplat::IO::Npy.write(valid, Numo::SFloat[1, 2])
    invalid = valid.string.sub("False", "True ")

    error = assert_raises(Gsplat::NotSupportedError) do
      Gsplat::IO::Npy.read(StringIO.new(invalid))
    end
    assert_match(/Fortran order/, error.message)
  end

  def test_reads_numpy_generated_fixtures
    fixture_dir = File.expand_path("../fixtures/npy", __dir__)
    skip "run tools/make_npy_fixtures.py" unless Dir.exist?(fixture_dir)

    FIXTURE_NAMES.each do |type, name|
      output = Gsplat::IO::Npy.read(File.join(fixture_dir, "#{name}.npy"))

      assert_instance_of type, output
      assert_equal [2, 3], output.shape
    end
  end
end
