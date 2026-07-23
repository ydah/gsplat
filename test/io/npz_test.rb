# frozen_string_literal: true

require "stringio"
require "test_helper"

class NpzTest < Minitest::Test
  def test_round_trips_deflated_arrays
    arrays = {
      "means" => Numo::SFloat[[1, 2, 3], [4, 5, 6]],
      opacity: Numo::DFloat[0.25, 0.75]
    }
    io = StringIO.new("".b)

    Gsplat::IO::Npy.write_npz(io, arrays)
    io.rewind
    output = Gsplat::IO::Npy.read_npz(io)

    assert_equal %w[means opacity], output.keys.sort
    assert_equal arrays["means"].to_a, output["means"].to_a
    assert_equal arrays[:opacity].to_a, output["opacity"].to_a
  end

  def test_round_trips_stored_arrays
    io = StringIO.new("".b)
    input = Numo::Int32[[1, 2], [3, 4]]

    Gsplat::IO::Npy.write_npz(io, { values: input }, compression: :stored)
    io.rewind
    output = Gsplat::IO::Npy.read_npz(io)

    assert_equal input.to_a, output.fetch("values").to_a
  end

  def test_rejects_unknown_compression
    error = assert_raises(ArgumentError) do
      Gsplat::IO::Npy.write_npz(StringIO.new("".b), {}, compression: :gzip)
    end

    assert_match(/compression/, error.message)
  end
end
