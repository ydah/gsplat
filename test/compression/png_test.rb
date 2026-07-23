# frozen_string_literal: true

require "tmpdir"
require "test_helper"

class CompressionPngTest < Minitest::Test
  def test_dependency_free_png_codec_round_trips_one_to_four_channels
    Dir.mktmpdir do |directory|
      (1..4).each do |channels|
        pixels = Numo::UInt8.new(3, 5, channels).seq
        path = File.join(directory, "channels_#{channels}.png")

        Gsplat::Compression::PngCodec.write(path, pixels)

        assert_equal pixels.to_a, Gsplat::Compression::PngCodec.read(path).to_a
      end
    end
  end

  # rubocop:disable Metrics/AbcSize
  def test_compression_round_trip_writes_compatible_layout_and_preserves_render
    Dir.mktmpdir do |directory|
      parameters = splats
      compressor = Gsplat::Compression::Png.new(kmeans_clusters: 4)
      compressor.compress(directory, parameters)
      restored = compressor.decompress(directory)

      expected_files = %w[
        extra.npz means_l.png means_u.png meta.json opacities.png quats.png
        scales.png sh0.png shN.npz
      ]
      assert_empty expected_files - Dir.children(directory)
      assert_equal parameters.keys.sort, restored.keys.sort
      assert_equal 16, restored.fetch(:means).shape[0]
      assert_allclose restored.fetch(:means).min(axis: 0), parameters.fetch(:means).min(axis: 0),
                      atol: 2e-5, rtol: 0.0
      assert_allclose restored.fetch(:extra).sort(axis: 0), parameters.fetch(:extra).sort(axis: 0),
                      atol: 0.0, rtol: 0.0
      expected_sh = parameters.fetch(:shN).reshape(16, 45).sort(axis: 0)
      actual_sh = restored.fetch(:shN).reshape(16, 45).sort(axis: 0)
      assert_allclose actual_sh, expected_sh, atol: 0.006, rtol: 0.0

      original_image = render(parameters)
      restored_image = render(restored)
      assert_operator Gsplat::Training::Losses.psnr(restored_image, original_image), :>, 35.0
    end
  end
  # rubocop:enable Metrics/AbcSize

  private

  # rubocop:disable Metrics/AbcSize
  def splats
    means = Numo::SFloat.cast(
      16.times.map do |index|
        [((index % 4) - 1.5) * 0.12, ((index / 4) - 1.5) * 0.12, 2 + (index * 0.01)]
      end
    )
    quaternions = Numo::SFloat.zeros(16, 4)
    quaternions[true, 0] = 1
    rgb = Numo::SFloat.cast(
      16.times.map { |index| [0.2 + (index * 0.02), 0.5, 0.8 - (index * 0.02)] }
    )
    spherical_rest = Numo::SFloat.zeros(16, 15, 3)
    16.times { |index| spherical_rest[index, true, true] = (index % 4) * 0.1 }
    {
      means: means,
      scales: Numo::SFloat.ones(16, 3) * ::Math.log(0.08),
      quats: quaternions,
      opacities: Numo::SFloat.ones(16) * 0.2,
      sh0: Gsplat::Utils.rgb_to_sh(rgb).reshape(16, 1, 3),
      shN: spherical_rest,
      extra: Numo::SFloat.new(16, 2).seq
    }
  end
  # rubocop:enable Metrics/AbcSize

  def render(parameters)
    viewmats = Numo::SFloat.eye(4).reshape(1, 4, 4)
    intrinsics = Numo::SFloat[[[20, 0, 8], [0, 20, 8], [0, 0, 1]]]
    image, = Gsplat.rasterization(
      means: parameters.fetch(:means),
      quats: parameters.fetch(:quats),
      scales: Numo::NMath.exp(parameters.fetch(:scales)),
      opacities: 1.0 / (1.0 + Numo::NMath.exp(-parameters.fetch(:opacities))),
      colors: parameters.fetch(:sh0),
      sh_degree: 0,
      viewmats: viewmats,
      ks: intrinsics,
      width: 16,
      height: 16
    )
    image
  end
end
