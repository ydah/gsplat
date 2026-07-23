# frozen_string_literal: true

require "tmpdir"
require "test_helper"

class ColmapTest < Minitest::Test
  FIXTURE_ROOT = File.expand_path("../fixtures/colmap", __dir__)

  def test_reads_complete_binary_sparse_model
    dataset = Gsplat::IO::Colmap.read(File.join(FIXTURE_ROOT, "binary"), data_factor: 2)

    assert_equal 3, dataset.cameras.length
    assert_equal 3, dataset.images.length
    assert_equal 100, dataset.points3d.length
    camera = dataset.cameras.fetch(3)
    assert_equal "OPENCV", camera.model
    assert_equal [320, 240], [camera.width, camera.height]
    assert_allclose camera.intrinsics, Numo::DFloat[[260, 0, 160], [0, 262.5, 120], [0, 0, 1]],
                    atol: 0.0, rtol: 0.0
    assert_allclose camera.distortion, Numo::DFloat[0.01, -0.02, 0.001, -0.002],
                    atol: 0.0, rtol: 0.0
  end

  def test_text_and_binary_models_are_numerically_identical
    binary = Gsplat::IO::Colmap.read(File.join(FIXTURE_ROOT, "binary"))
    text = Gsplat::IO::Colmap.read(File.join(FIXTURE_ROOT, "text"))

    compare_cameras(binary.cameras, text.cameras)
    compare_images(binary.images, text.images)
    compare_points(binary.points3d, text.points3d)
  end

  def test_builds_colmap_world_to_camera_matrix
    image = Gsplat::IO::Colmap.read_images(
      File.join(FIXTURE_ROOT, "binary", "images.bin")
    ).fetch(2)
    expected_rotation = Numo::DFloat[[0, 0, 1], [0, 1, 0], [-1, 0, 0]]

    assert_allclose image.rotation, expected_rotation, atol: 1e-12, rtol: 0.0
    assert_allclose image.world_to_camera[0...3, 3], Numo::DFloat[1, 2, 3],
                    atol: 0.0, rtol: 0.0
    assert_equal [100, 2], image.points2d.shape
    assert_equal 1000, image.point3d_ids[0]
  end

  def test_resolves_dataset_sparse_zero_directory
    Dir.mktmpdir do |directory|
      sparse = File.join(directory, "sparse")
      Dir.mkdir(sparse)
      File.symlink(File.join(FIXTURE_ROOT, "binary"), File.join(sparse, "0"))
      assert_equal 3, Gsplat::IO::Colmap.read(directory).cameras.length
    end
  end

  private

  def compare_cameras(binary, text)
    binary.each_key do |id|
      assert_equal binary[id].model, text[id].model
      assert_equal [binary[id].width, binary[id].height], [text[id].width, text[id].height]
      assert_allclose binary[id].params, text[id].params, atol: 0.0, rtol: 0.0
      assert_allclose binary[id].intrinsics, text[id].intrinsics, atol: 0.0, rtol: 0.0
      assert_allclose binary[id].distortion, text[id].distortion, atol: 0.0, rtol: 0.0
    end
  end

  def compare_images(binary, text)
    binary.each_key do |id|
      assert_allclose binary[id].qvec, text[id].qvec, atol: 0.0, rtol: 0.0
      assert_allclose binary[id].tvec, text[id].tvec, atol: 0.0, rtol: 0.0
      assert_allclose binary[id].points2d, text[id].points2d, atol: 0.0, rtol: 0.0
      assert_equal binary[id].point3d_ids.to_a, text[id].point3d_ids.to_a
      assert_allclose binary[id].world_to_camera, text[id].world_to_camera, atol: 0.0, rtol: 0.0
    end
  end

  def compare_points(binary, text)
    binary.each_key do |id|
      assert_allclose binary[id].xyz, text[id].xyz, atol: 0.0, rtol: 0.0
      assert_equal binary[id].rgb.to_a, text[id].rgb.to_a
      assert_equal binary[id].error, text[id].error
      assert_equal binary[id].track, text[id].track
    end
  end
end
