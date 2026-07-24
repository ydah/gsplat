# frozen_string_literal: true

require "open3"
require "rbconfig"
require "tmpdir"
require "test_helper"

class ExamplesCliTest < Minitest::Test
  def test_every_example_exposes_help
    example_paths.each do |path|
      output, error, status = run_example(path, "--help")

      assert_predicate status, :success?, "#{path}: #{error}"
      assert_includes output, "Usage:"
    end
  end

  def test_fit_image_completes_a_real_step
    _output, error, status = run_example(
      "examples/fit_image.rb",
      "--width", "4",
      "--height", "4",
      "--gaussians", "4",
      "--steps", "1",
      "--learning-rate", "1"
    )

    assert_predicate status, :success?, error
    assert_includes error, "fit complete:"
  end

  def test_simple_trainer_validates_bundled_sample
    _output, error, status = run_example(
      "examples/simple_trainer.rb",
      "--dry-run"
    )

    assert_predicate status, :success?, error
    assert_includes error, "cameras=1 images=3 points=16"
  end

  def test_render_path_validates_bundled_sample
    _output, error, status = run_example(
      "examples/render_path.rb",
      "--frames", "2",
      "--dry-run"
    )

    assert_predicate status, :success?, error
    assert_includes error, "gaussians=16 frames=2 degree=0"
  end

  def test_simple_trainer_trains_bundled_sample
    Dir.mktmpdir do |directory|
      _output, error, status = run_example(
        "examples/simple_trainer.rb",
        "--output", directory,
        "--steps", "1"
      )

      assert_predicate status, :success?, error
      assert_path_exists File.join(directory, "splats.ply")
    end
  end

  def test_render_path_renders_bundled_sample
    Dir.mktmpdir do |directory|
      _output, error, status = run_example(
        "examples/render_path.rb",
        "--output", directory,
        "--frames", "1"
      )

      assert_predicate status, :success?, error
      assert_path_exists File.join(directory, "frame_0000.png")
    end
  end

  def test_sample_data_generator_writes_colmap_and_ply_assets
    Dir.mktmpdir do |directory|
      _output, error, status = run_example(
        "examples/generate_sample_data.rb",
        "--output", directory
      )

      assert_predicate status, :success?, error
      assert_path_exists File.join(directory, "colmap", "images", "view_000.png")
      assert_path_exists File.join(directory, "colmap", "sparse", "0", "cameras.txt")
      assert_path_exists File.join(directory, "splats.ply")
    end
  end

  private

  def example_paths
    %w[
      examples/fit_image.rb
      examples/generate_sample_data.rb
      examples/simple_trainer.rb
      examples/render_path.rb
    ]
  end

  def run_example(path, *)
    Open3.capture3(RbConfig.ruby, "-Ilib", path, *)
  end
end
