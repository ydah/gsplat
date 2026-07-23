# frozen_string_literal: true

require "open3"
require "rbconfig"
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

  def test_simple_trainer_validates_colmap_fixture
    _output, error, status = run_example(
      "examples/simple_trainer.rb",
      "--data", "test/fixtures/colmap/binary",
      "--dry-run"
    )

    assert_predicate status, :success?, error
    assert_includes error, "cameras=3 images=3 points=100"
  end

  def test_render_path_validates_ply_fixture
    _output, error, status = run_example(
      "examples/render_path.rb",
      "--ply", "test/fixtures/inria_ascii.ply",
      "--frames", "2",
      "--dry-run"
    )

    assert_predicate status, :success?, error
    assert_match(/gaussians=\d+ frames=2 degree=\d+/, error)
  end

  private

  def example_paths
    %w[examples/fit_image.rb examples/simple_trainer.rb examples/render_path.rb]
  end

  def run_example(path, *)
    Open3.capture3(RbConfig.ruby, "-Ilib", path, *)
  end
end
