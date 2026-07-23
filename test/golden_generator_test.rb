# frozen_string_literal: true

require "open3"
require "test_helper"

class GoldenGeneratorTest < Minitest::Test
  def test_dry_run_lists_every_required_family_without_python_dependencies
    output, error, status = Open3.capture3(
      ENV.fetch("PYTHON", "python3"),
      "tools/generate_golden.py",
      "--dry-run"
    )

    assert_predicate status, :success?, error
    %w[quat sh proj isect raster render strategy relocation ssim].each do |family|
      assert_match(/#{family}_/, output)
    end
  end

  def test_python_dependencies_are_exactly_pinned
    requirements = File.readlines("tools/requirements.txt", chomp: true).reject do |line|
      line.empty? || line.start_with?("#")
    end

    assert(requirements.all? { |line| line.match?(/\A[a-zA-Z0-9_-]+==[^=]+\z/) })
    assert_includes requirements, "gsplat==1.5.3"
  end
end
