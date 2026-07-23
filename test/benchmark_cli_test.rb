# frozen_string_literal: true

require "open3"
require "rbconfig"
require "test_helper"

class BenchmarkCliTest < Minitest::Test
  SCRIPT = "benchmarks/bench_rasterize.rb"

  def test_lists_all_design_workloads
    output, error, status = run_benchmark("--list")

    assert_predicate status, :success?, error
    %w[raster fit_image colmap].each { |name| assert_match(/^#{name}:/, output) }
  end

  def test_quick_raster_and_fit_workloads_execute
    %w[raster fit_image].each do |scenario|
      output, error, status = run_benchmark(
        "--scenario", scenario, "--backend", "ruby", "--quick"
      )

      assert_predicate status, :success?, error
      assert_includes output, %("#{scenario}")
      assert_includes output, %("mean_ms")
    end
  end

  private

  def run_benchmark(*)
    Open3.capture3(RbConfig.ruby, "-Ilib", SCRIPT, *)
  end
end
