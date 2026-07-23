# frozen_string_literal: true

require "test_helper"

class GemspecTest < Minitest::Test
  def setup
    @specification = Gem::Specification.load("gsplat.gemspec")
  end

  def test_release_metadata
    assert_equal Gem::Version.new("1.0.0"), @specification.version
    assert_equal "https://github.com/ydah/gsplat", @specification.homepage
    assert_equal @specification.homepage, @specification.metadata.fetch("source_code_uri")
    assert_equal "true", @specification.metadata.fetch("rubygems_mfa_required")
  end

  def test_release_file_manifest
    %w[LICENSE.txt README.md gsplat.gemspec lib/gsplat.rb ext/gsplat_native/extconf.rb].each do |path|
      assert_includes @specification.files, path
    end
    refute(@specification.files.any? { |path| path.start_with?("test/") })
  end

  def test_runtime_dependency_is_bounded
    dependency = @specification.runtime_dependencies.find { |entry| entry.name == "numo-narray" }

    refute_nil dependency
    assert dependency.requirement.satisfied_by?(Gem::Version.new("0.9.2"))
    refute dependency.requirement.satisfied_by?(Gem::Version.new("1.0.0"))
  end
end
