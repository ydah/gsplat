# frozen_string_literal: true

require "open3"
require "rbconfig"
require "test_helper"

class ReadmeTest < Minitest::Test
  STANDARD_SECTIONS = [
    "Features",
    "Installation",
    "Usage",
    "API compatibility",
    "Limitations",
    "Documentation",
    "Development",
    "Contributing",
    "License"
  ].freeze

  def test_quick_start_is_executable
    readme = File.read("README.md")
    match = readme.match(
      /<!-- quickstart:start -->\s*```ruby\n(?<code>.*?)\n```\s*<!-- quickstart:end -->/m
    )
    refute_nil match, "README quick-start markers are missing"

    output, error, status = Open3.capture3(
      RbConfig.ruby, "-Ilib", "-e", match[:code]
    )

    assert_predicate status, :success?, error
    assert_equal "[[1, 4, 4, 3], [1, 4, 4, 1], [1, 1, 2], [1, 3]]\n", output
  end

  def test_sections_follow_the_standard_gem_readme_structure
    headings = File.read("README.md").scan(/^## (.+)$/).flatten

    assert_equal STANDARD_SECTIONS, headings
  end
end
