# frozen_string_literal: true

require "date"
require "test_helper"

class ArchitectureDecisionRecordsTest < Minitest::Test
  DECISIONS_DIR = File.expand_path("../docs/decisions", __dir__)
  INDEX_PATH = File.expand_path("../docs/DECISIONS.md", __dir__)
  ADR_FILENAME = /\A(?<number>\d{4})-[a-z0-9]+(?:-[a-z0-9]+)*\.md\z/
  TERMINAL_STATUSES = %w[Accepted Rejected Deprecated].freeze

  def setup
    @index = File.read(INDEX_PATH)
    @adr_paths = Dir[File.join(DECISIONS_DIR, "[0-9][0-9][0-9][0-9]-*.md")]
                 .reject { |path| File.basename(path).start_with?("0000-") }
                 .sort
  end

  def test_template_defines_the_minimal_record
    template = File.read(File.join(DECISIONS_DIR, "0000-template.md"))

    assert_match(/\A# NNNN: Decision title\n/, template)
    assert_includes template, "- Status: Proposed"
    assert_includes template, "- Date: YYYY-MM-DD"
    %w[Context Decision Consequences].each do |section|
      assert_includes template, "## #{section}"
    end
  end

  def test_records_have_valid_names_headings_and_metadata
    refute_empty @adr_paths

    @adr_paths.each do |path|
      basename = File.basename(path)
      match = ADR_FILENAME.match(basename)
      refute_nil match, "invalid ADR filename: #{basename}"

      content = File.read(path)
      assert_match(/\A# #{match[:number]}: \S/, content, "title number differs in #{basename}")
      assert_valid_status(content, basename)
      assert_valid_date(content, basename)
      %w[Context Decision Consequences].each do |section|
        assert_includes content, "## #{section}", "#{basename} is missing #{section}"
      end
    end
  end

  def test_index_links_every_record_once
    expected_filenames = @adr_paths.map { |path| File.basename(path) }
    indexed_filenames = @index.scan(%r{^\| \[\d{4}\]\(decisions/([^)]+)\) \|}).flatten

    assert_equal expected_filenames, indexed_filenames

    @adr_paths.each do |path|
      relative_link = "decisions/#{File.basename(path)}"

      assert_equal 1, @index.scan("(#{relative_link})").length,
                   "#{relative_link} must appear exactly once in the ADR index"
    end
  end

  private

  def assert_valid_status(content, basename)
    status = content[/^- Status: (.+)$/, 1]
    valid = status == "Proposed" ||
            TERMINAL_STATUSES.include?(status) ||
            status&.match?(/\ASuperseded by \d{4}\z/)

    assert valid, "invalid or missing status in #{basename}: #{status.inspect}"
  end

  def assert_valid_date(content, basename)
    value = content[/^- Date: (.+)$/, 1]
    parsed = Date.iso8601(value.to_s)

    assert_equal value, parsed.iso8601, "invalid date in #{basename}"
  rescue Date::Error
    flunk "invalid or missing date in #{basename}: #{value.inspect}"
  end
end
