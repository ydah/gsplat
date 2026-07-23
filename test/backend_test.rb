# frozen_string_literal: true

require "stringio"
require "open3"
require "rbconfig"
require "test_helper"

class BackendTest < Minitest::Test
  def setup
    @original_backend = Gsplat.backend
    @original_logger = Gsplat.logger
  end

  def teardown
    Gsplat.backend = @original_backend
    Gsplat.logger = @original_logger
  end

  def test_registers_and_dispatches_to_selected_backend
    Gsplat::Backend.register(:p0_double, :ruby, ->(value) { value * 2 })
    Gsplat.backend = :ruby

    assert_equal 6, Gsplat::Backend.dispatch(:p0_double, 3)
  end

  def test_forwards_keyword_arguments
    callable = ->(value, factor:) { value * factor }
    Gsplat::Backend.register(:p0_scale, :ruby, callable)
    Gsplat.backend = :ruby

    assert_equal 12, Gsplat::Backend.dispatch(:p0_scale, 4, factor: 3)
  end

  def test_auto_prefers_native_when_registered
    Gsplat::Backend.register(:p0_identity, :ruby, -> { :ruby })
    Gsplat::Backend.register(:p0_identity, :native, -> { :native })
    Gsplat.backend = :auto

    assert_equal :native, Gsplat::Backend.dispatch(:p0_identity)
  end

  def test_auto_falls_back_to_ruby_and_warns_once
    output = StringIO.new
    Gsplat.logger = Logger.new(output)
    Gsplat::Backend.register(:p0_fallback, :ruby, -> { :ruby })
    Gsplat.backend = :auto

    2.times { assert_equal :ruby, Gsplat::Backend.dispatch(:p0_fallback) }

    assert_equal 1, output.string.scan("native backend is unavailable").length
  end

  def test_explicit_missing_backend_raises
    Gsplat::Backend.register(:p0_missing_native, :ruby, -> { :ruby })
    Gsplat.backend = :native

    error = assert_raises(Gsplat::NotSupportedError) do
      Gsplat::Backend.dispatch(:p0_missing_native)
    end
    assert_match(/p0_missing_native/, error.message)
  end

  def test_rejects_unknown_backend
    assert_raises(ArgumentError) { Gsplat.backend = :gpu }
  end

  def test_reads_backend_from_environment
    script = 'require "gsplat"; print Gsplat.backend'
    env = { "GSPLAT_BACKEND" => "ruby" }
    output, error, status = Open3.capture3(env, RbConfig.ruby, "-Ilib", "-e", script)

    assert_predicate status, :success?, error
    assert_equal "ruby", output
  end
end
