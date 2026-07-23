# frozen_string_literal: true

require "logger"
require "numo/narray"

require_relative "gsplat/version"

# Differentiable 3D Gaussian splatting for Ruby.
module Gsplat
  class Error < StandardError; end
  class ShapeError < Error; end
  class NotSupportedError < Error; end

  class << self
    attr_writer :logger, :rng

    # Logger used for backend fallbacks and unsupported option warnings.
    #
    # @return [Logger]
    def logger
      @logger ||= Logger.new($stderr, level: Logger::WARN)
    end

    # Shared deterministic random source used by initialization and strategies.
    #
    # @return [Random]
    def rng
      @rng ||= Random.new
    end
  end
end
