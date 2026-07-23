# frozen_string_literal: true

require "logger"
require "numo/narray"

require_relative "gsplat/version"
require_relative "gsplat/backend"
require_relative "gsplat/autograd/context"
require_relative "gsplat/autograd/function"
require_relative "gsplat/autograd/variable"
require_relative "gsplat/math/mat"
require_relative "gsplat/math/quaternion"
require_relative "gsplat/io/npy"

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

    # Active operation backend.
    #
    # @return [Symbol] :auto, :ruby, or :native
    def backend
      @backend ||= Backend.normalize_backend(ENV.fetch("GSPLAT_BACKEND", "auto"))
    end

    # Selects the operation backend.
    #
    # @param value [Symbol, String] :auto, :ruby, or :native
    # @return [Symbol] normalized backend
    def backend=(value)
      @backend = Backend.normalize_backend(value)
    end
  end
end
