# frozen_string_literal: true

module Gsplat
  # Operation registry and runtime backend selector.
  module Backend
    # Backend names accepted by {Gsplat.backend=}.
    VALID_BACKENDS = %i[auto ruby native].freeze

    @registry = Hash.new { |operations, name| operations[name] = {} }
    @mutex = Mutex.new
    @fallback_warned = false

    class << self
      # Registers an operation implementation.
      #
      # @param op_name [Symbol, String] operation identifier
      # @param backend [Symbol, String] implementation backend (:ruby or :native)
      # @param callable [#call, nil] implementation callable
      # @yield implementation body when callable is omitted
      # @return [#call] the registered implementation
      def register(op_name, backend, callable = nil, &block)
        implementation = callable || block
        raise ArgumentError, "backend implementation must respond to #call" unless implementation.respond_to?(:call)

        backend = normalize_backend(backend, allow_auto: false)
        @mutex.synchronize { @registry[op_name.to_sym][backend] = implementation }
        implementation
      end

      # Dispatches an operation to the selected implementation.
      #
      # @param op_name [Symbol, String] operation identifier
      # @return [Object] operation result
      def dispatch(op_name, ...)
        operation = op_name.to_sym
        implementation = implementation_for(operation)
        implementation.call(...)
      end

      # Validates and normalizes a backend name.
      #
      # @param backend [Symbol, String]
      # @param allow_auto [Boolean]
      # @return [Symbol]
      def normalize_backend(backend, allow_auto: true)
        normalized = backend.to_s.downcase.to_sym
        valid = allow_auto ? VALID_BACKENDS : VALID_BACKENDS.drop(1)
        return normalized if valid.include?(normalized)

        raise ArgumentError, "unknown backend #{backend.inspect}; expected one of #{valid.join(', ')}"
      end

      private

      def implementation_for(operation)
        implementations = @mutex.synchronize { @registry.fetch(operation, {}).dup }
        selected = Gsplat.backend
        return implementations.fetch(selected) { missing_backend!(operation, selected) } unless selected == :auto
        return implementations[:native] if implementations.key?(:native)

        warn_fallback_once
        implementations.fetch(:ruby) { missing_backend!(operation, :ruby) }
      end

      def missing_backend!(operation, backend)
        raise NotSupportedError, "#{backend} backend does not implement #{operation}"
      end

      def warn_fallback_once
        should_warn = @mutex.synchronize do
          next false if @fallback_warned

          @fallback_warned = true
        end
        Gsplat.logger.warn("native backend is unavailable; falling back to ruby") if should_warn
      end
    end
  end
end
