# frozen_string_literal: true

module Gsplat
  # Lightweight reverse-mode differentiation for coarse gsplat operations.
  module Autograd
    GRAD_ENABLED_KEY = :gsplat_autograd_grad_enabled

    # Per-operation values retained for the backward pass.
    class Context
      attr_reader :needs_input_grad, :saved_values

      # @param needs_input_grad [Array<Boolean>] gradient requirement for each input
      def initialize(needs_input_grad)
        @needs_input_grad = needs_input_grad.freeze
        @saved_values = []
      end

      # Saves values required by the operation's backward implementation.
      #
      # @param values [Array<Object>]
      # @return [void]
      def save(*values)
        @saved_values.concat(values)
      end

      # Releases references retained for backward.
      #
      # @return [void]
      def clear
        @saved_values.clear
      end
    end

    module_function

    # Runs a block without recording an autograd graph.
    #
    # @yieldreturn [Object]
    def no_grad
      thread = Thread.current
      previous = thread.thread_variable_get(GRAD_ENABLED_KEY)
      thread.thread_variable_set(GRAD_ENABLED_KEY, false)
      yield
    ensure
      thread.thread_variable_set(GRAD_ENABLED_KEY, previous)
    end

    # Whether new operations should attach backward nodes.
    #
    # @return [Boolean]
    def grad_enabled?
      Thread.current.thread_variable_get(GRAD_ENABLED_KEY) != false
    end
  end
end
