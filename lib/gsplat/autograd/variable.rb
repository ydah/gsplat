# frozen_string_literal: true

module Gsplat
  module Autograd
    # A Numo array with an optional reverse-mode gradient and creator node.
    class Variable
      attr_reader :absgrad, :data, :grad, :creator

      # @param data [Numo::NArray] tensor data
      # @param requires_grad [Boolean] whether to accumulate a gradient
      # @param creator [GraphNode, nil] producing operation
      def initialize(data, requires_grad: false, creator: nil)
        raise ArgumentError, "Variable data must be a Numo::NArray" unless data.is_a?(Numo::NArray)

        @data = data
        @requires_grad = requires_grad
        @creator = creator
        @grad = nil
        @absgrad = nil
        @absgrad_targets = []
      end

      # Whether this variable accumulates gradients.
      #
      # @return [Boolean]
      def requires_grad?
        @requires_grad
      end

      # Clears an accumulated leaf gradient.
      #
      # @return [void]
      def zero_grad!
        @grad = nil
        @absgrad = nil
        @absgrad_targets.each { |target, key| target[key] = nil }
      end

      # Accumulates an auxiliary absolute gradient produced by rasterization.
      #
      # @api private
      # @param gradient [Numo::NArray]
      # @return [void]
      def accumulate_absgrad(gradient)
        gradient = cast_gradient(gradient)
        unless gradient.shape == data.shape
          raise ShapeError, "absgrad shape mismatch: expected #{data.shape.inspect}, got #{gradient.shape.inspect}"
        end

        @absgrad = @absgrad ? @absgrad + gradient : gradient.dup
        @absgrad_targets.each { |target, key| target[key] = @absgrad }
      end

      # Binds auxiliary absolute-gradient updates to a metadata hash entry.
      #
      # @api private
      # @return [void]
      def bind_absgrad(target, key)
        @absgrad_targets << [target, key]
        target[key] = @absgrad
      end

      # Removes a creator after its graph node has completed backward.
      #
      # @api private
      # @param node [GraphNode]
      # @return [void]
      def clear_creator(node)
        @creator = nil if @creator.equal?(node)
      end

      # Runs reverse-mode differentiation from this variable.
      #
      # @param gradient [Numo::NArray, Numeric, nil] output gradient; nil is valid for scalar output
      # @return [void]
      def backward(gradient = nil)
        raise Gsplat::Error, "cannot call backward on a variable that does not require gradients" unless requires_grad?

        accumulate_grad(initial_gradient(gradient))
        topological_nodes.reverse_each do |node|
          grad_outputs = node.outputs.map { |output| output.grad || zeros_like(output.data) }
          grad_inputs = normalize_grad_inputs(node.backward(*grad_outputs), node.inputs.length)
          node.inputs.each_with_index do |input, index|
            next unless input.is_a?(Variable) && node.context.needs_input_grad.fetch(index)

            input.accumulate_grad(grad_inputs.fetch(index))
          end
          node.release
        end
      end

      protected

      def accumulate_grad(gradient)
        return if gradient.nil?

        gradient = cast_gradient(gradient)
        unless gradient.shape == data.shape
          raise ShapeError, "gradient shape mismatch: expected #{data.shape.inspect}, got #{gradient.shape.inspect}"
        end

        @grad = @grad ? @grad + gradient : gradient.dup
      end

      private

      def initial_gradient(gradient)
        return cast_gradient(gradient) unless gradient.nil?
        raise Gsplat::Error, "backward on a non-scalar output requires an explicit gradient" unless data.size == 1

        data.shape.empty? ? data.class.cast(1) : data.class.ones(*data.shape)
      end

      def cast_gradient(gradient)
        gradient = gradient.data if gradient.is_a?(Variable)
        data.class.cast(gradient)
      end

      def zeros_like(array)
        array.shape.empty? ? array.class.cast(0) : array.class.zeros(*array.shape)
      end

      def normalize_grad_inputs(result, input_count)
        return result if input_count == 1 && result.is_a?(Array) && result.length == 1
        return [result] if input_count == 1
        return result if result.is_a?(Array) && result.length == input_count

        raise Gsplat::Error, "Function.backward returned gradients for the wrong number of inputs"
      end

      def topological_nodes
        ordered = []
        visited = {}.compare_by_identity
        visit = lambda do |node|
          return unless node
          return if visited.key?(node)

          visited[node] = true
          node.inputs.each { |input| visit.call(input.creator) if input.is_a?(Variable) }
          ordered << node
        end
        visit.call(creator)
        ordered
      end
    end
  end
end
