# frozen_string_literal: true

module Gsplat
  module Autograd
    # A recorded invocation shared by all outputs of one Function.
    class GraphNode
      attr_reader :function, :context, :inputs
      attr_accessor :outputs

      def initialize(function, context, inputs)
        @function = function
        @context = context
        @inputs = inputs
        @outputs = []
      end

      # Delegates output gradients to the operation's backward method.
      # @api private
      def backward(*grad_outputs)
        function.backward(context, *grad_outputs)
      end

      # Releases graph references after a backward traversal.
      # @api private
      def release
        outputs.each { |output| output.clear_creator(self) }
        context.clear
        @inputs = []
        @outputs = []
      end
    end

    # Base class for differentiable coarse-grained operations.
    class Function
      class << self
        # Executes forward and records a graph node when any input needs a gradient.
        #
        # @param inputs [Array<Object, Variable>]
        # @return [Variable, Array<Variable>]
        def apply(*inputs, **)
          needs_input_grad = inputs.map { |input| input.is_a?(Variable) && input.requires_grad? }
          context = Context.new(needs_input_grad, inputs)
          raw_inputs = inputs.map { |input| input.is_a?(Variable) ? input.data : input }
          result = forward(context, *raw_inputs, **)
          multiple_outputs = result.is_a?(Array)
          output_data = multiple_outputs ? result : [result]
          requires_grad = Autograd.grad_enabled? && needs_input_grad.any?
          node = GraphNode.new(self, context, inputs) if requires_grad
          outputs = output_data.map do |data|
            validate_output!(data)
            Variable.new(data, requires_grad: requires_grad, creator: node)
          end
          node.outputs = outputs if node

          multiple_outputs ? outputs : outputs.first
        end

        private

        def validate_output!(data)
          return if data.is_a?(Numo::NArray)

          raise Gsplat::Error, "Function.forward must return Numo::NArray output(s), got #{data.class}"
        end
      end
    end
  end
end
