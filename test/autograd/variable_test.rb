# frozen_string_literal: true

require "test_helper"

class AutogradVariableTest < Minitest::Test
  class Affine < Gsplat::Autograd::Function
    def self.forward(context, input, scale:, bias:)
      context.save(scale)
      (input * scale) + bias
    end

    def self.backward(context, grad_output)
      [grad_output * context.saved_values.fetch(0)]
    end
  end

  class SquareSum < Gsplat::Autograd::Function
    def self.forward(context, input)
      context.save(input)
      input.class.cast((input**2).sum)
    end

    def self.backward(context, grad_output)
      input = context.saved_values.fetch(0)
      [input * (2.0 * grad_output.to_f)]
    end
  end

  class Add < Gsplat::Autograd::Function
    def self.forward(_context, left, right)
      left + right
    end

    def self.backward(_context, grad_output)
      [grad_output, grad_output]
    end
  end

  class Multiply < Gsplat::Autograd::Function
    def self.forward(context, left, right)
      context.save(left, right)
      left * right
    end

    def self.backward(context, grad_output)
      left, right = context.saved_values
      [
        grad_output * right,
        context.needs_input_grad.fetch(1) ? grad_output * left : nil
      ]
    end
  end

  class Split < Gsplat::Autograd::Function
    def self.forward(_context, input)
      [input * 2.0, input * 3.0]
    end

    def self.backward(_context, grad_left, grad_right)
      [(grad_left * 2.0) + (grad_right * 3.0)]
    end
  end

  def test_composed_function_matches_analytic_gradient
    x = variable([1.0, 2.0, 3.0])
    affine = Affine.apply(x, scale: 2.0, bias: 3.0)
    loss = SquareSum.apply(affine)

    loss.backward

    assert_allclose x.grad, Numo::SFloat[20.0, 28.0, 36.0], atol: 1e-6, rtol: 0.0
  end

  def test_reused_branch_accumulates_gradient
    x = variable([1.0, 2.0])
    doubled = Add.apply(x, x)

    SquareSum.apply(doubled).backward

    assert_allclose x.grad, Numo::SFloat[8.0, 16.0], atol: 1e-6, rtol: 0.0
  end

  def test_no_grad_does_not_attach_a_creator
    x = variable([1.0])

    output = Gsplat::Autograd.no_grad { Affine.apply(x, scale: 2.0, bias: 3.0) }

    refute output.requires_grad?
    assert_nil output.creator
  end

  def test_context_identifies_inputs_that_do_not_need_gradients
    left = variable([2.0, 3.0])
    right = Gsplat::Autograd::Variable.new(Numo::SFloat[4.0, 5.0])

    SquareSum.apply(Multiply.apply(left, right)).backward

    assert_allclose left.grad, Numo::SFloat[64.0, 150.0], atol: 1e-6, rtol: 0.0
    assert_nil right.grad
  end

  def test_multiple_outputs_share_one_backward_node
    x = variable([1.0, 2.0])
    left, right = Split.apply(x)

    SquareSum.apply(Add.apply(left, right)).backward

    assert_allclose x.grad, Numo::SFloat[50.0, 100.0], atol: 1e-6, rtol: 0.0
  end

  def test_non_scalar_output_requires_an_explicit_gradient
    x = variable([1.0, 2.0])

    error = assert_raises(Gsplat::Error) { Affine.apply(x, scale: 2.0, bias: 3.0).backward }

    assert_match(/non-scalar/, error.message)
  end

  private

  def variable(values)
    Gsplat::Autograd::Variable.new(Numo::SFloat.cast(values), requires_grad: true)
  end
end
