# frozen_string_literal: true

require "test_helper"

class AdamTest < Minitest::Test
  def test_scalar_quadratic_matches_adam_reference_trajectory
    parameter = Gsplat::Autograd::Variable.new(Numo::DFloat[1.0], requires_grad: true)
    optimizer = Gsplat::Optim::Adam.new(parameter, lr: 0.1, eps: 1e-15)

    3.times do
      parameter.send(:accumulate_grad, 2 * parameter.data)
      optimizer.step.zero_grad!
    end

    assert_allclose parameter.data, Numo::DFloat[0.70158627], atol: 1e-7, rtol: 1e-7
    assert_equal 3, optimizer.state.step
  end

  def test_parameter_groups_use_independent_rates
    first = Gsplat::Autograd::Variable.new(Numo::DFloat[1.0], requires_grad: true)
    second = Gsplat::Autograd::Variable.new(Numo::DFloat[1.0], requires_grad: true)
    optimizer = Gsplat::Optim::Adam.new(
      first: { variable: first, lr: 0.1 },
      second: { variable: second, lr: 0.01 }
    )
    first.send(:accumulate_grad, Numo::DFloat[1])
    second.send(:accumulate_grad, Numo::DFloat[1])

    optimizer.step

    assert_allclose first.data, Numo::DFloat[0.9], atol: 1e-12, rtol: 0.0
    assert_allclose second.data, Numo::DFloat[0.99], atol: 1e-12, rtol: 0.0
  end

  def test_state_index_editing_preserves_alignment
    parameter = Gsplat::Autograd::Variable.new(Numo::DFloat[[1], [2], [3]], requires_grad: true)
    optimizer = Gsplat::Optim::Adam.new(parameter)
    parameter.send(:accumulate_grad, Numo::DFloat[[1], [2], [3]])
    optimizer.step

    optimizer.select!(Numo::Bit[1, 0, 1]).append!(2)
    optimizer.zero_state_at!([1, 3])

    assert_equal [4, 1], optimizer.state.exp_avg.shape
    assert_allclose optimizer.state.exp_avg[true, 0], Numo::DFloat[0.1, 0.0, 0.0, 0.0],
                    atol: 1e-12, rtol: 0.0
  end

  def test_exponential_lr_reaches_final_rate
    parameter = Gsplat::Autograd::Variable.new(Numo::DFloat[1], requires_grad: true)
    optimizer = Gsplat::Optim::Adam.new(parameter, lr: 1e-2)
    scheduler = Gsplat::Optim::ExponentialLR.new(optimizer, lr_final: 1e-4, max_steps: 100)

    scheduler.step(50)
    assert_in_delta 1e-3, optimizer.learning_rate, 1e-15

    scheduler.step(100)
    assert_in_delta 1e-4, optimizer.learning_rate, 1e-15
  end
end
