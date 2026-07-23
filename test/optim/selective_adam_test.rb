# frozen_string_literal: true

require "test_helper"

class SelectiveAdamTest < Minitest::Test
  def test_updates_only_visible_rows_and_moments
    parameter = variable([[1, 2], [3, 4], [5, 6]])
    optimizer = Gsplat::Optim::SelectiveAdam.new(parameter, lr: 0.1)
    parameter.send(:accumulate_grad, Numo::DFloat.ones(3, 2))

    optimizer.step(Numo::Bit[1, 0, 1])

    assert_allclose parameter.data,
                    Numo::DFloat[[0.9, 1.9], [3, 4], [4.9, 5.9]],
                    atol: 1e-12, rtol: 0.0
    assert_allclose optimizer.state.exp_avg,
                    Numo::DFloat[[0.1, 0.1], [0, 0], [0.1, 0.1]],
                    atol: 1e-12, rtol: 0.0
    assert_allclose optimizer.state.exp_avg_sq,
                    Numo::DFloat[[0.001, 0.001], [0, 0], [0.001, 0.001]],
                    atol: 1e-12, rtol: 0.0
  end

  def test_hidden_rows_remain_byte_for_byte_stable_across_steps
    parameter = variable([[1], [2]])
    optimizer = Gsplat::Optim::SelectiveAdam.new(parameter, lr: 0.1)
    parameter.send(:accumulate_grad, Numo::DFloat[[1], [1]])
    optimizer.step(Numo::Bit[1, 0]).zero_grad!
    first_row = parameter.data[0, 0]
    first_moment = optimizer.state.exp_avg[0, 0]
    parameter.send(:accumulate_grad, Numo::DFloat[[10], [1]])

    optimizer.step(Numo::Bit[0, 1])

    assert_equal first_row, parameter.data[0, 0]
    assert_equal first_moment, optimizer.state.exp_avg[0, 0]
    refute_equal 2.0, parameter.data[1, 0]
  end

  def test_all_visible_matches_dense_adam
    selective_parameter = variable([[1], [2]])
    dense_parameter = variable([[1], [2]])
    selective = Gsplat::Optim::SelectiveAdam.new(selective_parameter, lr: 0.01)
    dense = Gsplat::Optim::Adam.new(dense_parameter, lr: 0.01)
    gradient = Numo::DFloat[[0.5], [-0.25]]
    selective_parameter.send(:accumulate_grad, gradient)
    dense_parameter.send(:accumulate_grad, gradient)

    selective.step(Numo::Bit[1, 1])
    dense.step

    assert_allclose selective_parameter.data, dense_parameter.data, atol: 0.0, rtol: 0.0
    assert_allclose selective.state.exp_avg, dense.state.exp_avg, atol: 0.0, rtol: 0.0
    assert_allclose selective.state.exp_avg_sq, dense.state.exp_avg_sq, atol: 0.0, rtol: 0.0
  end

  def test_rejects_visibility_with_wrong_first_axis
    optimizer = Gsplat::Optim::SelectiveAdam.new(variable([[1], [2]]))

    error = assert_raises(Gsplat::ShapeError) { optimizer.step(Numo::Bit[1]) }
    assert_includes error.message, "visibility [2]"
  end

  private

  def variable(values)
    Gsplat::Autograd::Variable.new(Numo::DFloat.cast(values), requires_grad: true)
  end
end
