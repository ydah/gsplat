# frozen_string_literal: true

require "stringio"
require "test_helper"

class CheckpointTest < Minitest::Test
  def test_round_trip_restores_identical_adam_trajectory
    params, optimizers = training_state
    apply_gradient(params[:means], optimizers[:means], Numo::DFloat[0.2, -0.4])
    checkpoint = save_checkpoint(params, optimizers)
    expected_optimizer = optimizers[:means]
    apply_gradient(params[:means], expected_optimizer, Numo::DFloat[-0.1, 0.3])
    expected = params[:means].data.dup
    snapshot, restored_params, restored_optimizers = restore_checkpoint(checkpoint)
    apply_gradient(restored_params[:means], restored_optimizers[:means], Numo::DFloat[-0.1, 0.3])

    assert_equal 17, snapshot.step
    assert_equal({ max_steps: 100, strategy: "default" }, snapshot.config)
    assert_allclose restored_params[:means].data, expected, atol: 1e-15, rtol: 1e-15
    assert_equal expected_optimizer.state.step, restored_optimizers[:means].state.step
  end

  def test_load_exposes_parameter_and_optimizer_arrays
    params, optimizers = training_state
    output = StringIO.new("".b)
    Gsplat::IO::Checkpoint.save(
      output, params: params, optimizers: optimizers, step: 0, config: {}
    )

    snapshot = Gsplat::IO::Checkpoint.load(StringIO.new(output.string))

    assert_equal %i[means], snapshot.params.keys
    assert_equal [2], snapshot.optimizer_states.dig(:means, :default, :exp_avg).shape
  end

  private

  def training_state(values: [1, -2])
    variable = Gsplat::Autograd::Variable.new(Numo::DFloat.cast(values), requires_grad: true)
    [{ means: variable }, { means: Gsplat::Optim::Adam.new(variable, lr: 0.01) }]
  end

  def apply_gradient(variable, optimizer, gradient)
    variable.send(:accumulate_grad, gradient)
    optimizer.step.zero_grad!
  end

  def save_checkpoint(params, optimizers)
    output = StringIO.new("".b)
    Gsplat::IO::Checkpoint.save(
      output,
      params: params,
      optimizers: optimizers,
      step: 17,
      config: { max_steps: 100, strategy: "default" }
    )
    output
  end

  def restore_checkpoint(checkpoint)
    params, optimizers = training_state(values: [9, 9])
    snapshot = Gsplat::IO::Checkpoint.restore!(
      StringIO.new(checkpoint.string),
      params: params,
      optimizers: optimizers
    )
    [snapshot, params, optimizers]
  end
end
