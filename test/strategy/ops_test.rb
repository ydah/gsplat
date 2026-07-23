# frozen_string_literal: true

require "test_helper"

class StrategyOpsTest < Minitest::Test
  KEYS = %i[means quats scales opacities sh0 shN].freeze

  def setup
    @params = {
      means: variable([[0, 0, 0], [1, 0, 0]]),
      quats: variable([[1, 0, 0, 0], [1, 0, 0, 0]]),
      scales: variable([[0, 0, 0], [0, 0, 0]]),
      opacities: variable([0.0, 1.0]),
      sh0: variable([[[0.1, 0.2, 0.3]], [[0.4, 0.5, 0.6]]]),
      shN: variable(Numo::DFloat.zeros(2, 3, 3))
    }
    @optimizers = @params.transform_values { |parameter| Gsplat::Optim::Adam.new(parameter) }
    seed_optimizer_states
  end

  def test_base_sanity_checks_keys_counts_and_optimizer_ownership
    assert Gsplat::Strategy::Base.new.check_sanity(@params, @optimizers)

    error = assert_raises(ArgumentError) do
      Gsplat::Strategy::Base.new.check_sanity(@params.reject { |key| key == :means }, @optimizers)
    end
    assert_includes error.message, "means"
  end

  def test_duplicate_appends_selected_values_and_zero_moments
    added = Gsplat::Strategy::Ops.duplicate!(@params, @optimizers, Numo::Bit[1, 0])

    assert_equal 1, added
    assert_equal [3, 3], @params[:means].data.shape
    assert_allclose @params[:means].data[2, true], Numo::DFloat[0, 0, 0], atol: 1e-15, rtol: 0.0
    KEYS.each do |key|
      assert @optimizers[key].state.exp_avg[2, *Array.new(@params[key].data.ndim - 1, 0)].zero?
    end
  end

  def test_remove_selects_parameters_and_optimizer_state
    removed = Gsplat::Strategy::Ops.remove!(@params, @optimizers, Numo::Bit[1, 0])

    assert_equal 1, removed
    assert_equal [1, 3], @params[:means].data.shape
    assert_allclose @params[:means].data, Numo::DFloat[[1, 0, 0]], atol: 1e-15, rtol: 0.0
    assert_allclose @optimizers[:means].state.exp_avg,
                    Numo::DFloat.ones(1, 3) * 0.1, atol: 1e-12, rtol: 0.0
  end

  def test_split_replaces_parent_with_two_sampled_children
    split = Gsplat::Strategy::Ops.split!(
      @params,
      @optimizers,
      Numo::Bit[1, 0],
      rng: Random.new(5)
    )

    assert_equal 1, split
    assert_equal [3, 3], @params[:means].data.shape
    assert_allclose @params[:scales].data[1.., true],
                    Numo::DFloat.ones(2, 3) * -::Math.log(1.6), atol: 1e-12, rtol: 0.0
    assert @params[:means].data[1.., true].abs.sum.positive?
    assert_allclose @optimizers[:means].state.exp_avg[1.., true],
                    Numo::DFloat.zeros(2, 3), atol: 0.0, rtol: 0.0
  end

  def test_reset_opacity_caps_logits_and_zeros_state
    changed = Gsplat::Strategy::Ops.reset_opacity!(@params, @optimizers, maximum: 0.02)

    assert_equal 2, changed
    cap = ::Math.log(0.02 / 0.98)
    assert_allclose @params[:opacities].data, Numo::DFloat[cap, cap], atol: 1e-12, rtol: 0.0
    assert_allclose @optimizers[:opacities].state.exp_avg, Numo::DFloat.zeros(2),
                    atol: 0.0, rtol: 0.0
  end

  private

  def variable(value)
    Gsplat::Autograd::Variable.new(Numo::DFloat.cast(value), requires_grad: true)
  end

  def seed_optimizer_states
    @params.each do |key, parameter|
      parameter.send(:accumulate_grad, Numo::DFloat.ones(*parameter.data.shape))
      @optimizers.fetch(key).step.zero_grad!
      parameter.data[] = parameter.data + 0.001
    end
  end
end
