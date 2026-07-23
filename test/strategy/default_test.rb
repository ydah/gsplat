# frozen_string_literal: true

require "test_helper"

class DefaultStrategyTest < Minitest::Test
  def test_collects_screen_normalized_visible_gradients
    strategy = Gsplat::Strategy::Default.new(refine_start_iter: 10)
    params, optimizers = scene_params(2)
    state = strategy.initialize_state(scene_scale: 1.0)
    projected = variable(Numo::DFloat.zeros(1, 2, 2))
    projected.send(:accumulate_grad, Numo::DFloat[[[0.01, 0.02], [0.03, 0.04]]])
    info = {
      means2d: projected,
      radii: Numo::Int32[[2, 0]],
      width: 100,
      height: 50
    }

    strategy.step_post_backward(
      params: params,
      optimizers: optimizers,
      state: state,
      step: 1,
      info: info
    )

    assert_allclose state[:grad2d], Numo::DFloat[::Math.sqrt(0.5), 0], atol: 1e-12, rtol: 0.0
    assert_allclose state[:count], Numo::DFloat[1, 0], atol: 0.0, rtol: 0.0
  end

  def test_refinement_duplicates_small_and_splits_large_gaussians
    strategy = Gsplat::Strategy::Default.new(
      refine_start_iter: 0,
      refine_stop_iter: 10,
      refine_every: 1,
      grow_grad2d: 0.1,
      grow_scale3d: 0.05,
      reset_every: 100
    )
    params, optimizers = scene_params(2)
    params[:scales].data[0, true] = ::Math.log(0.01)
    params[:scales].data[1, true] = ::Math.log(0.2)
    state = strategy.initialize_state(scene_scale: 1.0)
    state[:grad2d] = Numo::DFloat[1, 1]
    state[:count] = Numo::DFloat[1, 1]
    state[:radii] = Numo::DFloat.zeros(2)

    strategy.send(:refine!, params, optimizers, state, 1)

    assert_equal 4, params[:means].data.shape[0]
    params.each_key do |key|
      assert_equal 4, optimizers[key].state.exp_avg.shape[0]
    end
  end

  def test_prunes_low_opacity
    strategy = Gsplat::Strategy::Default.new(prune_opa: 0.01)
    params, optimizers = scene_params(2)
    params[:opacities].data[] = Numo::DFloat[-10, 0]
    state = strategy.initialize_state(scene_scale: 1.0)

    strategy.send(:prune!, params, optimizers, state, 1)

    assert_equal 1, params[:means].data.shape[0]
  end

  def test_refinement_masks_match_golden_data
    fixture = golden("strategy_default_masks")
    count = fixture.fetch("scales").shape[0]
    params, = scene_params(count)
    params[:scales].data[] = fixture.fetch("scales")
    state = {
      scene_scale: 1.0,
      grad2d: fixture.fetch("grad2d"),
      count: fixture.fetch("observations"),
      radii: Numo::SFloat.zeros(count)
    }

    duplicate, split = Gsplat::Strategy::Default.new.refinement_masks(params, state)

    assert_equal fixture.fetch("duplicate_mask").to_a, duplicate.to_a
    assert_equal fixture.fetch("split_mask").to_a, split.to_a
  end

  private

  def scene_params(count)
    params = {
      means: variable(Numo::DFloat.zeros(count, 3)),
      quats: variable(identity_quaternions(count)),
      scales: variable(Numo::DFloat.zeros(count, 3)),
      opacities: variable(Numo::DFloat.zeros(count)),
      sh0: variable(Numo::DFloat.zeros(count, 1, 3)),
      shN: variable(Numo::DFloat.zeros(count, 3, 3))
    }
    optimizers = params.transform_values { |parameter| Gsplat::Optim::Adam.new(parameter) }
    [params, optimizers]
  end

  def identity_quaternions(count)
    output = Numo::DFloat.zeros(count, 4)
    output[true, 0] = 1
    output
  end

  def variable(value)
    Gsplat::Autograd::Variable.new(value, requires_grad: true)
  end
end
