# frozen_string_literal: true

require "test_helper"

class MCMCStrategyTest < Minitest::Test
  def setup
    @previous_backend = Gsplat.backend
    Gsplat.backend = :ruby
  end

  def teardown
    Gsplat.backend = @previous_backend
  end

  def test_relocates_dead_gaussians_and_grows_five_percent
    params, optimizers = scene_params(20)
    params[:opacities].data[0] = -20
    strategy = Gsplat::Strategy::MCMC.new(
      refine_start_iter: 0,
      refine_stop_iter: 10,
      refine_every: 1,
      noise_lr: 0,
      cap_max: 30
    )
    state = strategy.initialize_state(scene_scale: 1.0)

    strategy.step_post_backward(
      params: params,
      optimizers: optimizers,
      state: state,
      step: 1,
      info: {},
      lr: 1e-3
    )

    assert_equal 21, params[:means].data.shape[0]
    assert_operator sigmoid(params[:opacities].data[0]), :>=, strategy.min_opacity
    params.each_key { |key| assert_equal 21, optimizers[key].state.exp_avg.shape[0] }
  end

  def test_position_noise_has_expected_mean_and_variance
    count = 20_000
    params, = scene_params(count)
    params[:opacities].data[] = -20
    params[:scales].data[] = ::Math.log(0.5)
    scaler = 0.2

    Gsplat::Strategy::Ops.inject_position_noise!(
      params,
      scaler: scaler,
      rng: Random.new(91)
    )

    samples = params[:means].data
    gate = 1.0 / (1.0 + ::Math.exp(-100.0 * ((1.0 - sigmoid(-20)) - 0.995)))
    expected_variance = ((0.5**2) * scaler * gate)**2
    3.times do |axis|
      mean, variance = mean_and_variance(samples[true, axis])
      assert_in_delta 0.0, mean, 8e-4
      assert_in_delta expected_variance, variance, expected_variance * 0.06
    end
  end

  def test_noise_free_mcmc_preserves_render_and_optimizer_alignment
    params, optimizers = scene_params(20)
    strategy = Gsplat::Strategy::MCMC.new(
      refine_start_iter: 0,
      refine_stop_iter: 5,
      refine_every: 1,
      noise_lr: 0,
      min_opacity: 1e-6,
      cap_max: 21
    )
    state = strategy.initialize_state(scene_scale: 1.0)
    before = params[:sh0].data.mean.to_f

    3.times do |step|
      strategy.step_post_backward(
        params: params,
        optimizers: optimizers,
        state: state,
        step: step + 1,
        info: {},
        lr: 1e-3
      )
    end

    assert_equal 21, params[:means].data.shape[0]
    assert_in_delta before, params[:sh0].data.mean.to_f, 1e-12
    params.each_key { |key| assert_equal params[key].data.shape, optimizers[key].state.exp_avg.shape }
  end

  def test_noise_free_mcmc_image_fit_converges
    params, optimizers = fitting_scene
    target_colors = Numo::DFloat.zeros(16, 3)
    16.times { |index| target_colors[index, true] = [index.fdiv(15), 0.25, 1 - index.fdiv(15)] }
    target = Gsplat::Ops::TensorOps.data(render(params, target_colors))
    strategy = Gsplat::Strategy::MCMC.new(
      cap_max: 16,
      noise_lr: 0,
      refine_start_iter: 0,
      refine_stop_iter: 10,
      refine_every: 1,
      min_opacity: 1e-6
    )
    state = strategy.initialize_state(scene_scale: 1.0)
    initial = psnr(render(params, params[:sh0]), target)
    run_fit!(params, optimizers, strategy, state, target)

    assert_operator psnr(render(params, params[:sh0]), target), :>, initial
    assert_equal 16, params[:means].data.shape[0]
  end

  private

  def fitting_scene
    params, optimizers = scene_params(16)
    focal = 8.0
    16.times do |index|
      x_coord = ((index % 4) * 2) + 0.5
      y_coord = ((index / 4) * 2) + 0.5
      params[:means].data[index, true] = [(x_coord - 4) * 2 / focal, (y_coord - 4) * 2 / focal, 2]
    end
    params[:scales].data[] = ::Math.log(0.2)
    params[:opacities].data[] = ::Math.log(0.85 / 0.15)
    params[:sh0].replace_data!(Numo::DFloat.ones(16, 3) * 0.5)
    [params, optimizers]
  end

  def run_fit!(params, optimizers, strategy, state, target)
    8.times do |index|
      rendered = render(params, params[:sh0])
      Gsplat::Training::ImageFitter::MeanSquaredError.apply(rendered, target).backward
      strategy.step_post_backward(
        params: params, optimizers: optimizers, state: state, step: index + 1, info: {}, lr: 1e-3
      )
      params[:sh0].data[] = clipped(params[:sh0].data - (20 * params[:sh0].grad))
      params.each_value(&:zero_grad!)
    end
  end

  def render(params, colors)
    view = Numo::DFloat.zeros(1, 4, 4)
    view[0, true, true] = Numo::DFloat.eye(4)
    intrinsics = Numo::DFloat[[[8, 0, 4], [0, 8, 4], [0, 0, 1]]]
    Gsplat.rasterization(
      means: params[:means],
      quats: params[:quats].data,
      scales: Numo::NMath.exp(params[:scales].data),
      opacities: sigmoid_array(params[:opacities].data),
      colors: colors,
      viewmats: view,
      ks: intrinsics,
      width: 8,
      height: 8
    ).first
  end

  def psnr(rendered, target)
    values = Gsplat::Ops::TensorOps.data(rendered)
    mse = ((values - target)**2).mean.to_f
    -10 * ::Math.log10(mse)
  end

  def clipped(values)
    output = values.dup
    output[output.lt(0)] = 0
    output[output.gt(1)] = 1
    output
  end

  def scene_params(count)
    params = {
      means: variable(Numo::DFloat.zeros(count, 3)),
      quats: variable(identity_quaternions(count)),
      scales: variable(Numo::DFloat.zeros(count, 3)),
      opacities: variable(Numo::DFloat.zeros(count)),
      sh0: variable(Numo::DFloat.ones(count, 1, 3) * 0.25),
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

  def sigmoid(value)
    1.0 / (1.0 + ::Math.exp(-value))
  end

  def sigmoid_array(values)
    1.0 / (1.0 + Numo::NMath.exp(-values))
  end

  def mean_and_variance(values)
    mean = values.sum.to_f / values.size
    variance = ((values - mean)**2).sum.to_f / values.size
    [mean, variance]
  end
end
