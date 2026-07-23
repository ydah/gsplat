# frozen_string_literal: true

require "test_helper"

class TrainerTest < Minitest::Test
  def setup
    @previous_backend = Gsplat.backend
    Gsplat.backend = :ruby
  end

  def teardown
    Gsplat.backend = @previous_backend
  end

  def test_synthetic_eight_view_training_reaches_target_psnr
    scene, params = synthetic_scene
    config = Gsplat::Training::Config.new(
      max_steps: 80,
      batch_size: 2,
      sh_degree: 0,
      sh_degree_interval: 10,
      means_lr: 1e-8,
      means_lr_final: 1e-8,
      scales_lr: 1e-8,
      quats_lr: 1e-8,
      opacities_lr: 1e-8,
      sh0_lr: 0.08,
      shN_lr: 1e-8,
      eval_steps: [],
      save_steps: [],
      log_every: 1_000
    )
    strategy = Gsplat::Strategy::Default.new(refine_start_iter: 1_000)
    trainer = Gsplat::Training::Trainer.new(
      scene: scene, config: config, params: params, strategy: strategy
    )

    result = trainer.train

    assert_equal 80, result.step
    assert_operator result.final_metrics[:psnr], :>, result.initial_metrics[:psnr]
    assert_operator result.final_metrics[:psnr], :>=, 28.0
  end

  def test_config_rejects_unknown_options
    error = assert_raises(ArgumentError) do
      Gsplat::Training::Config.new(unknown: true)
    end

    assert_includes error.message, "unknown"
  end

  private

  def synthetic_scene
    params = raw_params
    teacher_colors = Numo::DFloat.zeros(16, 3)
    16.times do |index|
      fraction = index.fdiv(15)
      teacher_colors[index, true] = [fraction, 0.2 + (0.5 * fraction), 1 - fraction]
    end
    teacher_sh = Gsplat::Utils.rgb_to_sh(teacher_colors).reshape(16, 1, 3)
    views, intrinsics = cameras
    images, = Gsplat.rasterization(
      means: params[:means].data,
      quats: params[:quats].data,
      scales: Numo::NMath.exp(params[:scales].data),
      opacities: sigmoid(params[:opacities].data),
      colors: teacher_sh,
      viewmats: views,
      ks: intrinsics,
      width: 8,
      height: 8,
      sh_degree: 0
    )
    scene = Gsplat::Training::Scene.new(
      viewmats: views,
      intrinsics: intrinsics,
      images: images,
      points: params[:means].data.dup,
      colors: teacher_colors,
      scene_scale: 1.0
    )
    [scene, params]
  end

  def raw_params
    means = Numo::DFloat.zeros(16, 3)
    16.times do |index|
      pixel_x = ((index % 4) * 2) + 0.5
      pixel_y = ((index / 4) * 2) + 0.5
      means[index, true] = [(pixel_x - 4) / 4.0, (pixel_y - 4) / 4.0, 2]
    end
    {
      means: variable(means),
      quats: variable(identity_quaternions(16)),
      scales: variable(Numo::DFloat.ones(16, 3) * ::Math.log(0.2)),
      opacities: variable(Numo::DFloat.ones(16) * ::Math.log(0.85 / 0.15)),
      sh0: variable(Numo::DFloat.zeros(16, 1, 3)),
      shN: variable(Numo::DFloat.zeros(16, 0, 3))
    }
  end

  def cameras
    views = Numo::DFloat.zeros(8, 4, 4)
    intrinsics = Numo::DFloat.zeros(8, 3, 3)
    8.times do |index|
      views[index, true, true] = Numo::DFloat.eye(4)
      views[index, 0, 3] = (index - 3.5) * 0.02
      intrinsics[index, true, true] = Numo::DFloat[[8, 0, 4], [0, 8, 4], [0, 0, 1]]
    end
    [views, intrinsics]
  end

  def identity_quaternions(count)
    values = Numo::DFloat.zeros(count, 4)
    values[true, 0] = 1
    values
  end

  def variable(values)
    Gsplat::Autograd::Variable.new(values, requires_grad: true)
  end

  def sigmoid(values)
    1.0 / (1.0 + Numo::NMath.exp(-values))
  end
end
