# frozen_string_literal: true

require "test_helper"

class RelocationTest < Minitest::Test
  def test_binomial_table_contains_exact_coefficients
    table = Gsplat::Ops::Relocation.binomial_table(n_max: 6, dtype: Numo::DFloat)

    assert_equal [6, 6], table.shape
    assert_equal [1, 5, 10, 10, 5, 1], table[5, true].to_a.map(&:to_i)
  end

  def test_ratio_one_is_identity
    opacities = Numo::DFloat[0.01, 0.25, 0.9]
    scales = Numo::DFloat[[0.1, 0.2, 0.3], [1, 2, 3], [0.5, 0.7, 0.9]]
    ratios = Numo::Int32.ones(3)

    relocated_opacities, relocated_scales = Gsplat.relocation(opacities, scales, ratios)

    assert_allclose relocated_opacities, opacities, atol: 1e-12, rtol: 1e-12
    assert_allclose relocated_scales, scales, atol: 1e-12, rtol: 1e-12
  end

  def test_matches_closed_form_for_two_copies
    opacity = 0.36
    expected_opacity = 1 - ::Math.sqrt(1 - opacity)
    denominator = (2 * expected_opacity) - ((expected_opacity**2) / ::Math.sqrt(2))
    expected_scale = opacity / denominator

    output_opacity, output_scale = Gsplat.relocation(
      Numo::DFloat[opacity],
      Numo::DFloat[[1, 2, 3]],
      Numo::Int32[2]
    )

    assert_in_delta expected_opacity, output_opacity[0], 1e-12
    assert_allclose output_scale, Numo::DFloat[[1, 2, 3]] * expected_scale,
                    atol: 1e-12, rtol: 1e-12
  end

  def test_matches_golden_data
    fixture = golden("relocation_mcmc")
    opacities, scales = Gsplat.relocation(
      fixture.fetch("opacities"),
      fixture.fetch("scales"),
      fixture.fetch("ratios"),
      binoms: fixture.fetch("binoms")
    )

    assert_allclose opacities, fixture.fetch("new_opacities"), atol: 1e-5, rtol: 1e-5
    assert_allclose scales, fixture.fetch("new_scales"), atol: 1e-5, rtol: 1e-5
  end
end
