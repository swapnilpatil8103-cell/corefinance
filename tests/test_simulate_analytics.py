import numpy as np
import pytest

from corefin.assumptions.schema import RootConfig
from corefin.metrics.covenants import CovenantResult
from corefin.optimize.structure import build_structure
from corefin.simulate.analytics import (
    compute_convergence_check,
    compute_covenant_breach_analytics,
    compute_distress_analytics,
    compute_downside_analytics,
    compute_path_percentile_bands,
    compute_return_distribution,
    expected_shortfall,
)
from corefin.simulate.engine import run_simulation
from corefin.timeline import Timeline
from tests.conftest import minimal_config_dict


def _config(n_periods: int = 6, **scenario_overrides) -> RootConfig:
    data = minimal_config_dict(n_periods=n_periods)
    data["optimizer"] = {
        "decision_variables": [{"tranche_name": "TLB", "min_multiple": 1.0, "max_multiple": 3.0}],
    }
    data["scenario"].update(scenario_overrides)
    return RootConfig.model_validate(data)


def test_expected_shortfall_at_or_below_percentile_on_random_sample():
    rng = np.random.default_rng(0)
    values = rng.normal(0.15, 0.10, size=5000)
    p5 = np.percentile(values, 5)
    p10 = np.percentile(values, 10)
    assert expected_shortfall(values, 5) <= p5 + 1e-12
    assert expected_shortfall(values, 10) <= p10 + 1e-12


def test_expected_shortfall_matches_hand_calc_on_small_case():
    values = np.array([-0.5, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 1.0])
    # p10 of 10 sorted points via numpy's default (linear) interpolation:
    p10 = np.percentile(values, 10)
    expected = values[values <= p10].mean()
    assert expected_shortfall(values, 10) == pytest.approx(expected)


def test_return_distribution_probabilities_match_direct_computation():
    irr = np.array([-0.3, -0.05, 0.05, 0.1, 0.2, 0.3])
    moic = np.array([0.5, 0.9, 1.0, 1.5, 2.0, 3.0])
    dist = compute_return_distribution(irr, moic, irr_hurdle=0.15)
    assert dist.prob_moic_below_1 == pytest.approx(2 / 6)  # 0.5 and 0.9
    assert dist.prob_irr_below_hurdle == pytest.approx(4 / 6)  # -0.3,-0.05,0.05,0.1
    assert dist.mean_irr == pytest.approx(np.mean(irr))
    assert dist.median_moic == pytest.approx(np.median(moic))


def test_covenant_breach_analytics_on_constructed_case():
    # 3 scenarios x 4 periods. Covenant A breaches scenario 0 at period 2,
    # scenario 1 never. Covenant B breaches scenario 1 at period 1.
    shape = (3, 4)
    breach_a = np.zeros(shape, dtype=bool)
    breach_a[0, 2] = True
    breach_a[0, 3] = True  # stays breached
    breach_b = np.zeros(shape, dtype=bool)
    breach_b[1, 1] = True

    cov_a = CovenantResult(
        name="A",
        metric_value=np.zeros(shape),
        threshold=np.zeros(shape),
        headroom=np.zeros(shape),
        tested=np.ones(shape, dtype=bool),
        breach=breach_a,
    )
    cov_b = CovenantResult(
        name="B",
        metric_value=np.zeros(shape),
        threshold=np.zeros(shape),
        headroom=np.zeros(shape),
        tested=np.ones(shape, dtype=bool),
        breach=breach_b,
    )

    analytics = compute_covenant_breach_analytics([cov_a, cov_b])

    np.testing.assert_allclose(analytics.breach_probability_by_year["A"], breach_a.mean(axis=0))
    np.testing.assert_allclose(analytics.breach_probability_by_year["B"], breach_b.mean(axis=0))
    # combined: scenario 0 breaches periods 2,3; scenario 1 breaches period 1; scenario 2 never.
    expected_combined = np.array([0 / 3, 1 / 3, 1 / 3, 1 / 3])
    np.testing.assert_allclose(analytics.combined_breach_probability_by_year, expected_combined)
    assert analytics.overall_breach_probability == pytest.approx(2 / 3)  # scenarios 0 and 1
    np.testing.assert_allclose(analytics.time_to_first_breach, [2.0, 1.0, np.nan])


def test_covenant_breach_analytics_empty_when_no_covenants():
    analytics = compute_covenant_breach_analytics([])
    assert analytics.overall_breach_probability == 0.0
    assert analytics.breach_probability_by_year == {}


def test_distress_analytics_matches_direct_computation():
    distress = np.array([[False, True, True], [False, False, False], [True, False, False]])
    analytics = compute_distress_analytics(distress)
    np.testing.assert_allclose(analytics.probability_by_year, distress.mean(axis=0))
    assert analytics.overall_probability == pytest.approx(2 / 3)  # scenarios 0 and 2


def test_path_percentile_bands_are_monotonic_across_percentiles():
    rng = np.random.default_rng(1)
    path = rng.normal(4.0, 1.0, size=(2000, 5))
    bands = compute_path_percentile_bands(path)
    for t in range(5):
        values_at_t = [bands.percentiles[p][t] for p in (5, 10, 25, 50, 75, 90, 95)]
        assert values_at_t == sorted(values_at_t)


def test_convergence_check_trace_has_one_point_per_fraction_and_positive_se():
    rng = np.random.default_rng(2)
    irr = rng.normal(0.15, 0.08, size=4000)
    moic = rng.normal(2.5, 0.5, size=4000)
    check = compute_convergence_check(irr, moic)
    assert check.scenario_counts == [1000, 2000, 4000]
    assert len(check.mean_irr_trace) == 3
    assert check.standard_error_irr > 0
    assert check.standard_error_moic > 0


def test_convergence_check_reports_stable_on_large_iid_sample():
    rng = np.random.default_rng(3)
    irr = rng.normal(0.15, 0.05, size=20000)
    moic = rng.normal(2.5, 0.3, size=20000)
    check = compute_convergence_check(irr, moic)
    assert check.is_stable


def test_compute_downside_analytics_end_to_end():
    config = _config(
        driver_vol={
            "revenue_growth_std": 0.05,
            "ebitda_margin_std": 0.02,
            "base_rate_std": 0.01,
            "exit_multiple_std": 0.5,
        }
    )
    timeline = Timeline.annual(n_periods=6)
    candidate = build_structure(config, {"TLB": 2.0})
    result = run_simulation(config, candidate, timeline, n_scenarios=1000, seed=9)
    analytics = compute_downside_analytics(result, irr_hurdle=0.15)

    assert (
        analytics.returns.expected_shortfall_irr_5pct <= analytics.returns.percentiles_irr[5] + 1e-9
    )
    assert analytics.leverage_bands.percentiles[5].shape == (6,)
    assert analytics.liquidity_bands.mean.shape == (6,)
    assert 0.0 <= analytics.distress.overall_probability <= 1.0


def test_more_leverage_raises_mean_irr_and_downside_risk_on_constructed_case():
    """A deliberately unconstrained, low-starting-leverage case where more
    debt is unambiguously good for the sponsor's mean return (cheap debt,
    plenty of headroom) but also unambiguously riskier (worse expected
    shortfall) -- the return/risk trade-off the optimizer's relaxation
    table and this engine both exist to quantify."""
    config = _config(
        driver_vol={"revenue_growth_std": 0.06, "ebitda_margin_std": 0.03, "exit_multiple_std": 0.8}
    )
    timeline = Timeline.annual(n_periods=6)
    low_leverage = build_structure(config, {"TLB": 1.0})
    high_leverage = build_structure(config, {"TLB": 3.0})

    low_result = run_simulation(config, low_leverage, timeline, n_scenarios=3000, seed=42)
    high_result = run_simulation(config, high_leverage, timeline, n_scenarios=3000, seed=42)

    low_dist = compute_return_distribution(
        low_result.exit_result.irr, low_result.exit_result.moic, 0.15
    )
    high_dist = compute_return_distribution(
        high_result.exit_result.irr, high_result.exit_result.moic, 0.15
    )

    assert high_dist.mean_irr > low_dist.mean_irr
    assert high_dist.expected_shortfall_irr_10pct < low_dist.expected_shortfall_irr_10pct
