import numpy as np
import pytest

from corefin.assumptions.schema import DRIVER_ORDER, RootConfig
from corefin.scenarios.drivers import DriverSet
from corefin.scenarios.generator import (
    broadcast_drivers,
    deterministic_drivers,
    generate_correlated_shocks,
    generate_stochastic_drivers,
)
from corefin.timeline import Timeline
from tests.conftest import minimal_config_dict


def _config(n_periods: int = 4, **overrides) -> RootConfig:
    data = minimal_config_dict(n_periods=n_periods)
    data["scenario"].update(overrides)
    return RootConfig.model_validate(data)


def test_driver_set_rejects_mismatched_shapes():
    with pytest.raises(ValueError, match="ebitda_margin"):
        DriverSet(
            revenue_growth=np.zeros((2, 3)),
            ebitda_margin=np.zeros((2, 4)),
            capex_pct_revenue=np.zeros((2, 3)),
            nwc_pct_revenue=np.zeros((2, 3)),
            base_rate=np.zeros((2, 3)),
            exit_multiple=np.zeros(2),
        )


def test_driver_set_rejects_bad_exit_multiple_shape():
    with pytest.raises(ValueError, match="exit_multiple"):
        DriverSet(
            revenue_growth=np.zeros((2, 3)),
            ebitda_margin=np.zeros((2, 3)),
            capex_pct_revenue=np.zeros((2, 3)),
            nwc_pct_revenue=np.zeros((2, 3)),
            base_rate=np.zeros((2, 3)),
            exit_multiple=np.zeros(3),
        )


def test_deterministic_drivers_shapes_and_values():
    config = _config(n_periods=4)
    timeline = Timeline.annual(n_periods=4)
    drivers = deterministic_drivers(config, timeline)
    assert drivers.n_scenarios == 1
    assert drivers.n_periods == 4
    assert np.allclose(drivers.revenue_growth, 0.05)
    assert np.allclose(drivers.ebitda_margin, 0.20)
    assert drivers.exit_multiple.shape == (1,)
    assert np.allclose(drivers.exit_multiple, 8.0)


def test_broadcast_drivers_tiles_identically():
    config = _config(n_periods=3)
    timeline = Timeline.annual(n_periods=3)
    base = deterministic_drivers(config, timeline)
    tiled = broadcast_drivers(base, n_scenarios=100)
    assert tiled.n_scenarios == 100
    assert np.all(tiled.revenue_growth == tiled.revenue_growth[0])
    assert np.all(tiled.exit_multiple == tiled.exit_multiple[0])


def test_broadcast_drivers_rejects_already_multi_scenario():
    config = _config(n_periods=3)
    timeline = Timeline.annual(n_periods=3)
    base = deterministic_drivers(config, timeline)
    tiled = broadcast_drivers(base, n_scenarios=5)
    with pytest.raises(ValueError, match="single-scenario"):
        broadcast_drivers(tiled, n_scenarios=2)


def test_zero_vol_stochastic_matches_deterministic():
    config = _config(n_periods=4, n_scenarios=50)
    timeline = Timeline.annual(n_periods=4)
    deterministic = broadcast_drivers(deterministic_drivers(config, timeline), 50)
    stochastic = generate_stochastic_drivers(config, timeline, seed=1)
    assert np.allclose(stochastic.revenue_growth, deterministic.revenue_growth)
    assert np.allclose(stochastic.ebitda_margin, deterministic.ebitda_margin)
    assert np.allclose(stochastic.exit_multiple, deterministic.exit_multiple)


def test_positive_vol_produces_dispersion():
    config = _config(
        n_periods=4,
        n_scenarios=2000,
        driver_vol={"revenue_growth_std": 0.05, "exit_multiple_std": 1.0},
    )
    timeline = Timeline.annual(n_periods=4)
    stochastic = generate_stochastic_drivers(config, timeline, seed=7)
    assert stochastic.revenue_growth.std() > 0.02
    assert stochastic.exit_multiple.std() > 0.3
    assert np.isclose(stochastic.revenue_growth.mean(), 0.05, atol=0.01)


def test_seed_reproducibility():
    config = _config(n_periods=4, n_scenarios=200, driver_vol={"revenue_growth_std": 0.05})
    timeline = Timeline.annual(n_periods=4)
    a = generate_stochastic_drivers(config, timeline, seed=123)
    b = generate_stochastic_drivers(config, timeline, seed=123)
    assert np.array_equal(a.revenue_growth, b.revenue_growth)


def test_correlation_matrix_shape_validated():
    data = minimal_config_dict(n_periods=3)
    data["scenario"]["driver_correlation"] = [[1.0, 0.0], [0.0, 1.0]]
    with pytest.raises(Exception, match="driver_correlation must be"):
        RootConfig.model_validate(data)


def test_advanced_defaults_to_none():
    config = _config(n_periods=3)
    assert config.scenario.advanced is None


def test_simple_mode_matches_documented_formula_when_advanced_unset():
    """Characterization test pinning generate_stochastic_drivers' simple-mode
    (scenario.advanced unset) output to the pre-existing i.i.d.-shock
    formula, built from the same already-tested primitives -- so advanced-mode
    work in this module can never accidentally change corefin run/optimize."""
    config = _config(
        n_periods=5,
        n_scenarios=50,
        driver_vol={"revenue_growth_std": 0.04, "base_rate_std": 0.01, "exit_multiple_std": 0.5},
    )
    assert config.scenario.advanced is None
    timeline = Timeline.annual(n_periods=5)
    drivers = generate_stochastic_drivers(config, timeline, seed=17)

    base = broadcast_drivers(deterministic_drivers(config, timeline), 50)
    shocks = generate_correlated_shocks(config, timeline, 50, seed=17)
    vol = config.scenario.driver_vol
    stds = np.array(
        [
            vol.revenue_growth_std,
            vol.ebitda_margin_std,
            vol.capex_pct_revenue_std,
            vol.nwc_pct_revenue_std,
            vol.exit_multiple_std,
            vol.base_rate_std,
        ]
    )
    scaled = shocks * stds
    idx = {name: i for i, name in enumerate(DRIVER_ORDER)}

    np.testing.assert_array_equal(
        drivers.revenue_growth, base.revenue_growth + scaled[:, :, idx["revenue_growth"]]
    )
    np.testing.assert_array_equal(
        drivers.ebitda_margin, base.ebitda_margin + scaled[:, :, idx["ebitda_margin"]]
    )
    np.testing.assert_array_equal(
        drivers.capex_pct_revenue,
        base.capex_pct_revenue + scaled[:, :, idx["capex_pct_revenue"]],
    )
    np.testing.assert_array_equal(
        drivers.nwc_pct_revenue, base.nwc_pct_revenue + scaled[:, :, idx["nwc_pct_revenue"]]
    )
    np.testing.assert_array_equal(
        drivers.base_rate, base.base_rate + scaled[:, :, idx["base_rate"]]
    )
    np.testing.assert_array_equal(
        drivers.exit_multiple, base.exit_multiple + scaled[:, -1, idx["exit_multiple"]]
    )


def test_recession_regime_zero_probability_means_no_recessions():
    config = _config(n_periods=6, n_scenarios=500, advanced={"regime": {"annual_probability": 0.0}})
    timeline = Timeline.annual(n_periods=6)
    drivers = generate_stochastic_drivers(config, timeline, seed=1)
    deterministic = broadcast_drivers(deterministic_drivers(config, timeline), 500)
    np.testing.assert_allclose(drivers.revenue_growth, deterministic.revenue_growth)


def test_recession_regime_probability_one_means_every_scenario_has_one():
    config = _config(
        n_periods=6,
        n_scenarios=200,
        advanced={
            "regime": {
                "annual_probability": 1.0,
                "duration_years": 2,
                "revenue_growth_hit": -0.10,
            }
        },
    )
    timeline = Timeline.annual(n_periods=6)
    drivers = generate_stochastic_drivers(config, timeline, seed=2)
    deterministic = broadcast_drivers(deterministic_drivers(config, timeline), 200)
    hit_applied = np.isclose(drivers.revenue_growth, deterministic.revenue_growth - 0.10)
    assert np.all(np.any(hit_applied, axis=1))  # every scenario has at least one recession period


def test_recession_regime_frequency_matches_configured_probability():
    p = 0.2
    config = _config(
        n_periods=10,
        n_scenarios=3000,
        advanced={
            "regime": {"annual_probability": p, "duration_years": 1, "revenue_growth_hit": -0.10}
        },
    )
    timeline = Timeline.annual(n_periods=10)
    drivers = generate_stochastic_drivers(config, timeline, seed=3)
    deterministic = broadcast_drivers(deterministic_drivers(config, timeline), 3000)
    # duration_years=1 -- each recession is a single isolated period, so the
    # fraction of hit periods is directly comparable to the per-period
    # start probability (no multi-period compounding to account for).
    hit_applied = np.isclose(drivers.revenue_growth, deterministic.revenue_growth - 0.10)
    assert abs(hit_applied.mean() - p) < 0.02


def test_ar1_persistence_matches_configured_phi_on_large_sample():
    phi = 0.6
    n_periods = 20
    config = _config(
        n_periods=n_periods,
        n_scenarios=20000,
        driver_vol={"revenue_growth_std": 0.05},
        advanced={"persistence": {"revenue_growth_phi": phi}},
    )
    timeline = Timeline.annual(n_periods=n_periods)
    drivers = generate_stochastic_drivers(config, timeline, seed=11)
    deterministic = broadcast_drivers(deterministic_drivers(config, timeline), 20000)
    shock = drivers.revenue_growth - deterministic.revenue_growth
    # Lag-1 autocorrelation between the last two periods, well past the
    # transient where the process hasn't yet reached its stationary
    # variance (early-period empirical autocorrelation is biased low).
    corr = np.corrcoef(shock[:, n_periods - 2], shock[:, n_periods - 1])[0, 1]
    assert abs(corr - phi) < 0.05


def test_mean_reverting_base_rate_converges_toward_long_run_mean():
    long_run_mean = 0.06
    entry_rate = 0.02
    n_periods = 30
    config = _config(
        n_periods=n_periods,
        n_scenarios=10,
        base_rate=entry_rate,
        driver_vol={"base_rate_std": 0.0},
        advanced={"rate_mean_reversion": {"kappa": 0.3, "long_run_mean": long_run_mean}},
    )
    timeline = Timeline.annual(n_periods=n_periods)
    drivers = generate_stochastic_drivers(config, timeline, seed=5)
    assert drivers.base_rate[0, 0] == pytest.approx(entry_rate)  # period 0 anchored, no reversion
    initial_gap = abs(entry_rate - long_run_mean)
    final_gap = abs(drivers.base_rate[0, -1] - long_run_mean)
    assert final_gap < initial_gap
    assert final_gap < 0.001  # effectively converged after 30 periods at kappa=0.3


def test_exit_multiple_link_higher_growth_gives_higher_multiple():
    def _make(growth: float) -> RootConfig:
        data = minimal_config_dict(n_periods=5)
        data["company"]["revenue_growth"] = growth
        data["scenario"].update(
            {
                "n_scenarios": 1,
                "advanced": {
                    "exit_multiple_link": {
                        "beta_growth": 5.0,
                        "reference_cagr": 0.05,
                        "beta_rate": 0.0,
                        "reference_rate": 0.0,
                    }
                },
            }
        )
        return RootConfig.model_validate(data)

    timeline = Timeline.annual(n_periods=5)
    low_growth = generate_stochastic_drivers(_make(0.02), timeline, seed=1)
    high_growth = generate_stochastic_drivers(_make(0.10), timeline, seed=1)
    assert high_growth.exit_multiple[0] > low_growth.exit_multiple[0]


def test_exit_multiple_link_higher_rate_gives_lower_multiple():
    def _make(base_rate: float) -> RootConfig:
        data = minimal_config_dict(n_periods=5)
        data["scenario"].update(
            {
                "n_scenarios": 1,
                "base_rate": base_rate,
                "advanced": {
                    "exit_multiple_link": {
                        "beta_growth": 0.0,
                        "beta_rate": -4.0,
                        "reference_rate": 0.03,
                    }
                },
            }
        )
        return RootConfig.model_validate(data)

    timeline = Timeline.annual(n_periods=5)
    low_rate = generate_stochastic_drivers(_make(0.02), timeline, seed=1)
    high_rate = generate_stochastic_drivers(_make(0.08), timeline, seed=1)
    assert high_rate.exit_multiple[0] < low_rate.exit_multiple[0]


def test_advanced_mode_seed_reproducibility():
    config = _config(
        n_periods=6,
        n_scenarios=300,
        driver_vol={"revenue_growth_std": 0.03, "base_rate_std": 0.01, "exit_multiple_std": 0.5},
        advanced={
            "persistence": {"revenue_growth_phi": 0.5},
            "regime": {"annual_probability": 0.1},
            "fat_tails": {"degrees_of_freedom": 5.0},
            "rate_mean_reversion": {"kappa": 0.2, "long_run_mean": 0.05},
            "exit_multiple_link": {"beta_growth": 3.0, "beta_rate": -2.0},
        },
    )
    timeline = Timeline.annual(n_periods=6)
    a = generate_stochastic_drivers(config, timeline, seed=99)
    b = generate_stochastic_drivers(config, timeline, seed=99)
    for field in (
        "revenue_growth",
        "ebitda_margin",
        "capex_pct_revenue",
        "nwc_pct_revenue",
        "base_rate",
        "exit_multiple",
    ):
        np.testing.assert_array_equal(getattr(a, field), getattr(b, field))
