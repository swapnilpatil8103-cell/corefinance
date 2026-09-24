import numpy as np
import pytest

from corefin.assumptions.schema import RootConfig
from corefin.scenarios.drivers import DriverSet
from corefin.scenarios.generator import (
    broadcast_drivers,
    deterministic_drivers,
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
