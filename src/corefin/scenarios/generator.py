"""Deterministic base case + a simple correlated-shock generator.

The full Monte Carlo engine (time-series dynamics, regime switching, fat
tails, etc.) belongs to the downstream Sponsor LBO Monte Carlo project. This
module only provides: (1) a deterministic single-scenario DriverSet built
straight from the YAML assumptions, (2) broadcasting that base case to many
identical scenarios (useful for testing vectorization), and (3) i.i.d.
per-period normal shocks correlated across drivers via a Cholesky factor, as
a minimal reference implementation of the stochastic interface.
"""

from __future__ import annotations

import numpy as np

from corefin.assumptions.loader import expand_series
from corefin.assumptions.schema import DRIVER_ORDER, RootConfig
from corefin.scenarios.drivers import DriverSet
from corefin.timeline import Timeline


def deterministic_drivers(config: RootConfig, timeline: Timeline) -> DriverSet:
    n = timeline.n_periods
    company = config.company
    scenario = config.scenario
    return DriverSet(
        revenue_growth=expand_series(company.revenue_growth, n, "revenue_growth").reshape(1, n),
        ebitda_margin=expand_series(company.ebitda_margin, n, "ebitda_margin").reshape(1, n),
        capex_pct_revenue=expand_series(company.capex_pct_revenue, n, "capex_pct_revenue").reshape(
            1, n
        ),
        nwc_pct_revenue=expand_series(company.nwc_pct_revenue, n, "nwc_pct_revenue").reshape(1, n),
        base_rate=expand_series(scenario.base_rate, n, "base_rate").reshape(1, n),
        exit_multiple=np.array([scenario.exit_multiple]),
    )


def broadcast_drivers(base: DriverSet, n_scenarios: int) -> DriverSet:
    if base.n_scenarios != 1:
        raise ValueError("broadcast_drivers expects a single-scenario DriverSet as input")
    return DriverSet(
        revenue_growth=np.repeat(base.revenue_growth, n_scenarios, axis=0),
        ebitda_margin=np.repeat(base.ebitda_margin, n_scenarios, axis=0),
        capex_pct_revenue=np.repeat(base.capex_pct_revenue, n_scenarios, axis=0),
        nwc_pct_revenue=np.repeat(base.nwc_pct_revenue, n_scenarios, axis=0),
        base_rate=np.repeat(base.base_rate, n_scenarios, axis=0),
        exit_multiple=np.repeat(base.exit_multiple, n_scenarios, axis=0),
    )


def generate_correlated_shocks(
    config: RootConfig,
    timeline: Timeline,
    n_scenarios: int,
    seed: int | None = None,
) -> np.ndarray:
    """Standard-normal draws correlated across the 6 drivers in DRIVER_ORDER,
    i.i.d. across scenarios and periods. Shape (n_scenarios, n_periods, 6)."""
    n_drivers = len(DRIVER_ORDER)
    correlation = (
        np.asarray(config.scenario.driver_correlation)
        if config.scenario.driver_correlation is not None
        else np.eye(n_drivers)
    )
    chol = np.linalg.cholesky(correlation)
    rng = np.random.default_rng(seed if seed is not None else config.scenario.random_seed)
    z = rng.standard_normal(size=(n_scenarios, timeline.n_periods, n_drivers))
    return z @ chol.T


def generate_stochastic_drivers(
    config: RootConfig,
    timeline: Timeline,
    seed: int | None = None,
) -> DriverSet:
    """Applies additive N(0, std) shocks (per DriverVolConfig) to the deterministic
    base case. exit_multiple is a single per-scenario draw taken from the shocks'
    final period. A std of 0 for a driver reproduces the deterministic value exactly."""
    n_scenarios = config.scenario.n_scenarios
    base = broadcast_drivers(deterministic_drivers(config, timeline), n_scenarios)
    shocks = generate_correlated_shocks(config, timeline, n_scenarios, seed)
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
    return DriverSet(
        revenue_growth=base.revenue_growth + scaled[:, :, idx["revenue_growth"]],
        ebitda_margin=base.ebitda_margin + scaled[:, :, idx["ebitda_margin"]],
        capex_pct_revenue=base.capex_pct_revenue + scaled[:, :, idx["capex_pct_revenue"]],
        nwc_pct_revenue=base.nwc_pct_revenue + scaled[:, :, idx["nwc_pct_revenue"]],
        base_rate=base.base_rate + scaled[:, :, idx["base_rate"]],
        exit_multiple=base.exit_multiple + scaled[:, -1, idx["exit_multiple"]],
    )
