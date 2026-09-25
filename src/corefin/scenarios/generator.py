"""Deterministic base case + a simple correlated-shock generator, plus an
opt-in richer Monte Carlo generator for the Sponsor LBO Monte Carlo engine.

Three layers: (1) a deterministic single-scenario DriverSet built straight
from the YAML assumptions, (2) broadcasting that base case to many identical
scenarios, and (3) stochastic generation -- either the original "simple"
mode (i.i.d. per-period normal shocks correlated across drivers via a
Cholesky factor) or, when `scenario.advanced` is configured, a richer mode
with AR(1) persistence, recession regimes, fat-tailed innovations, a
mean-reverting base rate, and a fundamentals-linked exit multiple.
`generate_stochastic_drivers` dispatches between the two; the simple path's
code is untouched by the advanced-mode addition, so existing callers
(`corefin run`, `corefin optimize`) are unaffected unless a config
explicitly adds an `advanced:` block under `scenario:`.
"""

from __future__ import annotations

import numpy as np

from corefin.assumptions.loader import expand_series
from corefin.assumptions.schema import (
    DRIVER_ORDER,
    AdvancedScenarioConfig,
    ExitMultipleLinkConfig,
    FatTailsConfig,
    RateMeanReversionConfig,
    RecessionRegimeConfig,
    RootConfig,
)
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


def _correlated_normal_innovations(
    rng: np.random.Generator, correlation: np.ndarray, n_scenarios: int, n_periods: int
) -> np.ndarray:
    chol = np.linalg.cholesky(correlation)
    z = rng.standard_normal(size=(n_scenarios, n_periods, correlation.shape[0]))
    return z @ chol.T


def _correlation_matrix(config: RootConfig) -> np.ndarray:
    n_drivers = len(DRIVER_ORDER)
    return (
        np.asarray(config.scenario.driver_correlation)
        if config.scenario.driver_correlation is not None
        else np.eye(n_drivers)
    )


def generate_correlated_shocks(
    config: RootConfig,
    timeline: Timeline,
    n_scenarios: int,
    seed: int | None = None,
) -> np.ndarray:
    """Standard-normal draws correlated across the 6 drivers in DRIVER_ORDER,
    i.i.d. across scenarios and periods. Shape (n_scenarios, n_periods, 6)."""
    rng = np.random.default_rng(seed if seed is not None else config.scenario.random_seed)
    return _correlated_normal_innovations(
        rng, _correlation_matrix(config), n_scenarios, timeline.n_periods
    )


def _generate_simple_stochastic_drivers(
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


def _draw_innovations(
    rng: np.random.Generator,
    correlation: np.ndarray,
    n_scenarios: int,
    n_periods: int,
    fat_tails: FatTailsConfig | None,
) -> np.ndarray:
    """(n_scenarios, n_periods, 6), correlated across drivers, unit-variance
    per component. Normal by default; Student-t (variance-normalized so
    driver_vol's std fields keep their meaning) when fat_tails is set."""
    if fat_tails is None:
        return _correlated_normal_innovations(rng, correlation, n_scenarios, n_periods)
    df = fat_tails.degrees_of_freedom
    chol = np.linalg.cholesky(correlation)
    raw = rng.standard_t(df, size=(n_scenarios, n_periods, correlation.shape[0]))
    raw = raw / np.sqrt(df / (df - 2.0))
    return raw @ chol.T


def _apply_persistence(base: np.ndarray, innovation: np.ndarray, phi: float) -> np.ndarray:
    """base + an AR(1) shock path: shock_t = phi*shock_{t-1} + innovation_t
    (shock_{-1}=0). phi=0 reproduces the simple additive-noise behavior
    exactly (shock_t = innovation_t)."""
    if phi == 0.0:
        return base + innovation
    n_scenarios, n_periods = innovation.shape
    shock = np.zeros_like(innovation)
    prev = np.zeros(n_scenarios)
    for t in range(n_periods):
        prev = phi * prev + innovation[:, t]
        shock[:, t] = prev
    return base + shock


def _draw_recession_mask(
    rng: np.random.Generator, n_scenarios: int, n_periods: int, regime: RecessionRegimeConfig
) -> np.ndarray:
    """(n_scenarios, n_periods) bool: True in periods where the scenario is
    in an active recession. Non-overlapping -- a new recession can only
    start once the previous one (if any) has run its full duration."""
    mask = np.zeros((n_scenarios, n_periods), dtype=bool)
    remaining = np.zeros(n_scenarios, dtype=int)
    starts = rng.random(size=(n_scenarios, n_periods)) < regime.annual_probability
    for t in range(n_periods):
        can_start = remaining <= 0
        new_start = can_start & starts[:, t]
        remaining = np.where(new_start, regime.duration_years, remaining)
        in_recession = remaining > 0
        mask[:, t] = in_recession
        remaining = np.where(in_recession, remaining - 1, remaining)
    return mask


def _apply_rate_mean_reversion(
    entry_rate: np.ndarray, innovation: np.ndarray, config: RateMeanReversionConfig
) -> np.ndarray:
    """Period 0 is anchored to the deterministic entry rate plus its own
    shock (same as simple mode); periods 1.. evolve via rate_t = rate_{t-1}
    + kappa*(long_run_mean - rate_{t-1}) + innovation_t."""
    n_scenarios, n_periods = innovation.shape
    rate = np.zeros_like(innovation)
    rate[:, 0] = entry_rate[:, 0] + innovation[:, 0]
    for t in range(1, n_periods):
        rate[:, t] = (
            rate[:, t - 1]
            + config.kappa * (config.long_run_mean - rate[:, t - 1])
            + innovation[:, t]
        )
    return rate


def _compute_linked_exit_multiple(
    config: RootConfig,
    timeline: Timeline,
    revenue_growth: np.ndarray,
    ebitda_margin: np.ndarray,
    base_rate: np.ndarray,
    exit_multiple_std: float,
    link: ExitMultipleLinkConfig,
    rng: np.random.Generator,
    fat_tails: FatTailsConfig | None,
) -> np.ndarray:
    exit_idx = config.transaction.exit_year_index
    revenue = config.company.revenue_base_mm * np.cumprod(1.0 + revenue_growth, axis=1)
    exit_ebitda = revenue[:, exit_idx] * ebitda_margin[:, exit_idx]
    entry_margin = expand_series(config.company.ebitda_margin, timeline.n_periods, "ebitda_margin")[
        0
    ]
    entry_ebitda = config.company.revenue_base_mm * entry_margin
    n_years = exit_idx + 1
    cagr = (exit_ebitda / entry_ebitda) ** (1.0 / n_years) - 1.0
    exit_rate = base_rate[:, exit_idx]

    n_scenarios = exit_ebitda.shape[0]
    if fat_tails is not None:
        df = fat_tails.degrees_of_freedom
        noise = rng.standard_t(df, size=n_scenarios) / np.sqrt(df / (df - 2.0))
    else:
        noise = rng.standard_normal(size=n_scenarios)

    return (
        config.scenario.exit_multiple
        + link.beta_growth * (cagr - link.reference_cagr)
        + link.beta_rate * (exit_rate - link.reference_rate)
        + noise * exit_multiple_std
    )


def _generate_advanced_stochastic_drivers(
    config: RootConfig,
    timeline: Timeline,
    advanced: AdvancedScenarioConfig,
    seed: int | None = None,
) -> DriverSet:
    n_scenarios = config.scenario.n_scenarios
    n_periods = timeline.n_periods
    base = broadcast_drivers(deterministic_drivers(config, timeline), n_scenarios)
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
    idx = {name: i for i, name in enumerate(DRIVER_ORDER)}

    rng = np.random.default_rng(seed if seed is not None else config.scenario.random_seed)
    innovations = _draw_innovations(
        rng, _correlation_matrix(config), n_scenarios, n_periods, advanced.fat_tails
    )
    scaled = innovations * stds

    persistence = advanced.persistence
    revenue_growth = _apply_persistence(
        base.revenue_growth,
        scaled[:, :, idx["revenue_growth"]],
        persistence.revenue_growth_phi if persistence else 0.0,
    )
    ebitda_margin = _apply_persistence(
        base.ebitda_margin,
        scaled[:, :, idx["ebitda_margin"]],
        persistence.ebitda_margin_phi if persistence else 0.0,
    )
    capex_pct_revenue = base.capex_pct_revenue + scaled[:, :, idx["capex_pct_revenue"]]
    nwc_pct_revenue = base.nwc_pct_revenue + scaled[:, :, idx["nwc_pct_revenue"]]

    if advanced.regime is not None:
        recession = _draw_recession_mask(rng, n_scenarios, n_periods, advanced.regime)
        revenue_growth = revenue_growth + recession * advanced.regime.revenue_growth_hit
        ebitda_margin = ebitda_margin + recession * advanced.regime.ebitda_margin_hit

    if advanced.rate_mean_reversion is not None:
        base_rate = _apply_rate_mean_reversion(
            base.base_rate, scaled[:, :, idx["base_rate"]], advanced.rate_mean_reversion
        )
    else:
        base_rate = base.base_rate + scaled[:, :, idx["base_rate"]]

    if advanced.exit_multiple_link is not None:
        exit_multiple = _compute_linked_exit_multiple(
            config,
            timeline,
            revenue_growth,
            ebitda_margin,
            base_rate,
            vol.exit_multiple_std,
            advanced.exit_multiple_link,
            rng,
            advanced.fat_tails,
        )
    else:
        exit_multiple = base.exit_multiple + scaled[:, -1, idx["exit_multiple"]]

    return DriverSet(
        revenue_growth=revenue_growth,
        ebitda_margin=ebitda_margin,
        capex_pct_revenue=capex_pct_revenue,
        nwc_pct_revenue=nwc_pct_revenue,
        base_rate=base_rate,
        exit_multiple=exit_multiple,
    )


def generate_stochastic_drivers(
    config: RootConfig,
    timeline: Timeline,
    seed: int | None = None,
) -> DriverSet:
    """Dispatches to the simple i.i.d.-shock generator, or -- only when
    `config.scenario.advanced` is set -- the richer Monte Carlo generator
    (persistence, recession regimes, fat tails, mean-reverting rates,
    fundamentals-linked exit multiple). The simple path is untouched code,
    so every existing caller (corefin run/optimize) is unaffected unless a
    config explicitly opts into `scenario.advanced`."""
    if config.scenario.advanced is None:
        return _generate_simple_stochastic_drivers(config, timeline, seed)
    return _generate_advanced_stochastic_drivers(config, timeline, config.scenario.advanced, seed)
