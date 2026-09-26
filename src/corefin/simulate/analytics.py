"""Downside, distress and convergence analytics computed from a
SimulationResult (engine.py): return distributions, expected shortfall,
covenant breach timing, distress probability, leverage/liquidity percentile
bands, and a Monte Carlo standard error / convergence check.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from corefin.metrics.covenants import CovenantResult
from corefin.simulate.engine import SimulationResult

PERCENTILES = (5, 10, 25, 50, 75, 90, 95)


def _percentiles(values: np.ndarray) -> dict[float, float]:
    return {p: float(np.percentile(values, p)) for p in PERCENTILES}


def expected_shortfall(values: np.ndarray, tail_pct: float) -> float:
    """Average value among the worst tail_pct% of scenarios (e.g.
    tail_pct=5 -> average IRR in the worst 5%). Always <= the
    corresponding percentile, since it averages only values at or below
    that percentile's threshold."""
    threshold = np.percentile(values, tail_pct)
    tail = values[values <= threshold]
    return float(np.mean(tail)) if tail.size > 0 else float(threshold)


@dataclass(frozen=True)
class ReturnDistribution:
    mean_irr: float
    median_irr: float
    percentiles_irr: dict[float, float]
    mean_moic: float
    median_moic: float
    percentiles_moic: dict[float, float]
    expected_shortfall_irr_5pct: float
    expected_shortfall_irr_10pct: float
    prob_moic_below_1: float
    prob_irr_below_hurdle: float
    irr_hurdle: float


def compute_return_distribution(
    irr: np.ndarray, moic: np.ndarray, irr_hurdle: float
) -> ReturnDistribution:
    return ReturnDistribution(
        mean_irr=float(np.mean(irr)),
        median_irr=float(np.median(irr)),
        percentiles_irr=_percentiles(irr),
        mean_moic=float(np.mean(moic)),
        median_moic=float(np.median(moic)),
        percentiles_moic=_percentiles(moic),
        expected_shortfall_irr_5pct=expected_shortfall(irr, 5),
        expected_shortfall_irr_10pct=expected_shortfall(irr, 10),
        prob_moic_below_1=float(np.mean(moic < 1.0)),
        prob_irr_below_hurdle=float(np.mean(irr < irr_hurdle)),
        irr_hurdle=irr_hurdle,
    )


@dataclass(frozen=True)
class CovenantBreachAnalytics:
    breach_probability_by_year: dict[str, np.ndarray]
    combined_breach_probability_by_year: np.ndarray
    overall_breach_probability: float
    time_to_first_breach: np.ndarray  # (n_scenarios,) float period index, NaN if never breached
    time_to_first_breach_percentiles: dict[float, float]  # over breaching scenarios only


def compute_covenant_breach_analytics(
    covenant_results: list[CovenantResult],
) -> CovenantBreachAnalytics:
    if not covenant_results:
        return CovenantBreachAnalytics(
            breach_probability_by_year={},
            combined_breach_probability_by_year=np.array([]),
            overall_breach_probability=0.0,
            time_to_first_breach=np.array([]),
            time_to_first_breach_percentiles={},
        )
    breach_by_year = {cov.name: np.mean(cov.breach, axis=0) for cov in covenant_results}
    combined_breach = np.zeros_like(covenant_results[0].breach, dtype=bool)
    for cov in covenant_results:
        combined_breach |= cov.breach
    combined_by_year = np.mean(combined_breach, axis=0)

    has_breach = np.any(combined_breach, axis=1)
    first_idx = np.argmax(combined_breach, axis=1).astype(float)
    time_to_first_breach = np.where(has_breach, first_idx, np.nan)
    overall = float(np.mean(has_breach))
    breaching = time_to_first_breach[has_breach]
    ttfb_percentiles = _percentiles(breaching) if breaching.size > 0 else {}

    return CovenantBreachAnalytics(
        breach_probability_by_year=breach_by_year,
        combined_breach_probability_by_year=combined_by_year,
        overall_breach_probability=overall,
        time_to_first_breach=time_to_first_breach,
        time_to_first_breach_percentiles=ttfb_percentiles,
    )


@dataclass(frozen=True)
class DistressAnalytics:
    probability_by_year: np.ndarray
    overall_probability: float


def compute_distress_analytics(distress_by_year: np.ndarray) -> DistressAnalytics:
    return DistressAnalytics(
        probability_by_year=np.mean(distress_by_year, axis=0),
        overall_probability=float(np.mean(np.any(distress_by_year, axis=1))),
    )


@dataclass(frozen=True)
class PathPercentileBands:
    percentiles: dict[float, np.ndarray]  # {5: (n_periods,), ...}
    mean: np.ndarray


def compute_path_percentile_bands(path: np.ndarray) -> PathPercentileBands:
    return PathPercentileBands(
        percentiles={p: np.percentile(path, p, axis=0) for p in PERCENTILES},
        mean=np.mean(path, axis=0),
    )


@dataclass(frozen=True)
class ConvergenceCheck:
    """Traces the mean IRR at growing prefixes of the same draw (25%, 50%,
    100% of scenarios) -- cheap since scenarios are i.i.d., no re-draw
    needed. `is_stable` is a simple heuristic: the last two points in the
    trace agree within 2 standard errors of the full-sample mean."""

    scenario_counts: list[int]
    mean_irr_trace: list[float]
    standard_error_irr: float
    standard_error_moic: float
    is_stable: bool


def compute_convergence_check(
    irr: np.ndarray, moic: np.ndarray, fractions: tuple[float, ...] = (0.25, 0.5, 1.0)
) -> ConvergenceCheck:
    n = irr.shape[0]
    counts = [max(1, int(n * f)) for f in fractions]
    trace = [float(np.mean(irr[:c])) for c in counts]
    se_irr = float(np.std(irr, ddof=1) / np.sqrt(n)) if n > 1 else float("nan")
    se_moic = float(np.std(moic, ddof=1) / np.sqrt(n)) if n > 1 else float("nan")
    is_stable = (
        abs(trace[-1] - trace[-2]) < 2 * se_irr
        if len(trace) >= 2 and not np.isnan(se_irr)
        else True
    )
    return ConvergenceCheck(
        scenario_counts=counts,
        mean_irr_trace=trace,
        standard_error_irr=se_irr,
        standard_error_moic=se_moic,
        is_stable=is_stable,
    )


@dataclass(frozen=True)
class DownsideAnalytics:
    returns: ReturnDistribution
    covenant_breaches: CovenantBreachAnalytics
    distress: DistressAnalytics
    leverage_bands: PathPercentileBands
    liquidity_bands: PathPercentileBands  # cash + undrawn revolver capacity
    cash_bands: PathPercentileBands  # cash alone, kept separate -- see simulate/liquidity.py
    convergence: ConvergenceCheck


def compute_downside_analytics(
    simulation_result: SimulationResult, irr_hurdle: float
) -> DownsideAnalytics:
    exit_result = simulation_result.exit_result
    return DownsideAnalytics(
        returns=compute_return_distribution(exit_result.irr, exit_result.moic, irr_hurdle),
        covenant_breaches=compute_covenant_breach_analytics(simulation_result.covenant_results),
        distress=compute_distress_analytics(simulation_result.distress_by_year),
        leverage_bands=compute_path_percentile_bands(
            simulation_result.credit_metrics.total_net_leverage
        ),
        liquidity_bands=compute_path_percentile_bands(simulation_result.liquidity_mm),
        cash_bands=compute_path_percentile_bands(simulation_result.model_result.balance_sheet.cash),
        convergence=compute_convergence_check(exit_result.irr, exit_result.moic),
    )
