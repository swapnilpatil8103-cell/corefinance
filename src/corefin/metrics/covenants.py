"""Covenant evaluation: headroom and breach flags per scenario/period.

Leverage covenants are maximums (breach when value > threshold); coverage
covenants (interest coverage, FCCR) are minimums (breach when value < threshold).
Headroom is always signed so that positive = compliant, regardless of direction.

A covenant with `springing_revolver_draw_pct` set is only *tested* in periods
where the revolver's drawn balance / commitment strictly exceeds that
fraction -- untested periods are neither a pass nor a breach.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from corefin.assumptions.loader import expand_series
from corefin.assumptions.schema import CovenantConfig, CovenantMetric, TrancheConfig
from corefin.debt.circularity import DebtScheduleResult
from corefin.metrics.credit_metrics import CreditMetrics
from corefin.timeline import Timeline

MAXIMUM_METRICS = frozenset(
    {CovenantMetric.TOTAL_NET_LEVERAGE, CovenantMetric.SECURED_NET_LEVERAGE}
)
MINIMUM_METRICS = frozenset({CovenantMetric.INTEREST_COVERAGE, CovenantMetric.FCCR})

DEFAULT_MIN_HEADROOM_PCT = 0.15


@dataclass(frozen=True)
class CovenantResult:
    name: str
    metric_value: np.ndarray
    threshold: np.ndarray
    headroom: np.ndarray
    tested: np.ndarray
    breach: np.ndarray


def _metric_value(metric: CovenantMetric, metrics: CreditMetrics) -> np.ndarray:
    return {
        CovenantMetric.TOTAL_NET_LEVERAGE: metrics.total_net_leverage,
        CovenantMetric.SECURED_NET_LEVERAGE: metrics.secured_net_leverage,
        CovenantMetric.INTEREST_COVERAGE: metrics.interest_coverage,
        CovenantMetric.FCCR: metrics.fccr,
    }[metric]


def revolver_drawn_pct(
    debt_schedule: DebtScheduleResult, tranches: list[TrancheConfig]
) -> np.ndarray:
    """Drawn balance / commitment for the deal's revolver, shape (n_scenarios, n_periods).
    0 (never springs) if the revolver has no commitment."""
    revolver = next(t for t in tranches if t.is_revolver)
    balance = debt_schedule.ending_balance[revolver.name]
    if revolver.size_mm <= 0:
        return np.zeros_like(balance)
    return balance / revolver.size_mm


def _springing_tested_mask(
    covenant: CovenantConfig,
    debt_schedule: DebtScheduleResult | None,
    tranches: list[TrancheConfig] | None,
    shape: tuple[int, int],
) -> np.ndarray:
    if covenant.springing_revolver_draw_pct is None:
        return np.ones(shape, dtype=bool)
    if debt_schedule is None or tranches is None:
        raise ValueError(
            f"covenant '{covenant.name}' has springing_revolver_draw_pct set but "
            "no debt_schedule/tranches were provided to evaluate it"
        )
    return revolver_drawn_pct(debt_schedule, tranches) > covenant.springing_revolver_draw_pct


def evaluate_covenant(
    covenant: CovenantConfig,
    metrics: CreditMetrics,
    timeline: Timeline,
    debt_schedule: DebtScheduleResult | None = None,
    tranches: list[TrancheConfig] | None = None,
) -> CovenantResult:
    threshold = expand_series(covenant.threshold, timeline.n_periods, covenant.name).reshape(
        1, timeline.n_periods
    )
    value = _metric_value(covenant.metric, metrics)

    if covenant.metric in MAXIMUM_METRICS:
        headroom = threshold - value
    elif covenant.metric in MINIMUM_METRICS:
        headroom = value - threshold
    else:
        raise ValueError(f"unhandled covenant metric: {covenant.metric}")

    period_tested = timeline.period_index >= covenant.test_from_period
    springing_tested = _springing_tested_mask(covenant, debt_schedule, tranches, value.shape)
    tested = period_tested[np.newaxis, :] & springing_tested
    breach = tested & (headroom < 0)

    return CovenantResult(
        name=covenant.name,
        metric_value=value,
        threshold=np.broadcast_to(threshold, value.shape),
        headroom=headroom,
        tested=tested,
        breach=breach,
    )


def evaluate_covenants(
    covenants: list[CovenantConfig],
    metrics: CreditMetrics,
    timeline: Timeline,
    debt_schedule: DebtScheduleResult | None = None,
    tranches: list[TrancheConfig] | None = None,
) -> list[CovenantResult]:
    return [evaluate_covenant(c, metrics, timeline, debt_schedule, tranches) for c in covenants]


@dataclass(frozen=True)
class CovenantHeadroom:
    name: str
    tested: bool
    period_label: str | None
    metric_value: float | None
    threshold: float | None
    cushion_pct: float | None


def compute_base_case_headroom(
    covenant_results: list[CovenantResult], timeline: Timeline, scenario: int = 0
) -> list[CovenantHeadroom]:
    """Headroom at each covenant's first tested period, for a single (typically
    deterministic, zero-vol) scenario. A covenant that's never tested in this
    scenario (e.g. a springing covenant whose trigger never fires) reports
    tested=False rather than a fabricated headroom number."""
    summaries = []
    for cov in covenant_results:
        tested_periods = np.nonzero(cov.tested[scenario])[0]
        if tested_periods.size == 0:
            summaries.append(CovenantHeadroom(cov.name, False, None, None, None, None))
            continue
        t0 = int(tested_periods[0])
        threshold_t0 = float(cov.threshold[scenario, t0])
        value_t0 = float(cov.metric_value[scenario, t0])
        headroom_t0 = float(cov.headroom[scenario, t0])
        cushion_pct = headroom_t0 / threshold_t0 if threshold_t0 != 0 else float("inf")
        summaries.append(
            CovenantHeadroom(
                name=cov.name,
                tested=True,
                period_label=timeline.year_labels[t0],
                metric_value=value_t0,
                threshold=threshold_t0,
                cushion_pct=cushion_pct,
            )
        )
    return summaries


def low_headroom_warnings(
    headrooms: list[CovenantHeadroom], min_cushion_pct: float = DEFAULT_MIN_HEADROOM_PCT
) -> list[CovenantHeadroom]:
    """The subset of (tested) headrooms below the minimum cushion -- these are
    what a caller should surface as a warning, not an error."""
    return [h for h in headrooms if h.tested and h.cushion_pct < min_cushion_pct]
