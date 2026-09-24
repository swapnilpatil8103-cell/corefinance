"""Covenant evaluation: headroom and breach flags per scenario/period.

Leverage covenants are maximums (breach when value > threshold); coverage
covenants (interest coverage, FCCR) are minimums (breach when value < threshold).
Headroom is always signed so that positive = compliant, regardless of direction.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from corefin.assumptions.loader import expand_series
from corefin.assumptions.schema import CovenantConfig, CovenantMetric
from corefin.metrics.credit_metrics import CreditMetrics
from corefin.timeline import Timeline

MAXIMUM_METRICS = frozenset({CovenantMetric.TOTAL_NET_LEVERAGE, CovenantMetric.SENIOR_NET_LEVERAGE})
MINIMUM_METRICS = frozenset({CovenantMetric.INTEREST_COVERAGE, CovenantMetric.FCCR})


@dataclass(frozen=True)
class CovenantResult:
    name: str
    metric_value: np.ndarray
    threshold: np.ndarray
    headroom: np.ndarray
    breach: np.ndarray


def _metric_value(metric: CovenantMetric, metrics: CreditMetrics) -> np.ndarray:
    return {
        CovenantMetric.TOTAL_NET_LEVERAGE: metrics.total_net_leverage,
        CovenantMetric.SENIOR_NET_LEVERAGE: metrics.senior_net_leverage,
        CovenantMetric.INTEREST_COVERAGE: metrics.interest_coverage,
        CovenantMetric.FCCR: metrics.fccr,
    }[metric]


def evaluate_covenant(
    covenant: CovenantConfig, metrics: CreditMetrics, timeline: Timeline
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

    tested = timeline.period_index >= covenant.test_from_period
    breach = (headroom < 0) & tested[np.newaxis, :]

    return CovenantResult(
        name=covenant.name,
        metric_value=value,
        threshold=np.broadcast_to(threshold, value.shape),
        headroom=headroom,
        breach=breach,
    )


def evaluate_covenants(
    covenants: list[CovenantConfig], metrics: CreditMetrics, timeline: Timeline
) -> list[CovenantResult]:
    return [evaluate_covenant(c, metrics, timeline) for c in covenants]
