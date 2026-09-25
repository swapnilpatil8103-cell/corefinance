"""Distress definition, shared by stress scenarios (simulate/stress.py) and
the Monte Carlo downside analytics (simulate/analytics.py): a scenario-period
is in distress if the revolver is fully drawn with cash still below the
minimum, or if cash interest coverage falls below 1.0x.
"""

from __future__ import annotations

import numpy as np

from corefin.debt.circularity import DebtScheduleResult
from corefin.metrics.credit_metrics import CreditMetrics

INTEREST_COVERAGE_DISTRESS_THRESHOLD = 1.0


def compute_distress_flags(
    debt_schedule: DebtScheduleResult, credit_metrics: CreditMetrics
) -> np.ndarray:
    """(n_scenarios, n_periods) bool. `debt_schedule.shortfall_flag` is
    already exactly "cash remains below the minimum even after drawing the
    revolver to its full available capacity" (see debt/waterfall.py) --
    which only happens once the revolver is fully drawn, so it's reused
    directly rather than recomputed against tranche size_mm."""
    revolver_shortfall = debt_schedule.shortfall_flag
    thin_coverage = credit_metrics.interest_coverage < INTEREST_COVERAGE_DISTRESS_THRESHOLD
    return revolver_shortfall | thin_coverage
