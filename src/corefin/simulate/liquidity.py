"""Liquidity definition, shared by stress scenarios (simulate/stress.py) and
the Monte Carlo downside analytics (simulate/analytics.py): total liquidity
is cash plus undrawn revolver capacity, not cash alone -- a company with a
large undrawn revolver is more liquid than its cash balance alone suggests,
which matters for "minimum liquidity" and liquidity percentile bands.
"""

from __future__ import annotations

import numpy as np

from corefin.assumptions.schema import TrancheConfig
from corefin.debt.circularity import DebtScheduleResult


def compute_liquidity(
    cash: np.ndarray, debt_schedule: DebtScheduleResult, tranches: list[TrancheConfig]
) -> np.ndarray:
    """cash + (revolver commitment - drawn balance). Same shape as `cash`."""
    revolver = next(t for t in tranches if t.is_revolver)
    drawn = debt_schedule.ending_balance[revolver.name]
    undrawn_capacity = revolver.size_mm - drawn
    return cash + undrawn_capacity
