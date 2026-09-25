"""Integrity checks for the debt module."""

from __future__ import annotations

import numpy as np

from corefin.assumptions.schema import TrancheConfig
from corefin.checks.framework import CheckResult, check_bounds, check_close_to_zero
from corefin.debt.circularity import DebtScheduleResult

DEFAULT_TOLERANCE = 1e-6


def check_debt_rollforward(
    debt_schedule: DebtScheduleResult, tolerance: float = DEFAULT_TOLERANCE
) -> CheckResult:
    """delta(total debt) must equal draws - repayments + PIK accrual, every period."""
    total_beginning = sum(debt_schedule.beginning_balance.values())
    total_ending = sum(debt_schedule.ending_balance.values())
    total_repayments = (
        sum(debt_schedule.mandatory_amort.values())
        + sum(debt_schedule.sweep_amort.values())
        + debt_schedule.revolver_paydown
    )
    total_pik = sum(debt_schedule.pik_interest.values())
    expected_delta = debt_schedule.revolver_draw - total_repayments + total_pik
    residual = (total_ending - total_beginning) - expected_delta
    return check_close_to_zero("debt_rollforward", residual, tolerance)


def check_revolver_bounds(
    debt_schedule: DebtScheduleResult, revolver: TrancheConfig, tolerance: float = DEFAULT_TOLERANCE
) -> CheckResult:
    balance = debt_schedule.ending_balance[revolver.name]
    return check_bounds(
        "revolver_bounds", balance, lower=0.0, upper=revolver.size_mm, tolerance=tolerance
    )


def check_sweep_priority_respected(
    debt_schedule: DebtScheduleResult,
    tranches: list[TrancheConfig],
    tolerance: float = DEFAULT_TOLERANCE,
) -> CheckResult:
    """A lower-priority tranche can only receive sweep cash once every higher-priority,
    sweep-eligible tranche has been paid down to zero (before its own PIK accrual)."""
    eligible = sorted(
        (t for t in tranches if not t.is_revolver and t.cash_sweep_eligible),
        key=lambda t: t.sweep_priority,
    )
    if len(eligible) < 2:
        first = debt_schedule.sweep_amort[eligible[0].name] if eligible else np.array([[0.0]])
        return check_close_to_zero("sweep_priority_respected", np.zeros_like(first), tolerance)

    violation = None
    for i, higher in enumerate(eligible):
        remaining_after_own_sweep = (
            debt_schedule.ending_balance[higher.name] - debt_schedule.pik_interest[higher.name]
        )
        for lower in eligible[i + 1 :]:
            lower_got_swept = debt_schedule.sweep_amort[lower.name] > tolerance
            higher_not_repaid = remaining_after_own_sweep > tolerance
            bad = np.where(lower_got_swept & higher_not_repaid, remaining_after_own_sweep, 0.0)
            violation = bad if violation is None else np.maximum(violation, bad)
    return check_close_to_zero("sweep_priority_respected", violation, tolerance)
