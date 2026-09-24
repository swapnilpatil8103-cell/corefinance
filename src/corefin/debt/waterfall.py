"""One period's cash waterfall, given the interest/PIK accrual for that period.

Order (per the brief): minimum cash balance -> mandatory amortization ->
revolver draw if there's a shortfall -> revolver paydown -> optional cash
sweep by priority with a configurable sweep %. Pure function of the period's
inputs; the outer fixed-point loop over interest lives in `circularity.py`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from corefin.assumptions.schema import TrancheConfig


@dataclass(frozen=True)
class WaterfallOutcome:
    ending_balances: dict[str, np.ndarray]
    mandatory_amort: dict[str, np.ndarray]
    sweep_amort: dict[str, np.ndarray]
    revolver_draw: np.ndarray
    revolver_paydown: np.ndarray
    ending_cash: np.ndarray
    shortfall_flag: np.ndarray


def run_period_waterfall(
    tranches: list[TrancheConfig],
    beginning_balances: dict[str, np.ndarray],
    scheduled_mandatory_amort_t: dict[str, np.ndarray],
    cash_available_pre_financing: np.ndarray,
    minimum_cash_mm: float,
    sweep_pct: float,
    pik_accrual_t: dict[str, np.ndarray],
) -> WaterfallOutcome:
    revolver = next(t for t in tranches if t.is_revolver)
    non_revolver = [t for t in tranches if not t.is_revolver]

    mandatory_amort: dict[str, np.ndarray] = {}
    ending_after_mandatory: dict[str, np.ndarray] = {}
    total_mandatory = np.zeros_like(cash_available_pre_financing)
    for t in non_revolver:
        beginning = beginning_balances[t.name]
        amort = np.minimum(scheduled_mandatory_amort_t[t.name], beginning)
        mandatory_amort[t.name] = amort
        ending_after_mandatory[t.name] = beginning - amort
        total_mandatory = total_mandatory + amort

    cash_after_mandatory = cash_available_pre_financing - total_mandatory

    revolver_beginning = beginning_balances[revolver.name]
    revolver_availability = np.maximum(revolver.size_mm - revolver_beginning, 0.0)
    shortfall_needed = np.maximum(minimum_cash_mm - cash_after_mandatory, 0.0)
    revolver_draw = np.minimum(shortfall_needed, revolver_availability)
    shortfall_flag = (cash_after_mandatory + revolver_draw) < (minimum_cash_mm - 1e-9)

    excess = np.maximum(cash_after_mandatory - minimum_cash_mm, 0.0)
    revolver_paydown = np.minimum(excess, revolver_beginning)
    revolver_ending = revolver_beginning + revolver_draw - revolver_paydown

    cash_after_revolver = cash_after_mandatory + revolver_draw - revolver_paydown

    sweep_pool = np.maximum(cash_after_revolver - minimum_cash_mm, 0.0) * sweep_pct
    remaining_pool = sweep_pool
    sweep_amort: dict[str, np.ndarray] = {}
    ending_balances_out: dict[str, np.ndarray] = dict(ending_after_mandatory)
    eligible = sorted(
        (t for t in non_revolver if t.cash_sweep_eligible), key=lambda t: t.sweep_priority
    )
    for t in non_revolver:
        sweep_amort[t.name] = np.zeros_like(cash_available_pre_financing)
    for t in eligible:
        pay = np.minimum(remaining_pool, ending_balances_out[t.name])
        sweep_amort[t.name] = pay
        ending_balances_out[t.name] = ending_balances_out[t.name] - pay
        remaining_pool = remaining_pool - pay

    ending_cash = cash_after_revolver - (sweep_pool - remaining_pool)

    for t in non_revolver:
        ending_balances_out[t.name] = ending_balances_out[t.name] + pik_accrual_t.get(
            t.name, np.zeros_like(cash_available_pre_financing)
        )
    ending_balances_out[revolver.name] = revolver_ending

    return WaterfallOutcome(
        ending_balances=ending_balances_out,
        mandatory_amort=mandatory_amort,
        sweep_amort=sweep_amort,
        revolver_draw=revolver_draw,
        revolver_paydown=revolver_paydown,
        ending_cash=ending_cash,
        shortfall_flag=shortfall_flag,
    )
