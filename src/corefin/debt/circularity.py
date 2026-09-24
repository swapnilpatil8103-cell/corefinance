"""Vectorized fixed-point solver for interest-on-average-balance circularity,
and the full multi-period debt schedule built on top of it.

Interest for period t depends on period t's ending balance (average-balance
mode), which depends on cash available for debt service, which depends on
net income, which depends on interest for period t -- circular, but only
*within* a period, since the prior period's ending balance is already fixed
by the time period t starts. So the fixed point is solved once per period,
vectorized across every scenario simultaneously, before moving to the next
period (which is why the period loop itself cannot be vectorized away: it is
genuinely sequential).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from corefin.assumptions.schema import InterestMode, TrancheConfig, WaterfallConfig
from corefin.debt.schedule import (
    effective_rate,
    fee_amortization_schedule,
    scheduled_mandatory_amortization,
)
from corefin.debt.waterfall import WaterfallOutcome, run_period_waterfall
from corefin.scenarios.drivers import DriverSet
from corefin.statements.income_statement import nol_and_tax_one_period
from corefin.timeline import Timeline


@dataclass(frozen=True)
class DebtScheduleResult:
    beginning_balance: dict[str, np.ndarray]
    ending_balance: dict[str, np.ndarray]
    cash_interest: dict[str, np.ndarray]
    pik_interest: dict[str, np.ndarray]
    fee_amortization: dict[str, np.ndarray]
    mandatory_amort: dict[str, np.ndarray]
    sweep_amort: dict[str, np.ndarray]
    revolver_draw: np.ndarray
    revolver_paydown: np.ndarray
    ending_cash: np.ndarray
    shortfall_flag: np.ndarray
    iterations_used: np.ndarray


class CircularityNotConvergedError(RuntimeError):
    pass


def _recompute_interest(
    tranches: list[TrancheConfig],
    beginning_balances: dict[str, np.ndarray],
    ending_balances: dict[str, np.ndarray],
    rate_t: dict[str, np.ndarray],
    interest_mode: InterestMode,
) -> dict[str, np.ndarray]:
    new_interest: dict[str, np.ndarray] = {}
    for t in tranches:
        beginning = beginning_balances[t.name]
        ending = ending_balances[t.name]
        base_balance = (
            (beginning + ending) / 2.0
            if interest_mode is InterestMode.AVERAGE_BALANCE
            else beginning
        )
        coupon = base_balance * rate_t[t.name]
        if t.is_revolver and t.commitment_fee_pct:
            undrawn_beginning = t.size_mm - beginning
            undrawn_ending = t.size_mm - ending
            undrawn = (
                (undrawn_beginning + undrawn_ending) / 2.0
                if interest_mode is InterestMode.AVERAGE_BALANCE
                else undrawn_beginning
            )
            coupon = coupon + t.commitment_fee_pct * undrawn
        new_interest[t.name] = coupon
    return new_interest


def _run_iteration(
    tranches: list[TrancheConfig],
    waterfall_config: WaterfallConfig,
    beginning_balances: dict[str, np.ndarray],
    scheduled_amort_t: dict[str, np.ndarray],
    rate_t: dict[str, np.ndarray],
    interest_guess: dict[str, np.ndarray],
    fee_amort_total_t: np.ndarray,
    ebit_t: np.ndarray,
    da_t: np.ndarray,
    capex_t: np.ndarray,
    delta_nwc_t: np.ndarray,
    tax_rate: float,
    nol_beginning: np.ndarray,
    dividend_pct_of_ni: float,
    cash_beginning: np.ndarray,
) -> tuple[WaterfallOutcome, np.ndarray, dict[str, np.ndarray], dict[str, np.ndarray]]:
    total_interest_t = sum(interest_guess.values()) + fee_amort_total_t
    ebt_t = ebit_t - total_interest_t
    tax_t, nol_ending = nol_and_tax_one_period(ebt_t, tax_rate, nol_beginning)
    net_income_t = ebt_t - tax_t
    dividends_t = net_income_t * dividend_pct_of_ni

    pik_accrual_t = {t.name: interest_guess[t.name] * t.pik_fraction for t in tranches}
    pik_total_t = sum(pik_accrual_t.values())
    cfo_t = net_income_t + da_t + pik_total_t + fee_amort_total_t - delta_nwc_t
    cash_available_pre_financing = cash_beginning + cfo_t - capex_t - dividends_t

    outcome = run_period_waterfall(
        tranches=tranches,
        beginning_balances=beginning_balances,
        scheduled_mandatory_amort_t=scheduled_amort_t,
        cash_available_pre_financing=cash_available_pre_financing,
        minimum_cash_mm=waterfall_config.minimum_cash_mm,
        sweep_pct=waterfall_config.sweep_pct,
        pik_accrual_t=pik_accrual_t,
    )

    new_interest_guess = _recompute_interest(
        tranches,
        beginning_balances,
        outcome.ending_balances,
        rate_t,
        waterfall_config.interest_mode,
    )
    cash_interest_t = {t.name: interest_guess[t.name] * (1 - t.pik_fraction) for t in tranches}
    return outcome, nol_ending, new_interest_guess, cash_interest_t


def solve_period(
    tranches: list[TrancheConfig],
    waterfall_config: WaterfallConfig,
    beginning_balances: dict[str, np.ndarray],
    scheduled_amort_t: dict[str, np.ndarray],
    fee_amort_t: dict[str, np.ndarray],
    rate_t: dict[str, np.ndarray],
    ebit_t: np.ndarray,
    da_t: np.ndarray,
    capex_t: np.ndarray,
    delta_nwc_t: np.ndarray,
    tax_rate: float,
    nol_beginning: np.ndarray,
    dividend_pct_of_ni: float,
    cash_beginning: np.ndarray,
) -> tuple[WaterfallOutcome, dict[str, np.ndarray], dict[str, np.ndarray], np.ndarray, int]:
    fee_amort_total_t = sum(fee_amort_t.values())
    interest_guess = {t.name: beginning_balances[t.name] * rate_t[t.name] for t in tranches}
    revolver = next(t for t in tranches if t.is_revolver)
    if revolver.commitment_fee_pct:
        interest_guess[revolver.name] = interest_guess[
            revolver.name
        ] + revolver.commitment_fee_pct * (revolver.size_mm - beginning_balances[revolver.name])

    outcome = None
    nol_ending = nol_beginning
    converged_at = None
    for iteration in range(1, waterfall_config.circularity_max_iterations + 1):
        outcome, nol_ending, new_interest_guess, _ = _run_iteration(
            tranches,
            waterfall_config,
            beginning_balances,
            scheduled_amort_t,
            rate_t,
            interest_guess,
            fee_amort_total_t,
            ebit_t,
            da_t,
            capex_t,
            delta_nwc_t,
            tax_rate,
            nol_beginning,
            dividend_pct_of_ni,
            cash_beginning,
        )
        max_diff = max(
            float(np.max(np.abs(new_interest_guess[name] - interest_guess[name])))
            for name in interest_guess
        )
        interest_guess = new_interest_guess
        if max_diff < waterfall_config.circularity_tolerance:
            converged_at = iteration
            break

    if converged_at is None:
        raise CircularityNotConvergedError(
            f"interest circularity did not converge within "
            f"{waterfall_config.circularity_max_iterations} iterations"
        )

    outcome, nol_ending, _, cash_interest_t = _run_iteration(
        tranches,
        waterfall_config,
        beginning_balances,
        scheduled_amort_t,
        rate_t,
        interest_guess,
        fee_amort_total_t,
        ebit_t,
        da_t,
        capex_t,
        delta_nwc_t,
        tax_rate,
        nol_beginning,
        dividend_pct_of_ni,
        cash_beginning,
    )
    pik_interest_t = {t.name: interest_guess[t.name] * t.pik_fraction for t in tranches}
    return outcome, cash_interest_t, pik_interest_t, nol_ending, converged_at


def run_debt_schedule(
    tranches: list[TrancheConfig],
    waterfall_config: WaterfallConfig,
    timeline: Timeline,
    drivers: DriverSet,
    ebit: np.ndarray,
    da: np.ndarray,
    capex: np.ndarray,
    delta_nwc: np.ndarray,
    tax_rate: float,
    nol_beginning_mm: float,
    dividend_pct_of_ni: float,
    cash_beginning_mm: float,
) -> DebtScheduleResult:
    n_scenarios, n_periods = ebit.shape
    rate_arrays = {t.name: effective_rate(t, drivers.base_rate) for t in tranches}
    scheduled_amort = {
        t.name: scheduled_mandatory_amortization(t, n_periods)
        for t in tranches
        if not t.is_revolver
    }
    fee_schedules = {t.name: fee_amortization_schedule(t, n_periods) for t in tranches}
    non_revolver_names = [t.name for t in tranches if not t.is_revolver]

    beginning_balances = {
        t.name: np.full(n_scenarios, 0.0 if t.is_revolver else t.size_mm) for t in tranches
    }
    cash_running = np.full(n_scenarios, cash_beginning_mm, dtype=float)
    nol_running = np.full(n_scenarios, nol_beginning_mm, dtype=float)

    beginning_balance_out = {t.name: np.zeros((n_scenarios, n_periods)) for t in tranches}
    ending_balance_out = {t.name: np.zeros((n_scenarios, n_periods)) for t in tranches}
    cash_interest_out = {t.name: np.zeros((n_scenarios, n_periods)) for t in tranches}
    pik_interest_out = {t.name: np.zeros((n_scenarios, n_periods)) for t in tranches}
    fee_amort_out = {t.name: np.zeros((n_scenarios, n_periods)) for t in tranches}
    mandatory_amort_out = {name: np.zeros((n_scenarios, n_periods)) for name in non_revolver_names}
    sweep_amort_out = {name: np.zeros((n_scenarios, n_periods)) for name in non_revolver_names}
    revolver_draw_out = np.zeros((n_scenarios, n_periods))
    revolver_paydown_out = np.zeros((n_scenarios, n_periods))
    ending_cash_out = np.zeros((n_scenarios, n_periods))
    shortfall_flag_out = np.zeros((n_scenarios, n_periods), dtype=bool)
    iterations_used = np.zeros(n_periods, dtype=int)

    for t_idx in range(n_periods):
        scheduled_amort_t = {
            name: np.full(n_scenarios, scheduled_amort[name][t_idx]) for name in scheduled_amort
        }
        fee_amort_t = {
            name: np.full(n_scenarios, fee_schedules[name][t_idx]) for name in fee_schedules
        }
        rate_t = {name: rate_arrays[name][:, t_idx] for name in rate_arrays}

        outcome, cash_interest_t, pik_interest_t, nol_running, n_iter = solve_period(
            tranches=tranches,
            waterfall_config=waterfall_config,
            beginning_balances=beginning_balances,
            scheduled_amort_t=scheduled_amort_t,
            fee_amort_t=fee_amort_t,
            rate_t=rate_t,
            ebit_t=ebit[:, t_idx],
            da_t=da[:, t_idx],
            capex_t=capex[:, t_idx],
            delta_nwc_t=delta_nwc[:, t_idx],
            tax_rate=tax_rate,
            nol_beginning=nol_running,
            dividend_pct_of_ni=dividend_pct_of_ni,
            cash_beginning=cash_running,
        )

        for t in tranches:
            beginning_balance_out[t.name][:, t_idx] = beginning_balances[t.name]
            ending_balance_out[t.name][:, t_idx] = outcome.ending_balances[t.name]
            cash_interest_out[t.name][:, t_idx] = cash_interest_t[t.name]
            pik_interest_out[t.name][:, t_idx] = pik_interest_t[t.name]
            fee_amort_out[t.name][:, t_idx] = fee_amort_t[t.name]
        for name in non_revolver_names:
            mandatory_amort_out[name][:, t_idx] = outcome.mandatory_amort[name]
            sweep_amort_out[name][:, t_idx] = outcome.sweep_amort[name]
        revolver_draw_out[:, t_idx] = outcome.revolver_draw
        revolver_paydown_out[:, t_idx] = outcome.revolver_paydown
        ending_cash_out[:, t_idx] = outcome.ending_cash
        shortfall_flag_out[:, t_idx] = outcome.shortfall_flag
        iterations_used[t_idx] = n_iter

        beginning_balances = outcome.ending_balances
        cash_running = outcome.ending_cash

    return DebtScheduleResult(
        beginning_balance=beginning_balance_out,
        ending_balance=ending_balance_out,
        cash_interest=cash_interest_out,
        pik_interest=pik_interest_out,
        fee_amortization=fee_amort_out,
        mandatory_amort=mandatory_amort_out,
        sweep_amort=sweep_amort_out,
        revolver_draw=revolver_draw_out,
        revolver_paydown=revolver_paydown_out,
        ending_cash=ending_cash_out,
        shortfall_flag=shortfall_flag_out,
        iterations_used=iterations_used,
    )
