"""Income statement: pure numpy functions, no config/pydantic dependency.

All arrays are (n_scenarios, n_periods) unless noted. Units per
`assumptions.schema`: $mm for dollar amounts, decimal fractions for rates.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from corefin.scenarios.drivers import DriverSet


@dataclass(frozen=True)
class IncomeStatement:
    revenue: np.ndarray
    ebitda: np.ndarray
    da: np.ndarray
    ebit: np.ndarray
    cash_interest_expense: np.ndarray
    pik_interest_expense: np.ndarray
    ebt: np.ndarray
    tax: np.ndarray
    net_income: np.ndarray
    nol_balance: np.ndarray


def compute_revenue(revenue_base_mm: float, revenue_growth: np.ndarray) -> np.ndarray:
    """revenue_growth[:, t] is growth from period t-1 (period -1 = revenue_base_mm) to t."""
    return revenue_base_mm * np.cumprod(1.0 + revenue_growth, axis=1)


def nol_and_tax_one_period(
    ebt_t: np.ndarray, tax_rate: float, nol_beginning: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Single-period NOL/tax step, shared by the no-debt IS path (looped over periods
    below) and the debt module's per-period circularity solve, so both use identical
    tax logic. All args (n_scenarios,). Returns (tax_t, nol_ending)."""
    usable = np.minimum(nol_beginning, np.maximum(ebt_t, 0.0))
    taxable_t = np.maximum(ebt_t, 0.0) - usable
    tax_t = taxable_t * tax_rate
    new_losses = np.maximum(-ebt_t, 0.0)
    nol_ending = nol_beginning - usable + new_losses
    return tax_t, nol_ending


def compute_nol_and_tax(
    ebt: np.ndarray, tax_rate: float, nol_beginning_mm: float
) -> tuple[np.ndarray, np.ndarray]:
    """Taxes never go negative; a pretax loss instead grows the NOL carryforward,
    which offsets future taxable income dollar-for-dollar until exhausted.
    Returns (tax, nol_balance_end_of_period), both (n_scenarios, n_periods)."""
    n_scenarios, n_periods = ebt.shape
    tax = np.zeros_like(ebt)
    nol_balance = np.zeros_like(ebt)
    current_nol = np.full(n_scenarios, nol_beginning_mm, dtype=float)
    for t in range(n_periods):
        tax[:, t], current_nol = nol_and_tax_one_period(ebt[:, t], tax_rate, current_nol)
        nol_balance[:, t] = current_nol
    return tax, nol_balance


def compute_income_statement(
    drivers: DriverSet,
    revenue_base_mm: float,
    da_pct_revenue: np.ndarray,
    tax_rate: float,
    nol_beginning_mm: float,
    cash_interest_expense: np.ndarray | None = None,
    pik_interest_expense: np.ndarray | None = None,
) -> IncomeStatement:
    revenue = compute_revenue(revenue_base_mm, drivers.revenue_growth)
    ebitda = revenue * drivers.ebitda_margin
    da = revenue * da_pct_revenue
    ebit = ebitda - da
    if cash_interest_expense is None:
        cash_interest_expense = np.zeros_like(ebit)
    if pik_interest_expense is None:
        pik_interest_expense = np.zeros_like(ebit)
    ebt = ebit - cash_interest_expense - pik_interest_expense
    tax, nol_balance = compute_nol_and_tax(ebt, tax_rate, nol_beginning_mm)
    net_income = ebt - tax
    return IncomeStatement(
        revenue=revenue,
        ebitda=ebitda,
        da=da,
        ebit=ebit,
        cash_interest_expense=cash_interest_expense,
        pik_interest_expense=pik_interest_expense,
        ebt=ebt,
        tax=tax,
        net_income=net_income,
        nol_balance=nol_balance,
    )
