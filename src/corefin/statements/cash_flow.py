"""Cash flow statement. Pure numpy, (n_scenarios, n_periods) arrays.

Sign convention: debt_draws is a positive inflow, debt_repayments is a
negative outflow (already signed), dividends is a positive outflow (sign-
flipped internally).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CashFlowStatement:
    net_income: np.ndarray
    da: np.ndarray
    pik_interest_expense: np.ndarray
    fee_amortization: np.ndarray
    delta_nwc: np.ndarray
    cfo: np.ndarray
    capex: np.ndarray
    cfi: np.ndarray
    debt_draws: np.ndarray
    debt_repayments: np.ndarray
    dividends: np.ndarray
    cff: np.ndarray
    change_in_cash: np.ndarray


def compute_delta(level: np.ndarray, beginning_mm: float) -> np.ndarray:
    """First difference of a level series, given its period -1 beginning balance."""
    prior = np.concatenate([np.full((level.shape[0], 1), beginning_mm), level[:, :-1]], axis=1)
    return level - prior


def compute_cash_flow_statement(
    net_income: np.ndarray,
    da: np.ndarray,
    nwc: np.ndarray,
    nwc_beginning_mm: float,
    capex: np.ndarray,
    pik_interest_expense: np.ndarray | None = None,
    fee_amortization: np.ndarray | None = None,
    debt_draws: np.ndarray | None = None,
    debt_repayments: np.ndarray | None = None,
    dividends: np.ndarray | None = None,
) -> CashFlowStatement:
    """pik_interest_expense and fee_amortization are non-cash charges already
    subtracted in net_income (like D&A) and so are added back here."""
    delta_nwc = compute_delta(nwc, nwc_beginning_mm)
    if pik_interest_expense is None:
        pik_interest_expense = np.zeros_like(net_income)
    if fee_amortization is None:
        fee_amortization = np.zeros_like(net_income)
    cfo = net_income + da + pik_interest_expense + fee_amortization - delta_nwc
    cfi = -capex
    if debt_draws is None:
        debt_draws = np.zeros_like(net_income)
    if debt_repayments is None:
        debt_repayments = np.zeros_like(net_income)
    if dividends is None:
        dividends = np.zeros_like(net_income)
    cff = debt_draws + debt_repayments - dividends
    change_in_cash = cfo + cfi + cff
    return CashFlowStatement(
        net_income=net_income,
        da=da,
        pik_interest_expense=pik_interest_expense,
        fee_amortization=fee_amortization,
        delta_nwc=delta_nwc,
        cfo=cfo,
        capex=capex,
        cfi=cfi,
        debt_draws=debt_draws,
        debt_repayments=debt_repayments,
        dividends=dividends,
        cff=cff,
        change_in_cash=change_in_cash,
    )
