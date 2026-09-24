"""Balance sheet roll-forwards. Pure numpy, (n_scenarios, n_periods) arrays."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class BalanceSheet:
    cash: np.ndarray
    nwc: np.ndarray
    ppe: np.ndarray
    goodwill: np.ndarray
    deferred_financing_costs: np.ndarray
    total_debt: np.ndarray
    other_liabilities: np.ndarray
    equity: np.ndarray

    @property
    def total_assets(self) -> np.ndarray:
        return self.cash + self.nwc + self.ppe + self.goodwill + self.deferred_financing_costs

    @property
    def total_liabilities_and_equity(self) -> np.ndarray:
        return self.total_debt + self.other_liabilities + self.equity


def compute_nwc(revenue: np.ndarray, nwc_pct_revenue: np.ndarray) -> np.ndarray:
    return revenue * nwc_pct_revenue


def roll_forward_level(beginning_mm: float, period_deltas: np.ndarray) -> np.ndarray:
    """level[t] = beginning + sum(period_deltas[:, :t+1]) -- a cumulative roll-forward
    shared by cash, PP&E and equity."""
    return beginning_mm + np.cumsum(period_deltas, axis=1)


def roll_forward_ppe(ppe_beginning_mm: float, capex: np.ndarray, da: np.ndarray) -> np.ndarray:
    return roll_forward_level(ppe_beginning_mm, capex - da)


def roll_forward_cash(cash_beginning_mm: float, change_in_cash: np.ndarray) -> np.ndarray:
    return roll_forward_level(cash_beginning_mm, change_in_cash)


def roll_forward_equity(
    equity_beginning_mm: float, net_income: np.ndarray, dividends: np.ndarray
) -> np.ndarray:
    return roll_forward_level(equity_beginning_mm, net_income - dividends)


def roll_forward_deferred_financing_costs(
    beginning_mm: float, fee_amortization_total: np.ndarray
) -> np.ndarray:
    """The upfront fee/OID asset amortizes down (non-cash) by the same amount
    that hits interest expense on the IS each period."""
    return roll_forward_level(beginning_mm, -fee_amortization_total)


def balance_residual(balance_sheet: BalanceSheet) -> np.ndarray:
    """Assets minus liabilities+equity; should be ~0 in every scenario/period."""
    return balance_sheet.total_assets - balance_sheet.total_liabilities_and_equity
