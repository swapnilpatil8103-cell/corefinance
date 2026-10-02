"""ONE bank's income statement, aligned to the same jump-off-is-period-0
quarterly `Timeline` as `balance_sheet.py`. Pure numpy, (n_periods,)
arrays. Net interest income and noninterest income/expense are held flat
at their jump-off $ level (grown by `BankConfig.balance_sheet_growth_rate`,
same as the balance sheet lines they scale with) -- a static-margin
simplification appropriate for Stage 2's standalone single-bank model;
`corefin.ma` layers purchase-accounting mark accretion/CDI amortization
on top for the pro forma combined entity.

Tax simplification: no NOL carryforward (out of scope for this stage --
see schema.py's `BankConfig.tax_rate` docstring) -- a pretax loss pays no
tax (floored at zero) rather than generating a carryforward benefit.

PREFERRED STOCK: preferred dividends have priority over common -- EPS
(here and in `corefin.ma`) must use `net_income_available_to_common_mm`
(net income minus preferred dividends), never `net_income_mm` alone.
`preferred_dividends_mm` is held STATIC (unlike the other income lines,
never grown by `balance_sheet_growth_rate`), consistent with
`BankOpeningBalance.preferred_dividends_jumpoff_mm`'s own docstring."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from corefin.bank.schema import BankConfig
from corefin.timeline import Timeline


@dataclass(frozen=True)
class IncomeStatement:
    timeline: Timeline
    net_interest_income_mm: np.ndarray
    noninterest_income_mm: np.ndarray
    noninterest_expense_mm: np.ndarray
    provision_expense_mm: np.ndarray
    pretax_income_mm: np.ndarray
    tax_expense_mm: np.ndarray
    net_income_mm: np.ndarray
    preferred_dividends_mm: np.ndarray
    net_income_available_to_common_mm: np.ndarray
    dividends_mm: np.ndarray  # COMMON dividends only -- see preferred_dividends_mm above

    def __post_init__(self) -> None:
        n = self.timeline.n_periods
        for name in (
            "net_interest_income_mm",
            "noninterest_income_mm",
            "noninterest_expense_mm",
            "provision_expense_mm",
            "pretax_income_mm",
            "tax_expense_mm",
            "net_income_mm",
            "preferred_dividends_mm",
            "net_income_available_to_common_mm",
            "dividends_mm",
        ):
            array = getattr(self, name)
            if array.shape != (n,):
                raise ValueError(f"{name} has shape {array.shape}, expected ({n},)")

    @property
    def total_dividends_mm(self) -> np.ndarray:
        """Common + preferred -- the equity roll-forward's (balance_sheet.py)
        correct dividend deduction, since `BalanceSheet.equity_mm` includes
        preferred stock."""
        return self.dividends_mm + self.preferred_dividends_mm


def compute_income_statement(
    net_interest_income_jumpoff_mm: float,
    noninterest_income_jumpoff_mm: float,
    noninterest_expense_jumpoff_mm: float,
    provision_expense_total_mm: np.ndarray,
    config: BankConfig,
    timeline: Timeline,
    preferred_dividends_jumpoff_mm: float = 0.0,
) -> IncomeStatement:
    """provision_expense_total_mm: (n_periods,) -- the credit engine's
    `CreditLossProjection.provision_expense_mm.sum(axis=0)`, already
    aligned to `timeline`."""
    growth_factor = (1.0 + config.balance_sheet_growth_rate) ** timeline.period_index

    net_interest_income_mm = net_interest_income_jumpoff_mm * growth_factor
    noninterest_income_mm = noninterest_income_jumpoff_mm * growth_factor
    noninterest_expense_mm = noninterest_expense_jumpoff_mm * growth_factor

    pretax_income_mm = (
        net_interest_income_mm
        + noninterest_income_mm
        - noninterest_expense_mm
        - provision_expense_total_mm
    )
    tax_expense_mm = np.maximum(pretax_income_mm, 0.0) * config.tax_rate
    net_income_mm = pretax_income_mm - tax_expense_mm
    preferred_dividends_mm = np.full(timeline.n_periods, preferred_dividends_jumpoff_mm)
    net_income_available_to_common_mm = net_income_mm - preferred_dividends_mm
    dividends_mm = np.maximum(net_income_available_to_common_mm, 0.0) * config.dividend_payout_ratio

    return IncomeStatement(
        timeline=timeline,
        net_interest_income_mm=net_interest_income_mm,
        noninterest_income_mm=noninterest_income_mm,
        noninterest_expense_mm=noninterest_expense_mm,
        provision_expense_mm=provision_expense_total_mm,
        pretax_income_mm=pretax_income_mm,
        tax_expense_mm=tax_expense_mm,
        net_income_mm=net_income_mm,
        preferred_dividends_mm=preferred_dividends_mm,
        net_income_available_to_common_mm=net_income_available_to_common_mm,
        dividends_mm=dividends_mm,
    )
