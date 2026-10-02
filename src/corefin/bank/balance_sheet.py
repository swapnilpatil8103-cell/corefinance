"""ONE bank's balance sheet over a quarterly `Timeline` whose period 0 is
the jump-off (last-actual, non-projection) quarter -- the same convention
`corefin.credit.interface.CreditLossProjection` uses. Pure numpy, every
array shape (n_periods,) (NOT (n_scenarios, n_periods) -- this module
projects one bank under one scenario at a time; Stage 5's stress test
calls it once per scenario).

Static by default: every line except loans/allowance (owned by the
credit engine's own `CreditLossProjection`) and equity (which must roll
forward with retained earnings) stays flat at its jump-off value unless
`BankConfig.balance_sheet_growth_rate` is nonzero."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from corefin.bank.schema import BankConfig, BankOpeningBalance
from corefin.credit.interface import CreditLossProjection
from corefin.timeline import Timeline


@dataclass(frozen=True)
class BalanceSheet:
    timeline: Timeline
    categories: tuple[str, ...]

    loan_balance_mm: np.ndarray  # (n_categories, n_periods) -- from CreditLossProjection
    allowance_mm: np.ndarray  # (n_categories, n_periods) -- from CreditLossProjection

    cash_mm: np.ndarray
    securities_afs_mm: np.ndarray
    securities_htm_mm: np.ndarray
    other_assets_mm: np.ndarray
    goodwill_mm: np.ndarray
    other_intangibles_mm: np.ndarray

    deposits_mm: np.ndarray
    borrowings_mm: np.ndarray
    other_liabilities_mm: np.ndarray
    equity_mm: np.ndarray

    def __post_init__(self) -> None:
        n = self.timeline.n_periods
        n_categories = len(self.categories)
        for name in (
            "cash_mm",
            "securities_afs_mm",
            "securities_htm_mm",
            "other_assets_mm",
            "goodwill_mm",
            "other_intangibles_mm",
            "deposits_mm",
            "borrowings_mm",
            "other_liabilities_mm",
            "equity_mm",
        ):
            array = getattr(self, name)
            if array.shape != (n,):
                raise ValueError(f"{name} has shape {array.shape}, expected ({n},)")
        for name in ("loan_balance_mm", "allowance_mm"):
            array = getattr(self, name)
            if array.shape != (n_categories, n):
                raise ValueError(f"{name} has shape {array.shape}, expected ({n_categories}, {n})")

    @property
    def total_loans_mm(self) -> np.ndarray:
        return self.loan_balance_mm.sum(axis=0)

    @property
    def total_allowance_mm(self) -> np.ndarray:
        return self.allowance_mm.sum(axis=0)

    @property
    def net_loans_mm(self) -> np.ndarray:
        return self.total_loans_mm - self.total_allowance_mm

    @property
    def total_securities_mm(self) -> np.ndarray:
        return self.securities_afs_mm + self.securities_htm_mm

    @property
    def total_assets_mm(self) -> np.ndarray:
        return (
            self.cash_mm
            + self.total_securities_mm
            + self.net_loans_mm
            + self.goodwill_mm
            + self.other_intangibles_mm
            + self.other_assets_mm
        )

    @property
    def total_liabilities_mm(self) -> np.ndarray:
        return self.deposits_mm + self.borrowings_mm + self.other_liabilities_mm

    @property
    def total_liabilities_and_equity_mm(self) -> np.ndarray:
        return self.total_liabilities_mm + self.equity_mm


def roll_forward_from_jumpoff(
    opening_value_mm: float, period_deltas_mm: np.ndarray, is_projection: np.ndarray
) -> np.ndarray:
    """period_deltas_mm[t] is period t's OWN flow (e.g. net income minus
    dividends that quarter). Returns level[0] == opening_value_mm exactly
    (period 0 is the jump-off quarter -- its own flow already produced the
    reported opening level, so it is excluded from the cumulative sum, not
    double-counted) and level[t] == opening_value_mm +
    sum(period_deltas_mm[1:t+1]) for projected periods t >= 1."""
    deltas_for_cumsum = np.where(is_projection, period_deltas_mm, 0.0)
    return opening_value_mm + np.cumsum(deltas_for_cumsum)


def project_balance_sheet(
    opening: BankOpeningBalance,
    credit_projection: CreditLossProjection,
    net_income_mm: np.ndarray,
    dividends_mm: np.ndarray,
    config: BankConfig,
    timeline: Timeline,
) -> BalanceSheet:
    """`credit_projection` must already be aligned to `timeline` (same
    quarter-0-is-jump-off convention; `corefin.credit.interface` builds it
    that way). `net_income_mm`/`dividends_mm`: (n_periods,), from
    `income_statement.compute_income_statement`.

    CASH IS THE BALANCING PLUG, not independently grown: every other line
    is set first (securities/other assets/deposits/borrowings/other
    liabilities grown at `balance_sheet_growth_rate`, goodwill/intangibles
    static, loans/allowance from the credit engine, equity rolled forward
    with retained earnings), then cash is whatever value makes total
    assets equal total liabilities and equity. This is the standard
    simplification in a statement model without an explicit cash-flow
    statement -- without it, retained earnings would accumulate in equity
    with no asset-side counterpart (nothing to fund loan/securities growth
    or just sit as excess cash), and the balance sheet would not balance
    from period 1 onward."""
    growth_factor = (1.0 + config.balance_sheet_growth_rate) ** timeline.period_index

    equity_mm = roll_forward_from_jumpoff(
        opening.equity_mm, net_income_mm - dividends_mm, timeline.is_projection
    )
    securities_afs_mm = opening.securities_afs_mm * growth_factor
    securities_htm_mm = opening.securities_htm_mm * growth_factor
    other_assets_mm = opening.other_assets_mm * growth_factor
    goodwill_mm = np.full(timeline.n_periods, opening.goodwill_mm)
    other_intangibles_mm = np.full(timeline.n_periods, opening.other_intangibles_mm)
    deposits_mm = opening.deposits_mm * growth_factor
    borrowings_mm = opening.borrowings_mm * growth_factor
    other_liabilities_mm = opening.other_liabilities_mm * growth_factor

    net_loans_mm = credit_projection.balance_mm.sum(axis=0) - credit_projection.allowance_mm.sum(
        axis=0
    )
    total_liabilities_and_equity_mm = deposits_mm + borrowings_mm + other_liabilities_mm + equity_mm
    cash_mm = total_liabilities_and_equity_mm - (
        net_loans_mm
        + securities_afs_mm
        + securities_htm_mm
        + goodwill_mm
        + other_intangibles_mm
        + other_assets_mm
    )

    return BalanceSheet(
        timeline=timeline,
        categories=credit_projection.categories,
        loan_balance_mm=credit_projection.balance_mm,
        allowance_mm=credit_projection.allowance_mm,
        cash_mm=cash_mm,
        securities_afs_mm=securities_afs_mm,
        securities_htm_mm=securities_htm_mm,
        other_assets_mm=other_assets_mm,
        goodwill_mm=goodwill_mm,
        other_intangibles_mm=other_intangibles_mm,
        deposits_mm=deposits_mm,
        borrowings_mm=borrowings_mm,
        other_liabilities_mm=other_liabilities_mm,
        equity_mm=equity_mm,
    )
