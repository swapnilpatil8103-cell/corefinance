"""ONE bank's income statement, aligned to the same jump-off-is-period-0
quarterly `Timeline` as `balance_sheet.py`. Pure numpy, (n_periods,)
arrays. Net interest income and noninterest income/expense are held flat
at their jump-off $ level (grown by `BankConfig.balance_sheet_growth_rate`,
same as the balance sheet lines they scale with) -- a static-margin
simplification appropriate for Stage 2's standalone single-bank model;
`corefin.ma` layers purchase-accounting mark accretion/CDI amortization
on top for the pro forma combined entity. When `config.ppnr_stress` is
set AND `rate_path_pp`/`earning_assets_jumpoff_mm` are supplied, NII and
noninterest income instead respond to the projection's own macro
scenario -- see `corefin.bank.ppnr`'s own module docstring for the full
mechanism and why it exists.

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

from corefin.bank import ppnr
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
    rate_path_pp: np.ndarray | None = None,
    earning_assets_jumpoff_mm: float | None = None,
) -> IncomeStatement:
    """provision_expense_total_mm: (n_periods,) -- the credit engine's
    `CreditLossProjection.provision_expense_mm.sum(axis=0)`, already
    aligned to `timeline`. `rate_path_pp`/`earning_assets_jumpoff_mm`:
    only used when `config.ppnr_stress` is set (see `corefin.bank.
    ppnr.align_rate_path_to_timeline` for building the former); raises
    if `config.ppnr_stress` is set but either is missing."""
    growth_factor = (1.0 + config.balance_sheet_growth_rate) ** timeline.period_index

    if config.ppnr_stress is not None:
        if rate_path_pp is None or earning_assets_jumpoff_mm is None:
            raise ValueError(
                "config.ppnr_stress is set but rate_path_pp/earning_assets_jumpoff_mm "
                "wasn't supplied"
            )
        net_interest_income_mm = ppnr.compute_stressed_nii_mm(
            net_interest_income_jumpoff_mm,
            earning_assets_jumpoff_mm,
            rate_path_pp,
            config.ppnr_stress.nim_beta,
        )
        noninterest_income_mm = ppnr.compute_stressed_noninterest_income_mm(
            noninterest_income_jumpoff_mm,
            timeline.n_periods,
            config.ppnr_stress.noninterest_income_decline_pct,
        )
        # sticky in a downturn -- held flat, not even grown (see module docstring)
        noninterest_expense_mm = np.full(timeline.n_periods, noninterest_expense_jumpoff_mm)
    else:
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
