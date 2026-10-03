"""The pro forma combined income statement AND balance sheet, projected
forward across the deal's quarterly `Timeline`.

PERIOD CONVENTION: period 0 is the CLOSING snapshot (exactly
`DealResult.pro_forma_balance_sheet`/`pro_forma_cet1_bridge` -- no income
flow yet, the deal has just closed) -- NOT each bank's own pre-deal
jump-off quarter (which is what period 0 means in each bank's own
standalone `BankModelResult`). Periods 1..n-1 are full quarters of
combined POST-CLOSE activity, built from each bank's own standalone
Stage 2 projection (`BankModelResult.income_statement`, periods 1..n-1
only -- period 0 is skipped since that's pre-close) plus the deal-
specific adjustments below.

DEAL-SPECIFIC ADJUSTMENTS, each a quarterly schedule over periods 1..n-1
(period 0 is always 0.0 -- no flow at the close instant):
  - MARK ACCRETION: the loan rate mark and securities mark (negative --
    write-downs at close) unwind back into net interest income
    straight-line over their configured lives (`RateMarkConfig.
    rate_mark_life_years`/`SecuritiesMarkConfig.securities_mark_life_years`)
    -- `purchase_accounting.compute_straight_line_schedule`.
  - CDI AMORTIZATION: the new core deposit intangible, sum-of-years-
    digits (front-loaded) over `CdiConfig.cdi_amortization_years` --
    `purchase_accounting.compute_sum_of_years_digits_schedule`. A
    noninterest EXPENSE (reduces pretax income).
  - COST SAVES: `CostSaveConfig.cost_save_pct_of_target_noninterest_expense`
    applied to the TARGET's own standalone projected noninterest expense
    (`target_result.income_statement.noninterest_expense_mm`), phased in
    at `phase_in_year1_pct` for quarters 1-4 (year 1) and 100% from
    quarter 5 onward.
  - FOREGONE INTEREST: `ConsiderationConfig.cash_funding_cost_rate`
    applied to the cash consideration EVERY quarter -- the opportunity
    cost of deploying that cash into the deal instead of earning assets
    (Stage 3 assumes cash consideration is funded from the acquirer's
    own balance sheet, not new debt, so this is an opportunity cost, not
    interest expense on new borrowings).
  - ONE-TIME CHARGES (Day-2 non-PCD allowance + restructuring charge,
    both pretax): hit ONLY quarter 1 (the first full quarter after
    close), not spread across the horizon -- a real GAAP income-
    statement event in the quarter immediately following close."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from corefin.bank.model import BankModelResult
from corefin.bank.schema import BankConfig
from corefin.ma.purchase_accounting import (
    FairValueMarks,
    SourcesAndUses,
    compute_straight_line_schedule,
    compute_sum_of_years_digits_schedule,
)
from corefin.ma.schema import DealConfig
from corefin.timeline import Timeline

_PERIOD_LENGTH_YEARS = 0.25  # quarterly, matches corefin.bank's own convention


@dataclass(frozen=True)
class ProFormaProjection:
    timeline: Timeline

    mark_accretion_mm: np.ndarray
    cdi_amortization_mm: np.ndarray
    cost_saves_mm: np.ndarray
    foregone_interest_mm: np.ndarray
    one_time_charges_pretax_mm: np.ndarray

    pretax_income_mm: np.ndarray
    tax_expense_mm: np.ndarray
    net_income_mm: np.ndarray
    preferred_dividends_mm: np.ndarray
    net_income_available_to_common_mm: np.ndarray
    dividends_mm: np.ndarray

    equity_mm: np.ndarray
    goodwill_mm: np.ndarray
    other_intangibles_mm: np.ndarray
    tangible_common_equity_mm: np.ndarray

    def __post_init__(self) -> None:
        n = self.timeline.n_periods
        for name in (
            "mark_accretion_mm",
            "cdi_amortization_mm",
            "cost_saves_mm",
            "foregone_interest_mm",
            "one_time_charges_pretax_mm",
            "pretax_income_mm",
            "tax_expense_mm",
            "net_income_mm",
            "preferred_dividends_mm",
            "net_income_available_to_common_mm",
            "dividends_mm",
            "equity_mm",
            "goodwill_mm",
            "other_intangibles_mm",
            "tangible_common_equity_mm",
        ):
            array = getattr(self, name)
            if array.shape != (n,):
                raise ValueError(f"{name} has shape {array.shape}, expected ({n},)")


def _post_close_schedule(per_quarter_flows: list[float]) -> np.ndarray:
    """Prepends a 0.0 for period 0 (the close instant -- no flow yet) to
    a list of n-1 post-close quarterly flows."""
    return np.concatenate([[0.0], np.asarray(per_quarter_flows)])


def compute_pro_forma_projection(
    acquirer_result: BankModelResult,
    target_result: BankModelResult,
    marks: FairValueMarks,
    sources_and_uses: SourcesAndUses,
    pro_forma_equity_at_close_mm: float,
    pro_forma_goodwill_at_close_mm: float,
    pro_forma_other_intangibles_at_close_mm: float,
    acquirer_bank_config: BankConfig,
    config: DealConfig,
) -> ProFormaProjection:
    """`pro_forma_equity_at_close_mm`/`pro_forma_goodwill_at_close_mm`/
    `pro_forma_other_intangibles_at_close_mm`:
    `pro_forma.ProFormaBalanceSheet`'s own fields -- the period-0 anchor
    this projection rolls forward from. `acquirer_bank_config`: the
    `BankConfig` the acquirer was run with (for its
    `dividend_payout_ratio`, assumed to continue post-close)."""
    timeline = acquirer_result.income_statement.timeline
    n = timeline.n_periods
    n_post_close = n - 1

    rate_mark_accretion = compute_straight_line_schedule(
        -marks.rate_mark_mm,
        config.rate_mark.rate_mark_life_years,
        n_post_close,
        _PERIOD_LENGTH_YEARS,
    )
    securities_mark_accretion = compute_straight_line_schedule(
        -marks.securities_mark_mm,
        config.securities_mark.securities_mark_life_years,
        n_post_close,
        _PERIOD_LENGTH_YEARS,
    )
    mark_accretion_mm = _post_close_schedule(
        [a + b for a, b in zip(rate_mark_accretion, securities_mark_accretion, strict=True)]
    )

    cdi_amortization_schedule = compute_sum_of_years_digits_schedule(
        marks.cdi_gross_mm, config.cdi.cdi_amortization_years, n_post_close, _PERIOD_LENGTH_YEARS
    )
    cdi_amortization_mm = _post_close_schedule(cdi_amortization_schedule)

    quarter_in_year1 = np.arange(1, n) <= 4
    phase_in = np.where(quarter_in_year1, config.cost_saves.phase_in_year1_pct, 1.0)
    cost_saves_mm = _post_close_schedule(
        list(
            phase_in
            * config.cost_saves.cost_save_pct_of_target_noninterest_expense
            * target_result.income_statement.noninterest_expense_mm[1:]
        )
    )

    quarterly_foregone_rate = config.consideration.cash_funding_cost_rate * _PERIOD_LENGTH_YEARS
    foregone_interest_mm = _post_close_schedule(
        [sources_and_uses.cash_consideration_mm * quarterly_foregone_rate] * n_post_close
    )

    one_time_charges_pretax_mm = np.zeros(n)
    one_time_charges_pretax_mm[1] = (
        marks.day2_allowance_non_pcd_mm + config.cost_saves.restructuring_charge_mm
    )

    combined_standalone_pretax_mm = _post_close_schedule(
        list(
            acquirer_result.income_statement.pretax_income_mm[1:]
            + target_result.income_statement.pretax_income_mm[1:]
        )
    )
    pretax_income_mm = (
        combined_standalone_pretax_mm
        + mark_accretion_mm
        - cdi_amortization_mm
        + cost_saves_mm
        - foregone_interest_mm
        - one_time_charges_pretax_mm
    )
    tax_expense_mm = np.maximum(pretax_income_mm, 0.0) * config.tax_rate
    net_income_mm = pretax_income_mm - tax_expense_mm
    preferred_dividends_mm = _post_close_schedule(
        list(acquirer_result.income_statement.preferred_dividends_mm[1:])
    )
    net_income_available_to_common_mm = net_income_mm - preferred_dividends_mm
    dividends_mm = (
        np.maximum(net_income_available_to_common_mm, 0.0)
        * acquirer_bank_config.dividend_payout_ratio
    )
    dividends_mm[0] = 0.0  # no flow at the close instant

    total_dividends_mm = dividends_mm + preferred_dividends_mm
    retained_mm = np.where(timeline.is_projection, net_income_mm - total_dividends_mm, 0.0)
    equity_mm = pro_forma_equity_at_close_mm + np.cumsum(retained_mm)

    goodwill_mm = np.full(n, pro_forma_goodwill_at_close_mm)
    other_intangibles_mm = pro_forma_other_intangibles_at_close_mm - np.cumsum(cdi_amortization_mm)
    tangible_common_equity_mm = (
        equity_mm - acquirer_result.opening.preferred_stock_mm - goodwill_mm - other_intangibles_mm
    )

    return ProFormaProjection(
        timeline=timeline,
        mark_accretion_mm=mark_accretion_mm,
        cdi_amortization_mm=cdi_amortization_mm,
        cost_saves_mm=cost_saves_mm,
        foregone_interest_mm=foregone_interest_mm,
        one_time_charges_pretax_mm=one_time_charges_pretax_mm,
        pretax_income_mm=pretax_income_mm,
        tax_expense_mm=tax_expense_mm,
        net_income_mm=net_income_mm,
        preferred_dividends_mm=preferred_dividends_mm,
        net_income_available_to_common_mm=net_income_available_to_common_mm,
        dividends_mm=dividends_mm,
        equity_mm=equity_mm,
        goodwill_mm=goodwill_mm,
        other_intangibles_mm=other_intangibles_mm,
        tangible_common_equity_mm=tangible_common_equity_mm,
    )
