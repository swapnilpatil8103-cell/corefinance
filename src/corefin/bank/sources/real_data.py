"""Assembles a `BankOpeningBalance` from real, already-parsed Call Report
(`capital_parse.parse_bank_capital_zip`) and FR Y-9C
(`fr_y9c.parse_y9c_bulk_zip`) rows, per the level-of-consolidation choice
documented in `corefin.bank.schema`: capital/RWA/earnings/equity/balance-
sheet lines come from the holding company's own Y-9C row; deposits and
loans come from the bank subsidiary's own Call Report row (loans via the
credit engine's `CreditLossProjection`, passed in as a single jump-off
total rather than re-derived here).

TWO LINES ARE DERIVED RESIDUALS, not directly reported figures --
documented here, not hidden:
  - `other_liabilities_mm` = HC total liabilities minus bank-level
    deposits. Borrowings (FHLB advances, subordinated debt, etc.) are not
    separately itemized by this function and are folded in here;
    `borrowings_mm` is always 0.0 from this builder.
  - `other_assets_mm` = HC total assets minus cash minus securities minus
    the supplied bank-level net loan total minus goodwill minus other
    intangibles. Besides the genuine "other assets" bucket (premises,
    OREO, BOLI, etc.), this residual also absorbs the Call-Report-vs-Y-9C
    loan-level gap `model.reconcile_levels` reports separately -- by
    design, so the balance sheet still balances exactly at jump-off even
    though the loan figure feeding it is sourced at a different
    consolidation level than the rest of the balance sheet.

Both residuals are expected to be small and positive for a well-behaved
bank; `BankOpeningBalance`'s own `ge=0.0` validation on both fields will
raise if either DTA goes negative in a way that would indicate corefin.bank.schema
item codes resolve to something unexpected for a given bank.

YTD-TO-QUARTERLY -- Q4-ONLY BY DEFAULT, NOT YTD/4: every Y-9C income-
statement item used here (interest income/expense, noninterest income/
expense, preferred dividends) is reported calendar-year-to-date as of
the report date -- see schema.py's "YTD-VS-QUARTERLY". The CORRECT
quarterly run-rate is Q4 YTD minus Q3 YTD (the standard `ytd_to_quarterly`
differencing `corefin.credit.panel` already uses for charge-offs/
recoveries), supplied here via the optional `prior_quarter_*_ytd_mm`
parameters. A flat YTD/4 average is only used as a FALLBACK when no
prior-quarter figure is supplied, and is a real source of error for any
bank whose balance sheet changed materially during the year -- confirmed
against real data: the acquirer bank's own total assets grew ~25%
quarter-on-quarter between 2025Q3 and 2025Q4 (apparently its own
acquisition), making YTD/4 understate Q4-only net interest income by
~20% ($139mm vs the real $175mm) and noninterest expense by ~17% ($101mm
vs $122mm). `QOQ_ASSET_CHANGE_FLAG_THRESHOLD` flags any bank whose HC
total assets moved more than 10% quarter-on-quarter, specifically
because that is exactly the condition under which YTD/4 becomes
unreliable -- the flag is informational (`DataQualityFlags`), not an
error; callers can still proceed, but should prefer supplying real prior-
quarter figures when it fires.

PREFERRED DIVIDENDS have the same YTD issue, PLUS a second one: a
newly-issued or partially-redeemed preferred tranche can make even a
correct Q4-only figure an unreliable estimate of the GOING-FORWARD
quarterly dividend. `preferred_dividend_annual_rate` (a contractual
coupon rate on `preferred_stock_mm`, e.g. 0.06 for a 6% annual rate) is
offered as an alternative, explicit override for exactly this reason --
documented per bank when used, since it is an assumption, not a reported
figure."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from corefin.bank import schema
from corefin.bank.schema import BankOpeningBalance
from corefin.credit.panel import TOTAL_ALLOWANCE_RCFD_FALLBACK_ITEM
from corefin.credit.schema import TOTAL_ALLOWANCE_ITEM

_THOUSANDS_TO_MM = 1.0 / 1_000.0
_YTD_Q4_TO_QUARTERLY = 1.0 / 4.0

QOQ_ASSET_CHANGE_FLAG_THRESHOLD = 0.10


@dataclass(frozen=True)
class DataQualityFlags:
    """Informational flags from building ONE bank's `BankOpeningBalance`
    -- callers decide what to do with them (log, surface to the user,
    etc.); nothing here raises on its own."""

    qoq_asset_change_pct: float | None  # None if prior_quarter_hc_total_assets_mm wasn't given
    qoq_asset_change_flagged: bool
    used_annualized_fallback_for: tuple[str, ...]  # income items that fell back to YTD/4
    # because no prior-quarter figure (or override) was supplied for them


def _quarterly_run_rate(
    current_ytd_mm: float,
    prior_ytd_mm: float | None,
    override_mm: float | None,
) -> tuple[float, bool]:
    """Returns (quarterly_mm, used_fallback). Precedence: explicit
    override > Q4-minus-Q3 differencing > flat YTD/4 fallback."""
    if override_mm is not None:
        return override_mm, False
    if prior_ytd_mm is not None:
        return current_ytd_mm - prior_ytd_mm, False
    return current_ytd_mm * _YTD_Q4_TO_QUARTERLY, True


def build_opening_balance_from_real_data(
    name: str,
    bank_id: str,
    hc_rssd_id: str,
    call_report_row: pd.Series,
    y9c_row: pd.Series,
    bank_level_net_loans_mm: float,
    prior_quarter_y9c_row: pd.Series | None = None,
    net_interest_income_quarterly_override_mm: float | None = None,
    noninterest_income_quarterly_override_mm: float | None = None,
    noninterest_expense_quarterly_override_mm: float | None = None,
    preferred_dividends_quarterly_override_mm: float | None = None,
    preferred_dividend_annual_rate: float | None = None,
) -> tuple[BankOpeningBalance, DataQualityFlags]:
    """`call_report_row`/`y9c_row`: one row (e.g. `df.loc[idx]`) from
    `capital_parse.parse_bank_capital_zip`'s / `fr_y9c.parse_y9c_bulk_zip`'s
    output, already filtered to this bank/holding company. Dollar item
    values are in thousands (both sources' own convention); converted to
    $mm here. Also sets `BankOpeningBalance.reported_allowance_mm` from
    `call_report_row`'s own RCON3123 (RCFD3123 fallback) -- the real
    reported allowance `model.run_bank_model` anchors the credit engine's
    own modeled allowance to; see `corefin.bank.allowance`'s module
    docstring. `bank_level_net_loans_mm`: the credit engine's own jump-off
    total loan balance net of allowance (already in $mm), e.g.
    `CreditLossProjection.balance_total_mm[0] - CreditLossProjection.
    allowance_total_mm[0]`.

    `prior_quarter_y9c_row`: the SAME holding company's Y-9C row for the
    prior quarter (e.g. Q3 if `y9c_row` is Q4) -- used for the Q4-only
    run-rate differencing and the QoQ asset-change flag; see module
    docstring. Omit if unavailable (falls back to YTD/4 for whichever
    income items aren't also given an explicit override, and
    `qoq_asset_change_pct`/`qoq_asset_change_flagged` stay unset).

    `*_quarterly_override_mm`/`preferred_dividend_annual_rate`: explicit
    overrides, highest precedence -- see `_quarterly_run_rate`.
    `preferred_dividend_annual_rate` (e.g. 0.06) is applied as
    `rate * preferred_stock_mm / 4`; ignored if
    `preferred_dividends_quarterly_override_mm` is also given."""
    cash_mm = (
        y9c_row[schema.HC_CASH_NONINTEREST_ITEM] + y9c_row[schema.HC_CASH_INTEREST_BEARING_ITEM]
    ) * _THOUSANDS_TO_MM
    securities_afs_mm = y9c_row[schema.HC_SECURITIES_AFS_ITEM] * _THOUSANDS_TO_MM
    securities_htm_mm = y9c_row[schema.HC_SECURITIES_HTM_ITEM] * _THOUSANDS_TO_MM
    goodwill_mm = y9c_row[schema.HC_GOODWILL_ITEM] * _THOUSANDS_TO_MM
    other_intangibles_mm = y9c_row[schema.HC_OTHER_INTANGIBLES_ITEM] * _THOUSANDS_TO_MM
    equity_mm = y9c_row[schema.HC_TOTAL_EQUITY_CAPITAL_ITEM] * _THOUSANDS_TO_MM
    hc_total_assets_mm = y9c_row[schema.HC_TOTAL_ASSETS_ITEM] * _THOUSANDS_TO_MM
    hc_total_liabilities_mm = y9c_row[schema.HC_TOTAL_LIABILITIES_ITEM] * _THOUSANDS_TO_MM
    preferred_stock_mm = y9c_row[schema.HC_PREFERRED_STOCK_ITEM] * _THOUSANDS_TO_MM

    # bank-level (not HC-level), the SAME consolidation level as the credit engine's own
    # loan/NCO data this anchors against -- see corefin.bank.allowance's module docstring.
    # RCFD (consolidated) fallback wherever RCON (domestic-only) is missing or exactly zero,
    # matching credit.panel.apply_rcfd_fallback's own documented convention.
    rcon_allowance = call_report_row.get(TOTAL_ALLOWANCE_ITEM)
    rcfd_allowance = call_report_row.get(TOTAL_ALLOWANCE_RCFD_FALLBACK_ITEM)
    rcon_missing = rcon_allowance is None or pd.isna(rcon_allowance)
    rcfd_available = rcfd_allowance is not None and not pd.isna(rcfd_allowance)
    if (rcon_missing or rcon_allowance == 0) and rcfd_available:
        reported_allowance_mm: float | None = rcfd_allowance * _THOUSANDS_TO_MM
    elif not rcon_missing:
        reported_allowance_mm = rcon_allowance * _THOUSANDS_TO_MM
    else:
        reported_allowance_mm = None  # item not present in this row -- anchoring skipped

    deposits_mm = call_report_row[schema.TOTAL_DEPOSITS_ITEM] * _THOUSANDS_TO_MM
    borrowings_mm = 0.0  # see module docstring -- folded into other_liabilities_mm
    other_liabilities_mm = hc_total_liabilities_mm - deposits_mm
    other_assets_mm = hc_total_assets_mm - (
        cash_mm
        + securities_afs_mm
        + securities_htm_mm
        + bank_level_net_loans_mm
        + goodwill_mm
        + other_intangibles_mm
    )

    nii_income_item, nii_expense_item = schema.HC_NET_INTEREST_INCOME_ITEMS
    current_nii_ytd_mm = (y9c_row[nii_income_item] - y9c_row[nii_expense_item]) * _THOUSANDS_TO_MM
    current_noninterest_income_ytd_mm = (
        y9c_row[schema.HC_NONINTEREST_INCOME_ITEM] * _THOUSANDS_TO_MM
    )
    current_noninterest_expense_ytd_mm = (
        y9c_row[schema.HC_NONINTEREST_EXPENSE_ITEM] * _THOUSANDS_TO_MM
    )
    current_preferred_dividends_ytd_mm = (
        y9c_row[schema.HC_PREFERRED_DIVIDENDS_ITEM] * _THOUSANDS_TO_MM
    )

    prior_nii_ytd_mm = prior_noninterest_income_ytd_mm = None
    prior_noninterest_expense_ytd_mm = prior_preferred_dividends_ytd_mm = None
    prior_hc_total_assets_mm = None
    if prior_quarter_y9c_row is not None:
        prior_nii_ytd_mm = (
            prior_quarter_y9c_row[nii_income_item] - prior_quarter_y9c_row[nii_expense_item]
        ) * _THOUSANDS_TO_MM
        prior_noninterest_income_ytd_mm = (
            prior_quarter_y9c_row[schema.HC_NONINTEREST_INCOME_ITEM] * _THOUSANDS_TO_MM
        )
        prior_noninterest_expense_ytd_mm = (
            prior_quarter_y9c_row[schema.HC_NONINTEREST_EXPENSE_ITEM] * _THOUSANDS_TO_MM
        )
        prior_preferred_dividends_ytd_mm = (
            prior_quarter_y9c_row[schema.HC_PREFERRED_DIVIDENDS_ITEM] * _THOUSANDS_TO_MM
        )
        prior_hc_total_assets_mm = (
            prior_quarter_y9c_row[schema.HC_TOTAL_ASSETS_ITEM] * _THOUSANDS_TO_MM
        )

    net_interest_income_jumpoff_mm, nii_fallback = _quarterly_run_rate(
        current_nii_ytd_mm, prior_nii_ytd_mm, net_interest_income_quarterly_override_mm
    )
    noninterest_income_jumpoff_mm, noninterest_income_fallback = _quarterly_run_rate(
        current_noninterest_income_ytd_mm,
        prior_noninterest_income_ytd_mm,
        noninterest_income_quarterly_override_mm,
    )
    noninterest_expense_jumpoff_mm, noninterest_expense_fallback = _quarterly_run_rate(
        current_noninterest_expense_ytd_mm,
        prior_noninterest_expense_ytd_mm,
        noninterest_expense_quarterly_override_mm,
    )

    if preferred_dividends_quarterly_override_mm is not None:
        preferred_dividends_jumpoff_mm = preferred_dividends_quarterly_override_mm
        preferred_dividends_fallback = False
    elif preferred_dividend_annual_rate is not None:
        preferred_dividends_jumpoff_mm = preferred_dividend_annual_rate * preferred_stock_mm / 4.0
        preferred_dividends_fallback = False
    else:
        preferred_dividends_jumpoff_mm, preferred_dividends_fallback = _quarterly_run_rate(
            current_preferred_dividends_ytd_mm, prior_preferred_dividends_ytd_mm, None
        )

    fallbacks = []
    if nii_fallback:
        fallbacks.append("net_interest_income")
    if noninterest_income_fallback:
        fallbacks.append("noninterest_income")
    if noninterest_expense_fallback:
        fallbacks.append("noninterest_expense")
    if preferred_dividends_fallback:
        fallbacks.append("preferred_dividends")

    qoq_asset_change_pct = (
        float((hc_total_assets_mm - prior_hc_total_assets_mm) / prior_hc_total_assets_mm)
        if prior_hc_total_assets_mm
        else None
    )
    flags = DataQualityFlags(
        qoq_asset_change_pct=qoq_asset_change_pct,
        qoq_asset_change_flagged=bool(
            qoq_asset_change_pct is not None
            and abs(qoq_asset_change_pct) > QOQ_ASSET_CHANGE_FLAG_THRESHOLD
        ),
        used_annualized_fallback_for=tuple(fallbacks),
    )

    opening = BankOpeningBalance(
        name=name,
        bank_id=bank_id,
        hc_rssd_id=hc_rssd_id,
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
        net_interest_income_jumpoff_mm=net_interest_income_jumpoff_mm,
        noninterest_income_jumpoff_mm=noninterest_income_jumpoff_mm,
        noninterest_expense_jumpoff_mm=noninterest_expense_jumpoff_mm,
        preferred_dividends_jumpoff_mm=preferred_dividends_jumpoff_mm,
        reported_allowance_mm=reported_allowance_mm,
        preferred_stock_mm=preferred_stock_mm,
        goodwill_net_of_dtl_mm=y9c_row[schema.HC_GOODWILL_NET_OF_DTL_ITEM] * _THOUSANDS_TO_MM,
        other_intangibles_net_of_dtl_mm=(
            y9c_row[schema.HC_OTHER_INTANGIBLES_NET_OF_DTL_ITEM] * _THOUSANDS_TO_MM
        ),
        dta_nol_deduction_mm=y9c_row[schema.HC_DTA_NOL_DEDUCTION_ITEM] * _THOUSANDS_TO_MM,
        aoci_afs_unrealized_mm=y9c_row[schema.HC_AOCI_AFS_ITEM] * _THOUSANDS_TO_MM,
        aoci_cash_flow_hedge_mm=y9c_row[schema.HC_AOCI_CASH_FLOW_HEDGE_ITEM] * _THOUSANDS_TO_MM,
        aoci_pension_mm=y9c_row[schema.HC_AOCI_PENSION_ITEM] * _THOUSANDS_TO_MM,
        aoci_htm_mm=y9c_row[schema.HC_AOCI_HTM_ITEM] * _THOUSANDS_TO_MM,
        other_cet1_deductions_mm=y9c_row[schema.HC_OTHER_CET1_DEDUCTIONS_ITEM] * _THOUSANDS_TO_MM,
        reported_cet1_capital_mm=y9c_row[schema.HC_CET1_CAPITAL_ITEM] * _THOUSANDS_TO_MM,
        reported_cet1_ratio=y9c_row[schema.HC_CET1_RATIO_ITEM] / 100.0,
        reported_rwa_mm=y9c_row[schema.HC_RWA_ITEM] * _THOUSANDS_TO_MM,
        reported_tier1_leverage_ratio=y9c_row[schema.HC_TIER1_LEVERAGE_RATIO_ITEM] / 100.0,
        reported_tier1_capital_ratio=(
            y9c_row[schema.HC_TIER1_RISK_BASED_CAPITAL_RATIO_ITEM] / 100.0
        ),
        reported_total_capital_ratio=(
            y9c_row[schema.HC_TOTAL_RISK_BASED_CAPITAL_RATIO_ITEM] / 100.0
        ),
    )
    return opening, flags
