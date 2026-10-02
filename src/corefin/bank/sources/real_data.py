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

YTD-TO-QUARTERLY: every Y-9C income-statement item used here (interest
income/expense, noninterest income/expense, preferred dividends) is
reported calendar-year-to-date as of the report date -- see schema.py's
"YTD-VS-QUARTERLY". Each is divided by 4 here to approximate a single
quarter's run-rate for a Q4 jump-off."""

from __future__ import annotations

import pandas as pd

from corefin.bank import schema
from corefin.bank.schema import BankOpeningBalance

_THOUSANDS_TO_MM = 1.0 / 1_000.0
_YTD_Q4_TO_QUARTERLY = 1.0 / 4.0


def build_opening_balance_from_real_data(
    name: str,
    bank_id: str,
    hc_rssd_id: str,
    call_report_row: pd.Series,
    y9c_row: pd.Series,
    bank_level_net_loans_mm: float,
) -> BankOpeningBalance:
    """`call_report_row`/`y9c_row`: one row (e.g. `df.loc[idx]`) from
    `capital_parse.parse_bank_capital_zip`'s / `fr_y9c.parse_y9c_bulk_zip`'s
    output, already filtered to this bank/holding company. Dollar item
    values are in thousands (both sources' own convention); converted to
    $mm here. `bank_level_net_loans_mm`: the credit engine's own jump-off
    total loan balance net of allowance (already in $mm), e.g.
    `CreditLossProjection.balance_total_mm[0] - CreditLossProjection.
    allowance_total_mm[0]`."""
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
    net_interest_income_jumpoff_mm = (
        (y9c_row[nii_income_item] - y9c_row[nii_expense_item])
        * _THOUSANDS_TO_MM
        * _YTD_Q4_TO_QUARTERLY
    )
    noninterest_income_jumpoff_mm = (
        y9c_row[schema.HC_NONINTEREST_INCOME_ITEM] * _THOUSANDS_TO_MM * _YTD_Q4_TO_QUARTERLY
    )
    noninterest_expense_jumpoff_mm = (
        y9c_row[schema.HC_NONINTEREST_EXPENSE_ITEM] * _THOUSANDS_TO_MM * _YTD_Q4_TO_QUARTERLY
    )
    preferred_dividends_jumpoff_mm = (
        y9c_row[schema.HC_PREFERRED_DIVIDENDS_ITEM] * _THOUSANDS_TO_MM * _YTD_Q4_TO_QUARTERLY
    )

    return BankOpeningBalance(
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
        preferred_stock_mm=y9c_row[schema.HC_PREFERRED_STOCK_ITEM] * _THOUSANDS_TO_MM,
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
    )
