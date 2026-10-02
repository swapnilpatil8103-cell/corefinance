"""MDRM item-code mappings and the `BankConfig` assumption schema for the
single-bank statement model.

CONSOLIDATION LEVEL (per explicit user direction): capital, RWA, earnings
and equity use the FR Y-9C (bank HOLDING COMPANY, consolidated) report --
EPS and share counts are holding-company figures, so capital ratios must
be measured at the same level investors see them at. Loan categories (and
therefore the Credit-Loss Forecasting Engine's NCOs/provisions/allowance)
stay at the Call Report (insured bank subsidiary) level, per
`corefin.credit`'s own existing panel. `model.py` reconciles the two
levels (loans, deposits, equity) and reports the gap rather than silently
picking one; see `model.reconcile_levels`.

Every MDRM code below was looked up against the Federal Reserve's own live
MDRM Data Dictionary (https://www.federalreserve.gov/apps/mdrm/data-dictionary,
searched by Item Number and Item Name/Title -- distinct from the static
7.6MB MDRM zip `corefin.credit.schema`'s codes were checked against) and,
where noted, cross-checked against real downloaded 2025Q4 bulk data (Call
Report ZIP already cached under data/raw/ffiec/, FR Y-9C bulk file
BHCF20251231.ZIP). Two codes initially assumed by analogy to Call Report
items turned out to be WRONG on direct MDRM lookup, not just unconfirmed:

- RCOANC99 (bank) / BHCANC99 (HC) is NOT common equity tier 1 capital --
  its real MDRM title is "Standardized Approach for Counterparty Credit
  Risk opt-in election," an unrelated indicator. Likely coincidence: it
  happened to return a non-null numeric value for the real 2025Q4 test
  banks, which read as superficially plausible without checking the
  dictionary.
- RCOAP742 / BHCAP742 is NOT the CET1 ratio -- its real MDRM title is
  "Common equity tier 1 capital: common stock plus related surplus, net
  of treasury stock and unearned ESOP shares," a DOLLAR-AMOUNT
  sub-component of the CET1 build-up (RC-R Part I item 2), not the ratio
  itself (RC-R Part I item 46).

The real codes, confirmed via MDRM Item Number search (both items span
RCOA/RCFA at the bank level and BHCA at the HC level, FFIEC 101/FFIEC
031/multiple forms/FR Y-9C respectively, start date 2014-03-31 -- the
post-Basel-III-revised-rules RC-R Part I schedule):

  P859: "COMMON EQUITY TIER 1 CAPITAL" (the numerator dollar amount --
        Schedule HC-R item 12 less item 18, per its own MDRM description)
  P793: "COMMON EQUITY TIER 1 CAPITAL RATIO"

A different kind of correction -- a real but RETIRED code, not a
fabricated one -- surfaced from a user-suggested code for other
intangible assets, RCFD0426/RCON0426: confirmed via MDRM as genuinely
"OTHER IDENTIFIABLE INTANGIBLE ASSETS," but its MDRM end date is
2018-03-31 (which is exactly why an exhaustive search of the real 2025Q4
bulk data found zero matches). Its direct successor, confirmed via MDRM
(start date 2018-06-30, the very next quarter): RCONJF76/RCFDJF76 (bank),
BHCKJF76 (HC), "All other intangible assets."

A user-suggested alternate provision-expense code, RIADHT74, is confirmed
via MDRM to be something else entirely -- "Noninterest income: Income
from insurance activities" (FFIEC 051 only, start date 2018-06-30) -- not
a provision item at all. The credit engine's own `PROVISION_EXPENSE_ITEM`
(RIAD4230, "Provision for Loan and Lease Losses") is correct and is also
confirmed present at the HC level as BHCK4230. Its CECL-era broader
counterpart RIADJJ33 ("Provisions for credit losses on financial assets,"
covering all financial assets in ASU 2016-13's scope, not just loans/
leases) is a superset, not a replacement -- confirmed via real 2025Q4
data for RSSD 34537: RIADJJ33 (52,522) - RIAD4230 (51,260) = 1,262,
exactly matching RIADMG93 ("Provisions for credit losses on off-balance-
sheet credit exposures"), i.e. JJ33 = 4230 + the off-balance-sheet
provision. RIAD4230/BHCK4230 is used here (not JJ33) because it matches
the credit engine's own on-balance-sheet loan-category scope exactly.

All other item codes below (total assets, other assets, total deposits,
net income, pretax income, taxes, goodwill, RWA, leverage/total/tier-1
risk-based capital ratios) were independently re-verified via the same
live MDRM search and match what was originally proposed.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field

# -- Bank-level (Call Report, RCON=domestic office/RCFD=consolidated) --
# -- items, used for the balance sheet and the reconciliation check.  --

TOTAL_ASSETS_ITEM = "RCON2170"  # MDRM: "Total assets"
OTHER_ASSETS_ITEM = "RCON2160"  # MDRM: "Other assets" -- NOT total assets
TOTAL_DEPOSITS_ITEM = "RCON2200"  # MDRM: "Total deposits"
TOTAL_LIABILITIES_ITEM = "RCON2948"  # MDRM: "Total liabilities"
TOTAL_EQUITY_CAPITAL_ITEM = "RCON3210"  # MDRM: "Total equity capital"
INTANGIBLE_ASSETS_TOTAL_ITEM = "RCON2143"  # MDRM: "Intangible assets" (goodwill + other, RC)
GOODWILL_ITEM = "RCON3163"  # MDRM: "Goodwill" (Schedule RC-M; RCON is the domestic-office
# variant -- confirmed populated for both example banks, which are FFIEC 041/051 domestic-
# only filers; RCFD3163 is the consolidated/international-filer variant, NaN for both, same
# RCON-vs-RCFD split the credit engine's own schema.py documents for other items)
OTHER_INTANGIBLES_ITEM = "RCONJF76"  # MDRM: "All other intangible assets" (RC-M; post-2018Q1
# successor to the now-retired RCFD0426/RCON0426 -- see module docstring)

NET_INTEREST_INCOME_ITEMS = ("RIAD4107", "RIAD4073")  # total interest income, total interest
# expense (RIAD4074 "net interest income" is their difference, verified exact: 836,374 -
# 279,128 = 557,246 for RSSD 34537 2025Q4; kept as two items here, not the pre-differenced
# RIAD4074, so a caller can see the gross figures too)
NONINTEREST_INCOME_ITEM = "RIAD4079"  # MDRM: "Total noninterest income"
NONINTEREST_EXPENSE_ITEM = "RIAD4093"  # MDRM: "Total noninterest expense"
PRETAX_INCOME_ITEM = "RIAD4301"  # MDRM: "Income (loss) before applicable income taxes and
# discontinued operations"
APPLICABLE_INCOME_TAXES_ITEM = "RIAD4302"  # MDRM: "Applicable income taxes"
NET_INCOME_ITEM = "RIAD4340"  # MDRM: "Net income (loss)"

# Provision expense -- shared with corefin.credit.schema.PROVISION_EXPENSE_ITEM (RIAD4230);
# imported directly rather than redefined, so the two modules can never drift apart.

# -- Holding-company level (FR Y-9C, BHCK/BHCA/BHCP prefixes). --

HC_TOTAL_ASSETS_ITEM = "BHCK2170"
HC_TOTAL_LIABILITIES_ITEM = "BHCK2948"
HC_TOTAL_EQUITY_CAPITAL_ITEM = "BHCK3210"
HC_GOODWILL_ITEM = "BHCK3163"
HC_OTHER_INTANGIBLES_ITEM = "BHCKJF76"
HC_NET_INCOME_ITEM = "BHCK4340"
HC_PRETAX_INCOME_ITEM = "BHCK4301"
HC_APPLICABLE_INCOME_TAXES_ITEM = "BHCK4302"
HC_PROVISION_EXPENSE_ITEM = "BHCK4230"

HC_CET1_CAPITAL_ITEM = "BHCAP859"  # "Common equity tier 1 capital" (numerator $, RC-R Part I)
HC_CET1_RATIO_ITEM = "BHCAP793"  # "Common equity tier 1 capital ratio"
HC_RWA_ITEM = "BHCAA223"  # "Risk-weighted assets (net of allowances and other deductions)"
HC_TIER1_LEVERAGE_RATIO_ITEM = "BHCA7204"  # "Tier 1 leverage capital ratio"
HC_TOTAL_RISK_BASED_CAPITAL_RATIO_ITEM = "BHCA7205"  # "Total risk-based capital ratio"
HC_TIER1_RISK_BASED_CAPITAL_RATIO_ITEM = "BHCA7206"  # "Tier 1 risk-based capital ratio"

# Bank-level equivalents of the RC-R Part I capital items, same MDRM item numbers under the
# RCOA (domestic) prefix -- confirmed populated for both example banks (FFIEC 041/051
# domestic-only filers; the RCFA consolidated/international-filer variant is NaN for both).
# Used only for the Call-Report-vs-Y-9C reconciliation check (model.reconcile_levels), not as
# the primary capital source. CAVEAT confirmed against real data: Schedule RC-R Part I
# percentage items (RCOAP793/RCOA7204/7205/7206) are reported with a literal trailing "%"
# ("13.8212%") in the raw bulk file -- ffiec_parse.parse_bulk_zip strips it before numeric
# coercion (see that module), so by the time these reach a DataFrame they are plain
# percentage-POINT floats (13.8212, not 0.138212) -- divide by 100 for a decimal fraction.
BANK_CET1_CAPITAL_ITEM = "RCOAP859"
BANK_CET1_RATIO_ITEM = "RCOAP793"
BANK_RWA_ITEM = "RCOAA223"
BANK_TIER1_LEVERAGE_RATIO_ITEM = "RCOA7204"


class AssetRiskCategory(StrEnum):
    """Simplified risk-weight buckets for the bottom-up RWA calculation
    (`capital.compute_rwa`) -- deliberately coarser than the real Basel III
    standardized approach (which varies by LTV band, maturity, counterparty
    rating, etc.), calibrated at the jump-off quarter so the resulting RWA
    matches the bank's own reported RWA (`HC_RWA_ITEM`) within tolerance;
    see `capital.calibrate_risk_weights`."""

    CASH = "cash"
    SECURITIES = "securities"  # AFS + HTM, blended (mostly Treasury/Agency in practice)
    RESIDENTIAL_MORTGAGE = "residential_mortgage"
    HOME_EQUITY = "home_equity"
    OTHER_LOANS = "other_loans"  # C&I, CRE, credit card, auto, other consumer
    OTHER_ASSETS = "other_assets"  # includes goodwill/intangibles, which also directly
    # reduce CET1 (see capital.compute_cet1_capital) -- not double-counted as a capital charge
    # beyond the RWA risk weight itself, consistent with the simplified-approach framing.


# Basel III standardized-approach reference points (0% sovereign/cash, 20% GSE/agency-grade,
# 50% prudently-underwritten first-lien residential mortgage, 100% the general/unrated
# corporate-and-consumer bucket) -- used as the UNCALIBRATED starting weights; the single
# scalar calibration factor in capital.calibrate_risk_weights absorbs the gap between this
# simplified table and the bank's own real asset mix/reported RWA.
DEFAULT_RISK_WEIGHTS: dict[AssetRiskCategory, float] = {
    AssetRiskCategory.CASH: 0.0,
    AssetRiskCategory.SECURITIES: 0.20,
    AssetRiskCategory.RESIDENTIAL_MORTGAGE: 0.50,
    AssetRiskCategory.HOME_EQUITY: 1.00,
    AssetRiskCategory.OTHER_LOANS: 1.00,
    AssetRiskCategory.OTHER_ASSETS: 1.00,
}


class BankConfig(BaseModel):
    """Assumptions for ONE standalone bank's statement model. Units: all
    dollar amounts are in $mm (the credit engine's own convention, per
    `corefin.credit.interface`'s DOLLARS_TO_MM); tax_rate and
    dividend_payout_ratio are decimal fractions.

    aoci_opt_out: whether unrealized AFS-security gains/losses (AOCI) are
    excluded from CET1 -- the "AOCI opt-out election" available to
    non-advanced-approaches banking organizations under the 2013 revised
    capital rules (most community/regional banks elect this; it is the
    simpler, more common treatment and avoids modeling mark-to-market
    security price paths for a capital projection). Default True.

    tax_rate: combined federal + state effective rate, per explicit user
    direction (not the 21% federal statutory rate alone). Default 0.25.
    """

    model_config = ConfigDict(extra="forbid")

    aoci_opt_out: bool = True
    tax_rate: float = Field(default=0.25, ge=0.0, lt=1.0)
    dividend_payout_ratio: float = Field(default=0.0, ge=0.0, le=1.0)
    risk_weights: dict[AssetRiskCategory, float] = Field(
        default_factory=lambda: dict(DEFAULT_RISK_WEIGHTS)
    )
    balance_sheet_growth_rate: float = Field(
        default=0.0,
        description="Per-period growth applied to deposits/securities/borrowings/other "
        "balance sheet lines when not static (loan/allowance balances always come from the "
        "credit engine's own CreditLossProjection instead, regardless of this setting; "
        "goodwill/other intangibles stay static -- see balance_sheet.py).",
    )


class BankOpeningBalance(BaseModel):
    """ONE bank's jump-off (last-actual-quarter) balance-sheet and
    reported-capital inputs -- everything `balance_sheet.project_balance_sheet`
    and `capital.calibrate_risk_weights` need beyond the credit engine's
    own `CreditLossProjection` (which supplies loan balances/allowance).
    Dollar fields are $mm; `reported_cet1_ratio` is a decimal fraction
    (0.1382, not 13.82)."""

    model_config = ConfigDict(extra="forbid")

    name: str  # "Acquirer Bank A" / "Target Bank B" in any committed/published output --
    # never a real bank name (see corefin.bank module docstring)
    bank_id: str  # Call Report RSSD ID (loan categories / credit engine level)
    hc_rssd_id: str  # FR Y-9C holding company RSSD ID (capital/RWA/earnings/equity level)

    cash_mm: float = Field(ge=0.0)
    securities_afs_mm: float = Field(ge=0.0)
    securities_htm_mm: float = Field(ge=0.0)
    other_assets_mm: float = Field(ge=0.0)
    goodwill_mm: float = Field(ge=0.0)
    other_intangibles_mm: float = Field(ge=0.0)

    deposits_mm: float = Field(ge=0.0)
    borrowings_mm: float = Field(ge=0.0)
    other_liabilities_mm: float = Field(ge=0.0)
    equity_mm: float

    # Jump-off quarter's own $ income-statement figures -- held flat (optionally grown by
    # BankConfig.balance_sheet_growth_rate) across the projection horizon by
    # income_statement.compute_income_statement; see that module's docstring.
    net_interest_income_jumpoff_mm: float
    noninterest_income_jumpoff_mm: float = Field(ge=0.0)
    noninterest_expense_jumpoff_mm: float = Field(ge=0.0)

    reported_cet1_capital_mm: float = Field(gt=0.0)
    reported_cet1_ratio: float = Field(gt=0.0, lt=1.0)
    reported_rwa_mm: float = Field(gt=0.0)
    reported_tier1_leverage_ratio: float = Field(gt=0.0, lt=1.0)
