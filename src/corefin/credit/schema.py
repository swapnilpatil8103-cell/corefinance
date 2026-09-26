"""Loan category taxonomy and FFIEC Call Report (MDRM) item-code mappings
for the Credit-Loss Forecasting Engine's bank-quarter panel.

Every MDRM code below was looked up against the Federal Reserve's own
published MDRM Data Dictionary (a free, no-account-required 7.6MB zip at
https://www.federalreserve.gov/apps/mdrm/download_mdrm.htm -- distinct
from FFIEC's account-gated PWS), filtered to the three Call Report forms
(FFIEC 031/041/051) and to currently-active items, not recalled from
memory. Two categories are still flagged `confidence="needs_confirmation"`
-- see their `note` -- rather than silently presented as equally solid as
the rest; build the panel against those with extra care (or a sample of
real filed Call Reports) before trusting them.

RCON-prefixed items are domestic-office-only figures, the right scope for
this panel (RCFD "consolidated" figures also include foreign offices,
which the vast majority of the bank universe this project covers doesn't
have). Charge-off and recovery items (RIAD) are reported CALENDAR
YEAR-TO-DATE on the Call Report -- panel.py must difference consecutive
quarters within a calendar year to get quarterly figures, not use the
year-to-date value directly (documented there, not here).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class LoanCategory(StrEnum):
    CI = "commercial_and_industrial"
    CRE_CONSTRUCTION = "cre_construction"
    CRE_MULTIFAMILY = "cre_multifamily"
    CRE_NONFARM_NONRESIDENTIAL = "cre_nonfarm_nonresidential"
    RESIDENTIAL_MORTGAGE = "residential_mortgage"
    HOME_EQUITY = "home_equity"
    CREDIT_CARD = "credit_card"
    OTHER_CONSUMER = "other_consumer"


@dataclass(frozen=True)
class CategoryMdrmCodes:
    """`balance_items` (RC-C), `past_due_30_89_items`/`past_due_90_items`/
    `nonaccrual_items` (RC-N) and `chargeoff_items`/`recovery_items` (RI-B,
    year-to-date) are summed across their tuple when a category maps to
    more than one Call Report line (e.g. residential mortgage = first lien
    + junior lien, closed-end)."""

    balance_items: tuple[str, ...]
    past_due_30_89_items: tuple[str, ...]
    past_due_90_items: tuple[str, ...]
    nonaccrual_items: tuple[str, ...]
    chargeoff_items: tuple[str, ...]
    recovery_items: tuple[str, ...]
    confidence: str = "verified"  # "verified" or "needs_confirmation"
    note: str = ""


CATEGORY_MDRM_CODES: dict[LoanCategory, CategoryMdrmCodes] = {
    LoanCategory.CI: CategoryMdrmCodes(
        balance_items=("RCON1766",),
        past_due_30_89_items=("RCON1606",),
        past_due_90_items=("RCON1607",),
        nonaccrual_items=("RCON1608",),
        chargeoff_items=("RIAD4638",),
        recovery_items=("RIAD4608",),
    ),
    LoanCategory.CRE_CONSTRUCTION: CategoryMdrmCodes(
        balance_items=("RCON1415",),
        past_due_30_89_items=("RCON2759",),
        past_due_90_items=("RCON2769",),
        nonaccrual_items=("RCON3492",),
        chargeoff_items=("RIAD3582",),
        recovery_items=("RIAD3583",),
        note="RCON1415 is the RC-C total; RCONF158 (1-4 family residential "
        "construction) + RCONF159 (other construction/land development) sum to it "
        "on forms that report the split, if a finer breakout is ever needed.",
    ),
    LoanCategory.CRE_MULTIFAMILY: CategoryMdrmCodes(
        balance_items=("RCON1460",),
        past_due_30_89_items=("RCON3499",),
        past_due_90_items=("RCON3500",),
        nonaccrual_items=("RCON3501",),
        chargeoff_items=("RIAD3588",),
        recovery_items=("RIAD3589",),
    ),
    LoanCategory.CRE_NONFARM_NONRESIDENTIAL: CategoryMdrmCodes(
        balance_items=("RCON1480",),
        past_due_30_89_items=("RCON3502",),
        past_due_90_items=("RCON3503",),
        nonaccrual_items=("RCON3504",),
        chargeoff_items=("RIAD3590",),
        recovery_items=("RIAD3591",),
    ),
    LoanCategory.RESIDENTIAL_MORTGAGE: CategoryMdrmCodes(
        balance_items=("RCON5367", "RCON5368"),  # closed-end: first lien + junior lien
        past_due_30_89_items=("RCONC236", "RCONC238"),
        past_due_90_items=("RCONC237", "RCONC239"),
        nonaccrual_items=(),
        chargeoff_items=("RIADC234", "RIADC235"),
        recovery_items=("RIADC217", "RIADC218"),
        confidence="needs_confirmation",
        note="Nonaccrual item codes for closed-end 1-4 family (first/junior lien) "
        "not yet pinned in the MDRM search -- balance, past-due and charge-off/"
        "recovery codes are verified.",
    ),
    LoanCategory.HOME_EQUITY: CategoryMdrmCodes(
        balance_items=("RCON1797",),
        past_due_30_89_items=("RCON5398",),
        past_due_90_items=("RCON5399",),
        nonaccrual_items=("RCON5400",),
        chargeoff_items=("RIAD5411",),
        recovery_items=("RIAD5412",),
        confidence="needs_confirmation",
        note="RCON5398/5399/5400's assignment to 30-89/90+/nonaccrual is inferred "
        "from sequential numbering matching every other category's pattern (the "
        "MDRM item names were truncated mid-word in the search) -- confirm before "
        "a real fetch.",
    ),
    LoanCategory.CREDIT_CARD: CategoryMdrmCodes(
        balance_items=("RCONB538",),
        past_due_30_89_items=("RCONB575",),
        past_due_90_items=("RCONB576",),
        nonaccrual_items=("RCONB577",),
        chargeoff_items=("RIADB514",),
        recovery_items=("RIADB515",),
    ),
    LoanCategory.OTHER_CONSUMER: CategoryMdrmCodes(
        balance_items=("RCON2011",),
        past_due_30_89_items=(),
        past_due_90_items=(),
        nonaccrual_items=(),
        chargeoff_items=("RIADK205",),
        recovery_items=("RIADK206",),
        confidence="needs_confirmation",
        note="RCON2011 ('OTHER LOANS') is a low, old item code; its exact current "
        "scope (whether it precisely matches Schedule RC-C's 'other consumer' "
        "sub-item or is a broader legacy bucket) needs cross-checking against a "
        "real filed Call Report before use as a balance denominator. Charge-off/"
        "recovery codes (RIADK205/K206, explicitly 'other than credit cards') are "
        "solid; matching past-due/nonaccrual item codes weren't found.",
    ),
}

# "Allowance for Loan and Lease Losses" / "Provision for Loan and Lease Losses" --
# both item codes were carried through the 2020 CECL transition unchanged for
# adopting institutions (methodology shifts from incurred-loss ALLL to
# current-expected-credit-loss ACL under the same code); see panel.py's
# CECL-transition handling, not a code-mapping issue.
TOTAL_ALLOWANCE_ITEM = "RCON3123"
PROVISION_EXPENSE_ITEM = "RIAD4230"
