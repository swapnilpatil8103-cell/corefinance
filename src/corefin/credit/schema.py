"""Loan category taxonomy and date-effective FFIEC Call Report (MDRM)
item-code mappings for the Credit-Loss Forecasting Engine's bank-quarter
panel.

Every MDRM code below was looked up against the Federal Reserve's own
published MDRM Data Dictionary (a free, no-account-required 7.6MB zip at
https://www.federalreserve.gov/apps/mdrm/download_mdrm.htm -- distinct
from FFIEC's account-gated PWS), filtered to the three Call Report forms
(FFIEC 031/041/051), not recalled from memory. Call Report item codes are
NOT stable over the panel's full history -- some categories were
restructured mid-series, so each category maps to an ordered tuple of
`MdrmCodeSet`s, each valid for a contiguous span of calendar quarters
(`code_sets_for(category)` / `code_set_for_quarter` in panel.py pick the
right one per bank-quarter row). Two confirmed, real restructurings drove
this:

- CRE construction and CRE nonfarm-nonresidential both split at 2007Q1,
  confirmed not just against MDRM but against a REAL downloaded bulk Call
  Report file (2021Q4, via the bulk-download client in sources/ffiec.py):
  the pre-2007 combined items (RCON1415 construction balance; RCON1480
  nonfarm-nonresidential balance; RCON2759/2769/3492 and RCON3502/3503/
  3504 past-due/nonaccrual; RIAD3582/3583 and RIAD3590/3591 charge-off/
  recovery) are simply ABSENT as columns from the real 2021Q4 bulk
  extract, despite MDRM listing several of them as still "active" through
  12/31/9999 -- MDRM's activity flag does not mean a column is actually
  populated in the distributed bulk data. The replacements, confirmed
  present in that same file: balance splits into owner-occupied/other
  (RCONF160/RCONF161 for nonfarm-nonresidential) or 1-4-family/other
  (RCONF158/RCONF159 for construction); past-due and nonaccrual split the
  same way (RCONF172-177 for construction, RCONF178-183 for nonfarm-
  nonresidential). Charge-offs/recoveries, however, have NO real
  replacement at this granularity: the real bulk file's RI-B schedule
  breaks out multifamily charge-offs (RIAD3588/3589) but not construction
  or nonfarm-nonresidential specifically -- the nearest related item
  (RIAD5409/5410) is a memo of CRE/construction-*purpose* charge-offs
  carved out of the C&I lines, not a breakout of RC-C's real-estate-
  secured loan categories, and does not match this project's taxonomy.
  Both categories' 2007Q1-onward code sets therefore leave
  chargeoff_items/recovery_items empty rather than pointing at codes that
  don't actually appear in the data -- a real, confirmed reporting gap,
  not an oversight. The pre-2007 code sets' charge-off/recovery items are
  not independently verified against an actual pre-2007 bulk file (only
  their MDRM-documented validity range) since the bulk-download client
  hasn't been run that far back yet.
- Consumer loans: automobile loans were not broken out as their own
  Call Report line until 2011Q1 (RCONK137 balance; RCONK213/K214/K215
  past-due/nonaccrual; RIADK129/K133 charge-off/recovery -- all verified,
  all starting 2011Q1 on FFIEC 031/041, 2017Q1 on FFIEC 051). Before that,
  auto loans were bundled into a single "loans to individuals" total
  (RIAD4639, still active but a coarser aggregate than this project's
  category taxonomy needs) with no verified per-category breakout in the
  MDRM dictionary -- LoanCategory.AUTO is therefore only populated from
  2011Q1 onward; there is no confirmed pre-2011 balance/charge-off/
  recovery mapping for it, which the panel builder must treat as a real
  missing-data period, not silently backfill.
  The "other consumer" balance is NOT the old RCON2011 ("OTHER LOANS")
  item this project originally (incorrectly) used -- RCON2011 is Schedule
  RC-C Part I item 9's non-consumer "other loans" catch-all, a different
  line entirely. The charge-off/recovery pairing (RIADK205/K206, "CHARGE-
  OFFS/RECOVERIES ALL OTHER (INCLUDES SINGLE PAYMENT, INSTALLMENT, ALL
  STUDENT LOANS, AND REVOLVING CREDIT PLANS OTHER THAN CREDIT CARDS)")
  covers TWO separate RC-C Part I balance lines -- "other revolving credit
  plans" (RCONB539) and "other consumer loans" (RCONK207) -- so
  OTHER_CONSUMER's balance sums both; using RCONK207 alone would
  understate the NCO-rate denominator relative to what the charge-off
  item actually covers.

RCON-prefixed items are domestic-office-only figures, the right scope for
this panel (RCFD "consolidated" figures also include foreign offices).
Every current category's RCON balance item was verified to be reported
continuously by FFIEC 031 (international/large bank) filers for its
entire active date range -- no RCON-to-RCFD fallback is actually
triggered by any category as of this writing. panel.py still implements
`apply_rcfd_fallback` as a defensive mechanism (exercised in tests with a
synthetic gap) for whichever category eventually needs it; `rcfd_items`
below records each code set's RCFD equivalent for that fallback to use
when needed.

Charge-off and recovery items (RIAD) are reported CALENDAR YEAR-TO-DATE
on the Call Report -- panel.py's `ytd_to_quarterly` must difference
consecutive quarters within a calendar year to get quarterly figures.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

import pandas as pd


class LoanCategory(StrEnum):
    CI = "commercial_and_industrial"
    CRE_CONSTRUCTION = "cre_construction"
    CRE_MULTIFAMILY = "cre_multifamily"
    CRE_NONFARM_NONRESIDENTIAL = "cre_nonfarm_nonresidential"
    RESIDENTIAL_MORTGAGE = "residential_mortgage"
    HOME_EQUITY = "home_equity"
    CREDIT_CARD = "credit_card"
    AUTO = "auto"
    OTHER_CONSUMER = "other_consumer"


@dataclass(frozen=True)
class MdrmCodeSet:
    """One version of a category's MDRM item-code mapping, valid for a
    contiguous, inclusive span of calendar quarters. `valid_from`/
    `valid_to` are "YYYYQN" strings; `valid_to=None` means still active.
    `balance_items` (RC-C), `past_due_30_89_items`/`past_due_90_items`/
    `nonaccrual_items` (RC-N) and `chargeoff_items`/`recovery_items`
    (RI-B, year-to-date) are summed across their tuple when more than one
    Call Report line applies. `rcfd_items` (optional) is the RCFD
    (consolidated) equivalent of `balance_items`, used only as an
    RCON-unavailable fallback for FFIEC 031 filers -- see panel.py's
    `apply_rcfd_fallback`."""

    valid_from: str
    valid_to: str | None
    balance_items: tuple[str, ...]
    past_due_30_89_items: tuple[str, ...] = ()
    past_due_90_items: tuple[str, ...] = ()
    nonaccrual_items: tuple[str, ...] = ()
    chargeoff_items: tuple[str, ...] = ()
    recovery_items: tuple[str, ...] = ()
    rcfd_items: tuple[str, ...] = ()
    confidence: str = "verified"  # "verified" or "needs_confirmation"
    note: str = ""

    def covers(self, quarter: pd.Period) -> bool:
        """`quarter`: a pandas Period (freq="Q")."""
        start = pd.Period(self.valid_from, freq="Q")
        if quarter < start:
            return False
        if self.valid_to is None:
            return True
        return quarter <= pd.Period(self.valid_to, freq="Q")


CATEGORY_MDRM_CODES: dict[LoanCategory, tuple[MdrmCodeSet, ...]] = {
    LoanCategory.CI: (
        MdrmCodeSet(
            valid_from="1984Q1",
            valid_to=None,
            balance_items=("RCON1766",),
            past_due_30_89_items=("RCON1606",),
            past_due_90_items=("RCON1607",),
            nonaccrual_items=("RCON1608",),
            chargeoff_items=("RIAD4638",),
            recovery_items=("RIAD4608",),
            rcfd_items=("RCFD1766",),
        ),
    ),
    LoanCategory.CRE_CONSTRUCTION: (
        MdrmCodeSet(
            valid_from="1984Q1",
            valid_to="2006Q4",
            balance_items=("RCON1415",),
            past_due_30_89_items=("RCON2759",),
            past_due_90_items=("RCON2769",),
            nonaccrual_items=("RCON3492",),
            chargeoff_items=("RIAD3582",),
            recovery_items=("RIAD3583",),
            confidence="needs_confirmation",
            note="Not independently verified against a real pre-2007 bulk file "
            "(only against MDRM's documented validity range) -- see the module "
            "docstring's construction/nonfarm-nonresidential note.",
        ),
        MdrmCodeSet(
            valid_from="2007Q1",
            valid_to=None,
            balance_items=("RCONF158", "RCONF159"),  # 1-4 family resi construction + other
            past_due_30_89_items=("RCONF172", "RCONF173"),
            past_due_90_items=("RCONF174", "RCONF175"),
            nonaccrual_items=("RCONF176", "RCONF177"),
            note="Verified against a real downloaded 2021Q4 bulk file: RCON1415 "
            "(combined) and RIAD3582/3583 (charge-off/recovery) are absent from the "
            "actual data from this point on, despite MDRM listing them as "
            "'active' -- replaced for balance/past-due/nonaccrual by the 1-4-family "
            "(RCONF158/F172/F174/F176) / other (RCONF159/F173/F175/F177) split. No "
            "charge-off/recovery replacement at this granularity exists in the real "
            "data -- see the module docstring; chargeoff_items/recovery_items are "
            "deliberately left empty rather than pointed at absent columns.",
        ),
    ),
    LoanCategory.CRE_MULTIFAMILY: (
        MdrmCodeSet(
            valid_from="1984Q1",
            valid_to=None,
            balance_items=("RCON1460",),
            past_due_30_89_items=("RCON3499",),
            past_due_90_items=("RCON3500",),
            nonaccrual_items=("RCON3501",),
            chargeoff_items=("RIAD3588",),
            recovery_items=("RIAD3589",),
            rcfd_items=("RCFD1460",),
        ),
    ),
    LoanCategory.CRE_NONFARM_NONRESIDENTIAL: (
        MdrmCodeSet(
            valid_from="1984Q1",
            valid_to="2006Q4",
            balance_items=("RCON1480",),
            past_due_30_89_items=("RCON3502",),
            past_due_90_items=("RCON3503",),
            nonaccrual_items=("RCON3504",),
            chargeoff_items=("RIAD3590",),
            recovery_items=("RIAD3591",),
            confidence="needs_confirmation",
            note="Not independently verified against a real pre-2007 bulk file "
            "(only against MDRM's documented validity range) -- see the module "
            "docstring's construction/nonfarm-nonresidential note.",
        ),
        MdrmCodeSet(
            valid_from="2007Q1",
            valid_to=None,
            balance_items=("RCONF160", "RCONF161"),
            past_due_30_89_items=("RCONF178", "RCONF179"),
            past_due_90_items=("RCONF180", "RCONF181"),
            nonaccrual_items=("RCONF182", "RCONF183"),
            note="Verified against a real downloaded 2021Q4 bulk file: RCON1480 "
            "(combined balance), RCON3502/3503/3504 (combined past-due/nonaccrual) "
            "and RIAD3590/3591 (charge-off/recovery) are all absent from the actual "
            "data from this point on, despite MDRM listing them as 'active' -- "
            "replaced for balance/past-due/nonaccrual by the owner-occupied "
            "(RCONF160/F178/F180/F182) / other (RCONF161/F179/F181/F183) split. No "
            "charge-off/recovery replacement at this granularity exists in the real "
            "data -- see the module docstring; chargeoff_items/recovery_items are "
            "deliberately left empty rather than pointed at absent columns.",
        ),
    ),
    LoanCategory.RESIDENTIAL_MORTGAGE: (
        MdrmCodeSet(
            valid_from="1991Q1",
            valid_to=None,
            balance_items=("RCON5367", "RCON5368"),  # closed-end: first lien + junior lien
            past_due_30_89_items=("RCONC236", "RCONC238"),
            past_due_90_items=("RCONC237", "RCONC239"),
            nonaccrual_items=("RCONC229", "RCONC230"),
            chargeoff_items=("RIADC234", "RIADC235"),
            recovery_items=("RIADC217", "RIADC218"),
            rcfd_items=("RCFD5367", "RCFD5368"),
            note="Nonaccrual codes (RCONC229 first lien, RCONC230 junior lien) "
            "confirmed active since 2002Q1 -- balance/past-due/charge-off/recovery "
            "codes predate that (1991Q1); nonaccrual for 1991Q1-2001Q4 is not "
            "independently verified and should be treated as unavailable rather "
            "than backfilled.",
        ),
    ),
    LoanCategory.HOME_EQUITY: (
        MdrmCodeSet(
            valid_from="1987Q4",
            valid_to=None,
            balance_items=("RCON1797",),
            past_due_30_89_items=("RCON5398",),
            past_due_90_items=("RCON5399",),
            nonaccrual_items=("RCON5400",),
            chargeoff_items=("RIAD5411",),
            recovery_items=("RIAD5412",),
            rcfd_items=("RCFD1797",),
            confidence="needs_confirmation",
            note="RCON5398/5399/5400's assignment to 30-89/90+/nonaccrual is inferred "
            "from sequential numbering matching every other category's pattern (the "
            "MDRM item names were truncated mid-word in the search) -- confirm "
            "before a real fetch.",
        ),
    ),
    LoanCategory.CREDIT_CARD: (
        MdrmCodeSet(
            valid_from="2001Q1",
            valid_to=None,
            balance_items=("RCONB538",),
            past_due_30_89_items=("RCONB575",),
            past_due_90_items=("RCONB576",),
            nonaccrual_items=("RCONB577",),
            chargeoff_items=("RIADB514",),
            recovery_items=("RIADB515",),
            rcfd_items=("RCFDB538",),
        ),
    ),
    LoanCategory.AUTO: (
        MdrmCodeSet(
            valid_from="2011Q1",
            valid_to=None,
            balance_items=("RCONK137",),
            past_due_30_89_items=("RCONK213",),
            past_due_90_items=("RCONK214",),
            nonaccrual_items=("RCONK215",),
            chargeoff_items=("RIADK129",),
            recovery_items=("RIADK133",),
            rcfd_items=("RCFDK137",),
            note="Automobile loans were not a separate Call Report line before "
            "2011Q1 (bundled into the pre-2011 consumer-loan totals with no "
            "verified per-category breakout) -- there is deliberately no code set "
            "covering quarters before 2011Q1; those bank-quarters are real missing "
            "data for this category, not an oversight.",
        ),
    ),
    LoanCategory.OTHER_CONSUMER: (
        MdrmCodeSet(
            valid_from="2011Q1",
            valid_to=None,
            balance_items=("RCONB539", "RCONK207"),
            chargeoff_items=("RIADK205",),
            recovery_items=("RIADK206",),
            note="Balance = RCONB539 (other revolving credit plans) + RCONK207 "
            "(other consumer loans, i.e. single payment/installment/student loans) "
            "-- this matches the exact scope described by the RIADK205/K206 "
            "charge-off/recovery item names ('...REVOLVING CREDIT PLANS OTHER THAN "
            "CREDIT CARDS'); RCONK207 alone would understate the NCO-rate "
            "denominator. No RC-N past-due/nonaccrual breakout at this granularity "
            "was found in the MDRM dictionary (only loan-modification memo items, "
            "which are not the same thing) -- left empty rather than guessed. "
            "Before 2011Q1, auto loans were bundled into this same combined "
            "consumer bucket with no verified breakout -- see LoanCategory.AUTO's "
            "note; there is no code set here for pre-2011Q1 either, for the same "
            "reason.",
        ),
    ),
}

# "Allowance for Loan and Lease Losses" / "Provision for Loan and Lease Losses" --
# both item codes were carried through the 2020 CECL transition unchanged for
# adopting institutions (methodology shifts from incurred-loss ALLL to
# current-expected-credit-loss ACL under the same code); see panel.py's
# CECL-transition handling, not a code-mapping issue.
TOTAL_ALLOWANCE_ITEM = "RCON3123"
PROVISION_EXPENSE_ITEM = "RIAD4230"
