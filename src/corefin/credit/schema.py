"""Loan category taxonomy and date-effective FFIEC Call Report (MDRM)
item-code mappings for the Credit-Loss Forecasting Engine's bank-quarter
panel.

Every MDRM code below was looked up against the Federal Reserve's own
published MDRM Data Dictionary (a free, no-account-required 7.6MB zip at
https://www.federalreserve.gov/apps/mdrm/download_mdrm.htm -- distinct
from FFIEC's account-gated PWS), filtered to the three Call Report forms
(FFIEC 031/041/051), not recalled from memory, AND cross-checked against
real bulk Call Report downloads (via sources/ffiec.py) for 2006Q4, 2008Q4,
2021Q4 and 2026Q2 -- MDRM's "active" flag on an item code does not
guarantee that code is actually a populated column in the distributed
bulk data (several codes below are MDRM-active but empirically absent;
see the per-category notes). Call Report item codes are NOT stable over
the panel's full history -- some categories were restructured mid-series,
so each category maps to an ordered tuple of `MdrmCodeSet`s, each valid
for a contiguous span of calendar quarters (`code_set_for_quarter` in
panel.py picks the right one per bank-quarter row). Three confirmed, real
restructurings drove this:

- CRE construction and CRE nonfarm-nonresidential both split at 2007Q1,
  confirmed against real 2006Q4 (pre-split codes present) and 2008Q4/
  2021Q4 (post-split codes present) bulk files -- but 2007 itself is a
  full-year TRANSITION period, not a clean cutover: verified directly
  against a real 2007Q1 bulk file, ALL 7,896 banks still report the
  combined item (RCON1480) that quarter, while only 4,724 (~60%) ALSO
  report the new split items (RCONF160/F161) -- an earlier version of
  this module wrongly assumed a clean 2007Q1 cutover (based on 2008Q4/
  2021Q4/2026Q2 data, where the combined items really are gone) and
  treated banks still using the combined codes during 2007 as missing
  data, producing spuriously low (~58-60%) coverage for all four 2007
  quarters. The fix: 2007Q1-2007Q4 get their OWN `MdrmCodeSet` whose
  PRIMARY items are the new split codes and whose `fallback_*_items` are
  the old combined codes, applied per bank-quarter row (not per quarter)
  by `panel.apply_category_mapping` -- a bank reporting the split uses
  it, a bank still on the combined item that quarter falls back to it,
  and coverage is ~100% either way. By 2008Q1 the combined items are
  confirmed fully retired (verified against 2008Q4/2021Q4/2026Q2), so the
  fallback isn't needed from then on. The replacements: balance splits
  into owner-occupied/other (RCONF160/RCONF161 for nonfarm-nonresidential)
  or 1-4-family/other (RCONF158/RCONF159 for construction); past-due and
  nonaccrual split the same way (RCONF172-177 for construction, RCONF178-
  183 for nonfarm-nonresidential); charge-offs/recoveries ALSO split the
  same way and are confirmed present (RCONF158/F159's charge-off/recovery
  counterparts RIADC891-894 for construction, RIADC895-898 for
  nonfarm-nonresidential) -- an earlier version of this module wrongly
  concluded no charge-off/recovery breakout existed at this granularity;
  it does, just under a different (C8xx) item-code range than the
  balance/past-due F-series. The SAME split-vs-combined coexistence
  pattern was verified for charge-offs/recoveries too, in the same real
  2007Q1 file: RIADC891/C893 (construction) and RIADC895/C897 (nonfarm-
  nonresidential) are each populated for only ~4,712 of 7,896 banks,
  while their combined-era predecessors RIAD3582/3583 (construction) and
  RIAD3590/3591 (nonfarm-nonresidential) are populated for all 7,896 --
  so the 2007 transition code sets' chargeoff_items/recovery_items ALSO
  carry a fallback, mirroring the balance/past-due fields exactly.
- Automobile loans were not broken out as their own Call Report line
  until 2011Q1 (RCONK137 balance; RCONK213/K214/K215 past-due/nonaccrual;
  RIADK129/K133 charge-off/recovery -- confirmed present in 2021Q4/2026Q2
  bulk data). Before that, auto loans were bundled with "other consumer"
  loans into a single combined line (see next point) -- LoanCategory.AUTO
  is therefore only populated from 2011Q1 onward.
- "Other consumer" (RC-C Part I, loans to individuals excluding credit
  cards and secured real estate): before 2011Q1, auto + other-than-credit-
  card consumer loans were ONE combined item, RCON2011 ("OTHER LOANS").
  MDRM's Description field (not just its Item Name, which is misleadingly
  generic) confirms this explicitly: "Includes all other loans to
  individuals for household, family, and other personal expenditures...
  1) purchases of private passenger automobiles... 3) educational
  expenses, including student loans..." and its own COMPARABILITY note
  states "RCON2011 derived beginning 3/31/2011. SUM(RCONK137, RCONK207)"
  -- i.e. FFIEC's own dictionary defines RCON2011, from 2011Q1 onward, AS
  the sum of auto (K137) + other-consumer (K207). An earlier version of
  this module wrongly read RCON2011's generic Item Name ("OTHER LOANS")
  as Schedule RC-C's unrelated non-consumer catch-all and left it out
  entirely -- that was a real error, corrected here. Because RCON2011
  spans BOTH auto and "other consumer" (RCONB539 other-revolving-credit +
  RCONK207 single-payment/installment/student), it is kept as a THIRD,
  DERIVED category (AUTO_AND_OTHER_CONSUMER_COMBINED) rather than folded
  into either LoanCategory.AUTO or LoanCategory.OTHER_CONSUMER -- both of
  those stay 2011Q1-onward only and continue to represent their own
  narrower slice. AUTO_AND_OTHER_CONSUMER_COMBINED runs continuously from
  2001Q1 (confirmed present in 2006Q4/2008Q4 real data: RCON2011 balance,
  RCONB578/B579/B580 past-due/nonaccrual, RIADB516/B517 charge-off/
  recovery) through the 2011Q1 switch (to RCONK137+RCONK207 balance,
  RCONK213+RCONK216 / RCONK214+RCONK217 / RCONK215+RCONK218 past-due/
  nonaccrual, RIADK129+RIADK205 / RIADK133+RIADK206 charge-off/recovery,
  per RCON2011's own MDRM COMPARABILITY note: "derived beginning 3/31/2011:
  SUM(RCONK137, RCONK207)") to today -- built specifically so a 2008-2010
  backtest has ONE continuous consumer series to work with. It OVERLAPS
  with (but is NOT identical to) AUTO + OTHER_CONSUMER from 2011Q1 onward:
  it is exactly AUTO's balance + RCONK207 (i.e. OTHER_CONSUMER's balance
  MINUS RCONB539, "other revolving credit plans"), because RCON2011's
  official derivation formula excludes RCONB539 entirely -- RCONB539 has
  been its own standalone RC-C line since 2001Q1 and was never part of
  "OTHER LOANS." Never sum this category alongside AUTO and/or
  OTHER_CONSUMER when totaling "all loan categories"; use it on its own
  for a continuous consumer series, or use AUTO/OTHER_CONSUMER separately
  for 2011Q1-onward granularity (which additionally captures RCONB539).
  OTHER_CONSUMER's own balance is RCONB539 (other revolving credit plans)
  + RCONK207 (other consumer loans) from 2011Q1 -- matching the exact
  scope of its RIADK205/K206 charge-off/recovery pairing -- and now also
  has past-due/nonaccrual (RCONK216/K217/K218, confirmed present in
  2021Q4/2026Q2 data), filling a gap an earlier version of this module
  left empty because the search that produced it didn't find these codes.
  Those past-due/nonaccrual codes cover only RCONK207's portion, though --
  MDRM has no dedicated RC-N past-due/nonaccrual item for RCONB539 at all
  (confirmed) -- so OTHER_CONSUMER's past-due/nonaccrual figures are
  partial relative to its own balance denominator; see its code set's note.

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
    # Derived, overlapping series -- see module docstring. Not part of the
    # 8-category balance-sheet taxonomy; exists only for pre-2011Q1
    # backtest continuity. Never sum alongside AUTO/OTHER_CONSUMER.
    AUTO_AND_OTHER_CONSUMER_COMBINED = "auto_and_other_consumer_combined"


@dataclass(frozen=True)
class MdrmCodeSet:
    """One version of a category's MDRM item-code mapping, valid for a
    contiguous, inclusive span of calendar quarters. `valid_from`/
    `valid_to` are "YYYYQN" strings; `valid_to=None` means still active.
    `balance_items` (RC-C), `past_due_30_89_items`/`past_due_90_items`/
    `nonaccrual_items` (RC-N) and `chargeoff_items`/`recovery_items`
    (RI-B, year-to-date) are summed across their tuple when more than one
    Call Report line applies. `rcfd_items` (optional) is the RCFD
    (consolidated) equivalent of `balance_items`, used as a fallback --
    for a bank-quarter row where the primary RCON figure is missing OR
    exactly zero, not only missing (see panel.py's `apply_rcfd_fallback`
    and its module docstring for why a zero, not just a missing value,
    needs this).

    `fallback_*_items` (optional, one per primary field): a PER-ROW
    fallback to an entirely different item code, used during a genuine
    reporting transition where some banks have switched to this code
    set's primary items and others are still using an older set -- unlike
    a plain "old code set, then new code set" quarter boundary (most
    categories), a transition quarter needs BOTH available at once,
    because which one an individual bank uses varies row by row, not
    quarter by quarter. See CRE_CONSTRUCTION/CRE_NONFARM_NONRESIDENTIAL's
    2007Q1-2007Q4 code sets for the confirmed real case this exists for."""

    valid_from: str
    valid_to: str | None
    balance_items: tuple[str, ...]
    past_due_30_89_items: tuple[str, ...] = ()
    past_due_90_items: tuple[str, ...] = ()
    nonaccrual_items: tuple[str, ...] = ()
    chargeoff_items: tuple[str, ...] = ()
    recovery_items: tuple[str, ...] = ()
    rcfd_items: tuple[str, ...] = ()
    fallback_balance_items: tuple[str, ...] = ()
    fallback_past_due_30_89_items: tuple[str, ...] = ()
    fallback_past_due_90_items: tuple[str, ...] = ()
    fallback_nonaccrual_items: tuple[str, ...] = ()
    fallback_chargeoff_items: tuple[str, ...] = ()
    fallback_recovery_items: tuple[str, ...] = ()
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
            note="Verified present in real 2006Q4/2008Q4/2021Q4 bulk data.",
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
            note="Verified present in real 2006Q4 bulk data (every item in this "
            "code set).",
        ),
        MdrmCodeSet(
            valid_from="2007Q1",
            valid_to="2007Q4",
            balance_items=("RCONF158", "RCONF159"),  # 1-4 family resi construction + other
            past_due_30_89_items=("RCONF172", "RCONF173"),
            past_due_90_items=("RCONF174", "RCONF175"),
            nonaccrual_items=("RCONF176", "RCONF177"),
            chargeoff_items=("RIADC891", "RIADC893"),
            recovery_items=("RIADC892", "RIADC894"),
            fallback_balance_items=("RCON1415",),
            fallback_past_due_30_89_items=("RCON2759",),
            fallback_past_due_90_items=("RCON2769",),
            fallback_nonaccrual_items=("RCON3492",),
            fallback_chargeoff_items=("RIAD3582",),
            fallback_recovery_items=("RIAD3583",),
            note="2007 is a full-year TRANSITION period, not a clean cutover --"
            "verified against a real 2007Q1 bulk file: the combined items (RCON1415,"
            " RCON2759/2769/3492, RIAD3582/3583) are populated for ALL 7,896 banks"
            " that quarter, while the new split items (primary, above) are populated"
            " for only ~4,712-4,722 (~60%). Every field here has a matching fallback"
            " to the pre-2007 combined item, applied per bank-quarter row by"
            " panel.apply_category_mapping -- a bank using the split reports it, a"
            " bank still on the combined item that quarter falls back to it.",
        ),
        MdrmCodeSet(
            valid_from="2008Q1",
            valid_to=None,
            balance_items=("RCONF158", "RCONF159"),
            past_due_30_89_items=("RCONF172", "RCONF173"),
            past_due_90_items=("RCONF174", "RCONF175"),
            nonaccrual_items=("RCONF176", "RCONF177"),
            chargeoff_items=("RIADC891", "RIADC893"),
            recovery_items=("RIADC892", "RIADC894"),
            note="Verified present in real 2008Q4/2021Q4 bulk data (every item in "
            "this code set, including the charge-off/recovery split RIADC891-894, "
            "found in Schedule RIBI). RCON1415/RIAD3582/RIAD3583 (pre-2007 combined "
            "items) are confirmed ABSENT from bulk data from this point on -- no "
            "fallback needed from 2008Q1, unlike the 2007 transition code set above.",
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
            note="Verified present in real 2006Q4/2008Q4/2021Q4 bulk data.",
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
            note="Verified present in real 2006Q4 bulk data (every item in this "
            "code set).",
        ),
        MdrmCodeSet(
            valid_from="2007Q1",
            valid_to="2007Q4",
            balance_items=("RCONF160", "RCONF161"),
            past_due_30_89_items=("RCONF178", "RCONF179"),
            past_due_90_items=("RCONF180", "RCONF181"),
            nonaccrual_items=("RCONF182", "RCONF183"),
            chargeoff_items=("RIADC895", "RIADC897"),
            recovery_items=("RIADC896", "RIADC898"),
            fallback_balance_items=("RCON1480",),
            fallback_past_due_30_89_items=("RCON3502",),
            fallback_past_due_90_items=("RCON3503",),
            fallback_nonaccrual_items=("RCON3504",),
            fallback_chargeoff_items=("RIAD3590",),
            fallback_recovery_items=("RIAD3591",),
            note="2007 is a full-year TRANSITION period, not a clean cutover --"
            "verified against a real 2007Q1 bulk file: the combined items (RCON1480,"
            " RCON3502/3503/3504, RIAD3590/3591) are populated for ALL 7,896 banks"
            " that quarter, while the new split items (primary, above) are populated"
            " for only ~4,712-4,717 (~60%). Every field here has a matching fallback"
            " to the pre-2007 combined item, applied per bank-quarter row by"
            " panel.apply_category_mapping -- a bank using the split reports it, a"
            " bank still on the combined item that quarter falls back to it.",
        ),
        MdrmCodeSet(
            valid_from="2008Q1",
            valid_to=None,
            balance_items=("RCONF160", "RCONF161"),
            past_due_30_89_items=("RCONF178", "RCONF179"),
            past_due_90_items=("RCONF180", "RCONF181"),
            nonaccrual_items=("RCONF182", "RCONF183"),
            chargeoff_items=("RIADC895", "RIADC897"),
            recovery_items=("RIADC896", "RIADC898"),
            note="Verified present in real 2008Q4/2021Q4 bulk data (every item in "
            "this code set, including the charge-off/recovery split RIADC895-898, "
            "found in Schedule RIBI). RCON1480/RCON3502-3504/RIAD3590/RIAD3591 "
            "(pre-2007 combined items) are confirmed ABSENT from bulk data from "
            "this point on -- no fallback needed from 2008Q1, unlike the 2007 "
            "transition code set above.",
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
            note="Verified present in real 2006Q4/2008Q4/2021Q4 bulk data (every "
            "item in this code set, including nonaccrual). Nonaccrual codes "
            "(RCONC229 first lien, RCONC230 junior lien) are MDRM-documented active "
            "since 2002Q1 only -- balance/past-due/charge-off/recovery codes "
            "predate that (1991Q1); nonaccrual for 1991Q1-2001Q4 is not "
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
            note="RCON5398/5399/5400's assignment to 30-89/90+/nonaccrual confirmed "
            "via their full (untruncated) MDRM item names ('...PAST DUE 30 THROUGH "
            "89 DAYS AND STILL ACCRUING' / '...PAST DUE 90 DAYS OR MORE...' / "
            "'...NONACCRUAL' respectively) and verified present in real 2006Q4/"
            "2008Q4/2021Q4 bulk data.",
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
            note="Verified present in real 2006Q4/2008Q4/2021Q4 bulk data.",
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
            note="Verified present in real 2021Q4/2026Q2 bulk data. Automobile "
            "loans were not a separate Call Report line before 2011Q1 (bundled "
            "into RCON2011 -- see LoanCategory.AUTO_AND_OTHER_CONSUMER_COMBINED for "
            "a continuous pre/post-2011 series) -- there is deliberately no code "
            "set here covering quarters before 2011Q1.",
        ),
    ),
    LoanCategory.OTHER_CONSUMER: (
        MdrmCodeSet(
            valid_from="2011Q1",
            valid_to=None,
            balance_items=("RCONB539", "RCONK207"),
            past_due_30_89_items=("RCONK216",),
            past_due_90_items=("RCONK217",),
            nonaccrual_items=("RCONK218",),
            chargeoff_items=("RIADK205",),
            recovery_items=("RIADK206",),
            note="Verified present in real 2021Q4/2026Q2 bulk data (every item in "
            "this code set, including the RCONK216/K217/K218 past-due/nonaccrual "
            "codes an earlier version of this module failed to find). Balance = "
            "RCONB539 (other revolving credit plans) + RCONK207 (other consumer "
            "loans, i.e. single payment/installment/student loans) -- matches the "
            "exact scope described by the RIADK205/K206 charge-off/recovery item "
            "names ('...REVOLVING CREDIT PLANS OTHER THAN CREDIT CARDS'); RCONK207 "
            "alone would understate the NCO-rate denominator. CAVEAT: "
            "past_due_30_89/90/nonaccrual (RCONK216/K217/K218) cover ONLY RCONK207's "
            "portion -- MDRM has no dedicated RC-N past-due/nonaccrual item for "
            "RCONB539 (confirmed: no such item exists), so the past-due/nonaccrual "
            "figures here are partial relative to the balance denominator, not a "
            "like-for-like NPL ratio numerator. Before 2011Q1, auto + other consumer "
            "were one combined line (RCON2011, which itself excludes RCONB539 -- "
            "see LoanCategory.AUTO_AND_OTHER_CONSUMER_COMBINED's note) -- there is "
            "deliberately no code set here covering quarters before 2011Q1.",
        ),
    ),
    LoanCategory.AUTO_AND_OTHER_CONSUMER_COMBINED: (
        MdrmCodeSet(
            valid_from="2001Q1",
            valid_to="2010Q4",
            balance_items=("RCON2011",),
            past_due_30_89_items=("RCONB578",),
            past_due_90_items=("RCONB579",),
            nonaccrual_items=("RCONB580",),
            chargeoff_items=("RIADB516",),
            recovery_items=("RIADB517",),
            note="Verified present in real 2006Q4/2008Q4 bulk data (every item in "
            "this code set). RCON2011's own MDRM Description confirms it covers "
            "'all other loans to individuals for household, family, and other "
            "personal expenditures' including automobiles and student loans -- an "
            "earlier version of this module misread its generic Item Name ('OTHER "
            "LOANS') as an unrelated non-consumer catch-all; that was a real error.",
        ),
        MdrmCodeSet(
            valid_from="2011Q1",
            valid_to=None,
            balance_items=("RCONK137", "RCONK207"),
            past_due_30_89_items=("RCONK213", "RCONK216"),
            past_due_90_items=("RCONK214", "RCONK217"),
            nonaccrual_items=("RCONK215", "RCONK218"),
            chargeoff_items=("RIADK129", "RIADK205"),
            recovery_items=("RIADK133", "RIADK206"),
            note="Balance = RCONK137 + RCONK207 exactly, per MDRM's own "
            "COMPARABILITY note for RCON2011: 'derived beginning 3/31/2011: "
            "SUM(RCONK137, RCONK207)' -- confirming this is the true continuation "
            "of the pre-2011 combined series, not a coincidental reconstruction. "
            "NOTE this does NOT equal LoanCategory.AUTO + LoanCategory.OTHER_CONSUMER "
            "summed: OTHER_CONSUMER's own balance also includes RCONB539 (other "
            "revolving credit plans, needed there to match its RIADK205/K206 "
            "charge-off scope), which RCON2011's official derivation formula "
            "excludes. This code set's past-due/nonaccrual (K213+K216/K214+K217/"
            "K215+K218) and charge-off/recovery (K129+K205/K133+K206) mirror that "
            "same K137+K207-only scope, consistent with the balance. Overlaps (but "
            "is not identical to) AUTO + OTHER_CONSUMER from this quarter onward -- "
            "see module docstring.",
        ),
    ),
}

# "Allowance for Loan and Lease Losses" / "Provision for Loan and Lease Losses" --
# both item codes were carried through the 2020 CECL transition unchanged for
# adopting institutions (methodology shifts from incurred-loss ALLL to
# current-expected-credit-loss ACL under the same code); see panel.py's
# CECL-transition handling, not a code-mapping issue. Both verified present in
# real 2006Q4/2008Q4/2021Q4 bulk data.
TOTAL_ALLOWANCE_ITEM = "RCON3123"
PROVISION_EXPENSE_ITEM = "RIAD4230"

# Bank-level TOTAL charge-offs/recoveries -- Schedule RI-B Part II (the
# "Allowance" memo section reported alongside RCON3123/RIAD4230/RIAD4605),
# NOT the sum of this project's mapped loan categories, which is the right
# denominator for the allowance roll-forward identity. Schedule RI-B has
# several charge-off items (lease financing, farmland loans, loans to
# foreign governments, and more) this project doesn't map into any
# LoanCategory at all, so the mapped-category sum understates the true
# total by construction -- see panel.py's
# `compute_bank_allowance_rollforward`, which keeps both figures.
#
# RI-B actually has TWO charge-off total lines, both verified present in
# real 2006Q4/2021Q4 bulk data: RIAD4635 (Part I, in Schedule RIBI) and
# RIADC079 (Part II, in Schedule RIBII). They are usually identical but not
# always -- verified against real 2021Q4 data, they differ for 29 of 4887
# banks, and RIADC079 is always >= RIAD4635 where they differ. This matches
# MDRM's own COMPARABILITY note for RIAD4635/C079: beginning 2001Q2, "the
# item number was changed to C079 from 4635; the definition was changed to
# include write-down[s] arising from transfers of loans to a held-for-sale
# account" -- i.e. C079 is a superset covering banks with held-for-sale
# transfers that quarter. RIADC079 (Part II) is used here since it is the
# figure reported alongside the allowance roll-forward itself, in the same
# schedule as RCON3123/RIAD4230. RIAD4605 (recoveries) has no equivalent
# ambiguity -- verified byte-for-byte identical whether read from Schedule
# RIBI or RIBII, across all 4887 banks in the 2021Q4 file.
TOTAL_CHARGEOFF_ITEM = "RIADC079"
TOTAL_RECOVERY_ITEM = "RIAD4605"

# CECL (ASU 2016-13) adoption indicators -- both confirmed present in real
# 2020Q1/2023Q1 bulk data, both ItemType "F" (a reported dollar figure, not
# a boolean flag despite the names): RIADJJ26 "Adoption of Current Expected
# Credit Losses Methodology" and its companion RIADJJ28 "Effect of adoption
# of current expected credit losses methodology on allowances...". Per
# MDRM, both are valid ONLY 2019Q1-2023Q4 (the item was retired once every
# bank had transitioned) and, per the Call Report instructions, are
# reported non-zero starting the bank's own adoption quarter -- confirmed
# against real data: 220 of 5167 banks show a nonzero RIADJJ26 in 2020Q1
# (the mandatory date for large SEC filers) and a much larger cohort
# (~1800-1900 of ~4700) from 2023Q1 onward (the final mandatory date for
# smaller/private companies), matching ASU 2016-13's known two-wave
# rollout. See panel.py's `derive_cecl_adoption_quarters`, which takes the
# first quarter either item is nonzero as that bank's adoption quarter.
CECL_ADOPTION_INDICATOR_ITEMS = ("RIADJJ26", "RIADJJ28")
