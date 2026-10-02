"""Purchase accounting at deal close: fair value marks, the PCD/Day-2
allowance split, CDI, and goodwill = consideration minus the fair value
of net assets acquired.

ORDER OF OPERATIONS AT CLOSE (per the approved plan): (1) the target's
EXISTING allowance is eliminated entirely -- loans move onto the
acquirer's books at fair value, which already reflects expected losses
through the credit mark, so the old contra-asset allowance account has
no further role. (2) PCD loans are "grossed up": their amortized cost
basis is increased by an allowance equal to their share of the credit
mark, with NO net effect on net loans, equity, or the income statement
(the gross-up and its offsetting allowance exactly cancel -- the
required "PCD has no income effect at close" property). (3) Non-PCD
loans get a NEW Day-2 CECL allowance -- unlike PCD's gross-up, this is a
real add -- POST-acquisition event booked through PROVISION EXPENSE (a
real income-statement hit, once, immediately after close), not part of
the initial fair value measurement -- the required "non-PCD Day-2
allowance hits provision expense once" property.

DEFERRED TAXES: in the common carryover-tax-basis deal structure (a
tax-free stock-for-stock reorganization, the typical structure for a
public bank-to-bank merger -- not elected explicitly here, just assumed
as the default), the TARGET's tax basis in its loans/securities carries
over unchanged even though the BOOK basis is marked to fair value. A
book write-DOWN (credit mark, and a negative rate/securities mark) with
unchanged tax basis is a deductible temporary difference -- a deferred
tax ASSET (a future tax benefit). CDI is a new, BOOK-ONLY intangible (no
tax basis at all in this structure) -- book amortization will never be
tax-deductible, a deferred tax LIABILITY. New deal GOODWILL is assumed
non-tax-deductible (standard for a stock acquisition without a 338(h)(10)
election) -- no DTL on goodwill itself."""

from __future__ import annotations

from dataclasses import dataclass

from corefin.bank.schema import BankOpeningBalance
from corefin.ma.schema import Day2AllowanceMethod, DealConfig


def compute_target_tangible_common_equity_mm(target: BankOpeningBalance) -> float:
    """GAAP/book tangible common equity -- the `ConsiderationConfig.
    price_to_tbv` base. Uses the gross carrying values of goodwill/other
    intangibles (`target.goodwill_mm`/`other_intangibles_mm`), NOT the
    regulatory net-of-DTL CET1 bridge figures -- TBV is an accounting
    concept, not a regulatory capital one."""
    return (
        target.equity_mm
        - target.preferred_stock_mm
        - target.goodwill_mm
        - target.other_intangibles_mm
    )


@dataclass(frozen=True)
class FairValueMarks:
    target_gross_loans_mm: float
    target_existing_allowance_mm: float  # eliminated entirely at close (see module docstring)

    credit_mark_total_mm: float  # negative: total loan write-down to fair value for credit risk
    credit_mark_pcd_mm: float  # negative: PCD's share of credit_mark_total_mm
    credit_mark_non_pcd_mm: float  # negative: non-PCD's share
    pcd_gross_up_mm: float  # positive: added to both PCD loans' basis AND the day-1 allowance
    day2_allowance_non_pcd_mm: float  # positive: new allowance for non-PCD loans, via provision

    rate_mark_mm: float  # signed: loan interest-rate fair value adjustment
    securities_mark_mm: float  # signed: securities fair value adjustment

    cdi_gross_mm: float  # positive: new core deposit intangible
    cdi_dtl_mm: float  # positive: deferred tax liability on CDI (see module docstring)
    cdi_net_of_dtl_mm: float

    net_dta_on_marks_mm: float  # positive if a net DTA (typical: net write-downs), negative if
    # a net DTL (net write-ups) -- combined credit/rate/securities marks, NOT CDI (tracked
    # separately above since corefin.bank.model's CET1 bridge reports it on its own line)


def compute_fair_value_marks(
    target: BankOpeningBalance,
    target_gross_loans_mm: float,
    target_existing_allowance_mm: float,
    config: DealConfig,
) -> FairValueMarks:
    credit_mark_total_mm = -config.credit_mark.credit_mark_pct * target_gross_loans_mm
    credit_mark_pcd_mm = credit_mark_total_mm * config.credit_mark.pcd_share
    credit_mark_non_pcd_mm = credit_mark_total_mm * (1.0 - config.credit_mark.pcd_share)
    pcd_gross_up_mm = -credit_mark_pcd_mm  # positive -- the gross-up exactly offsets the mark

    non_pcd_loans_mm = target_gross_loans_mm * (1.0 - config.credit_mark.pcd_share)
    if config.credit_mark.day2_allowance_method == Day2AllowanceMethod.CREDIT_MARK_RATE:
        day2_rate = config.credit_mark.credit_mark_pct
    else:
        day2_rate = (
            target_existing_allowance_mm / target_gross_loans_mm
            if target_gross_loans_mm > 0
            else 0.0
        )
    day2_allowance_non_pcd_mm = day2_rate * non_pcd_loans_mm

    rate_mark_mm = config.rate_mark.rate_mark_pct * target_gross_loans_mm
    securities_total_mm = target.securities_afs_mm + target.securities_htm_mm
    securities_mark_mm = config.securities_mark.securities_mark_pct * securities_total_mm

    cdi_gross_mm = config.cdi.cdi_pct_of_core_deposits * target.deposits_mm
    cdi_dtl_mm = cdi_gross_mm * config.tax_rate
    cdi_net_of_dtl_mm = cdi_gross_mm - cdi_dtl_mm

    net_dta_on_marks_mm = -(credit_mark_total_mm + rate_mark_mm + securities_mark_mm) * (
        config.tax_rate
    )

    return FairValueMarks(
        target_gross_loans_mm=target_gross_loans_mm,
        target_existing_allowance_mm=target_existing_allowance_mm,
        credit_mark_total_mm=credit_mark_total_mm,
        credit_mark_pcd_mm=credit_mark_pcd_mm,
        credit_mark_non_pcd_mm=credit_mark_non_pcd_mm,
        pcd_gross_up_mm=pcd_gross_up_mm,
        day2_allowance_non_pcd_mm=day2_allowance_non_pcd_mm,
        rate_mark_mm=rate_mark_mm,
        securities_mark_mm=securities_mark_mm,
        cdi_gross_mm=cdi_gross_mm,
        cdi_dtl_mm=cdi_dtl_mm,
        cdi_net_of_dtl_mm=cdi_net_of_dtl_mm,
        net_dta_on_marks_mm=net_dta_on_marks_mm,
    )


@dataclass(frozen=True)
class SourcesAndUses:
    target_tangible_common_equity_mm: float
    consideration_mm: float
    cash_consideration_mm: float
    stock_consideration_mm: float  # = new common stock issued
    fair_value_of_net_assets_acquired_mm: float
    goodwill_mm: float


def compute_sources_and_uses(
    target: BankOpeningBalance, marks: FairValueMarks, config: DealConfig
) -> SourcesAndUses:
    target_tbv_mm = compute_target_tangible_common_equity_mm(target)
    consideration_mm = config.consideration.price_to_tbv * target_tbv_mm
    cash_consideration_mm = consideration_mm * config.consideration.cash_pct
    stock_consideration_mm = consideration_mm * config.consideration.stock_pct

    fair_value_of_net_assets_acquired_mm = (
        target_tbv_mm
        + marks.target_existing_allowance_mm  # eliminated -- adds back to loan FV
        + marks.credit_mark_total_mm  # negative
        + marks.rate_mark_mm  # signed
        + marks.securities_mark_mm  # signed
        + marks.cdi_gross_mm
        + marks.net_dta_on_marks_mm
        - marks.cdi_dtl_mm
    )
    goodwill_mm = consideration_mm - fair_value_of_net_assets_acquired_mm

    return SourcesAndUses(
        target_tangible_common_equity_mm=target_tbv_mm,
        consideration_mm=consideration_mm,
        cash_consideration_mm=cash_consideration_mm,
        stock_consideration_mm=stock_consideration_mm,
        fair_value_of_net_assets_acquired_mm=fair_value_of_net_assets_acquired_mm,
        goodwill_mm=goodwill_mm,
    )


def compute_sum_of_years_digits_schedule(
    total_mm: float, life_years: float, n_periods: int, period_length_years: float
) -> list[float]:
    """Front-loaded sum-of-years-digits amortization/accretion schedule
    over `n_periods` periods of `period_length_years` each, for the
    standard bank-M&A convention (CDI amortization; also usable for
    mark accretion if a deal wants SYD instead of straight-line). Returns
    each period's own amount (not cumulative); periods beyond `life_years`
    are 0.0. Sums to `total_mm` over the periods within the life (subject
    to rounding from discretizing continuous years into periods)."""
    n_life_periods = round(life_years / period_length_years)
    if n_life_periods <= 0:
        return [0.0] * n_periods
    digit_sum = n_life_periods * (n_life_periods + 1) / 2.0
    schedule = []
    for t in range(n_periods):
        if t < n_life_periods:
            weight = (n_life_periods - t) / digit_sum
            schedule.append(total_mm * weight)
        else:
            schedule.append(0.0)
    return schedule


def compute_straight_line_schedule(
    total_mm: float, life_years: float, n_periods: int, period_length_years: float
) -> list[float]:
    """Straight-line accretion/amortization schedule -- the standard
    convention for mark accretion (as opposed to CDI's front-loaded SYD).
    Sums to `total_mm` over the periods within the life."""
    n_life_periods = round(life_years / period_length_years)
    if n_life_periods <= 0:
        return [0.0] * n_periods
    per_period = total_mm / n_life_periods
    return [per_period if t < n_life_periods else 0.0 for t in range(n_periods)]
