"""Regulatory capital for ONE bank over a quarterly `Timeline`: CET1
capital, risk-weighted assets (bottom-up, simplified, calibrated), the
CET1 ratio, and the Tier 1 leverage ratio.

CET1 CAPITAL: `equity_mm - goodwill_mm - other_intangibles_mm`, plus a
single calibrated constant (`other_cet1_adjustments_mm`) solved once at
the jump-off quarter so the computed figure matches the bank's own
reported CET1 capital (`BankOpeningBalance.reported_cet1_capital_mm`)
exactly at period 0 -- the real regulatory CET1 build-up also includes
AOCI (excluded here by the AOCI-opt-out assumption -- `corefin.bank`
doesn't model unrealized AFS mark-to-market in `equity_mm` at all, so
there is nothing to add back) plus smaller items this simplified model
doesn't carry (DTAs subject to threshold deductions, MSRs, minority
interest, etc.); rather than silently ignore that gap, it is captured
once, explicitly, as a flat ongoing adjustment -- the same "calibrate
once at jump-off, hold flat" approach used for RWA below.

RWA: bottom-up over `corefin.bank.schema.AssetRiskCategory` buckets using
`BankConfig.risk_weights`, computed on NET balances (loans net of
allowance) -- matching `HC_RWA_ITEM`'s own MDRM title, "Risk-weighted
assets (net of allowances and other deductions)". Goodwill/other
intangibles are EXCLUDED from RWA entirely (0% effective weight): they
are already deducted from the CET1 numerator above, and the real Basel
III treatment excludes capital-deducted assets from RWA to avoid
double-penalizing them. A single scalar `rwa_calibration_factor`
(bottom-up jump-off RWA / reported jump-off RWA) is solved once and
applied to every period -- the simplified risk-weight table on its own
has no reason to reproduce a specific bank's real reported RWA exactly
(it deliberately doesn't vary weights by LTV band, counterparty rating,
maturity, etc.), so a single multiplicative correction is used instead of
pretending the simplified weights are literally accurate."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from corefin.bank.balance_sheet import BalanceSheet
from corefin.bank.schema import AssetRiskCategory, BankConfig, BankOpeningBalance
from corefin.timeline import Timeline

# corefin.credit.schema.LoanCategory values -> AssetRiskCategory bucket. Residential mortgage
# and home equity keep their own (lower/higher) weights; every other loan category falls into
# the single OTHER_LOANS bucket (C&I, CRE x3, credit card, auto, other consumer) -- see
# schema.py's AssetRiskCategory docstring for why this is deliberately coarser than the real
# standardized approach.
_LOAN_CATEGORY_TO_RISK_CATEGORY: dict[str, AssetRiskCategory] = {
    "residential_mortgage": AssetRiskCategory.RESIDENTIAL_MORTGAGE,
    "home_equity": AssetRiskCategory.HOME_EQUITY,
    "commercial_and_industrial": AssetRiskCategory.OTHER_LOANS,
    "cre_construction": AssetRiskCategory.OTHER_LOANS,
    "cre_multifamily": AssetRiskCategory.OTHER_LOANS,
    "cre_nonfarm_nonresidential": AssetRiskCategory.OTHER_LOANS,
    "credit_card": AssetRiskCategory.OTHER_LOANS,
    "auto": AssetRiskCategory.OTHER_LOANS,
    "other_consumer": AssetRiskCategory.OTHER_LOANS,
}


@dataclass(frozen=True)
class CapitalResult:
    timeline: Timeline
    cet1_capital_mm: np.ndarray
    rwa_mm: np.ndarray
    average_assets_mm: np.ndarray  # leverage-ratio denominator
    rwa_calibration_factor: float
    other_cet1_adjustments_mm: float

    def __post_init__(self) -> None:
        n = self.timeline.n_periods
        for name in ("cet1_capital_mm", "rwa_mm", "average_assets_mm"):
            array = getattr(self, name)
            if array.shape != (n,):
                raise ValueError(f"{name} has shape {array.shape}, expected ({n},)")

    @property
    def cet1_ratio(self) -> np.ndarray:
        return self.cet1_capital_mm / self.rwa_mm

    @property
    def tier1_leverage_ratio(self) -> np.ndarray:
        return self.cet1_capital_mm / self.average_assets_mm


def _bottom_up_rwa(
    balance_sheet: BalanceSheet, risk_weights: dict[AssetRiskCategory, float]
) -> np.ndarray:
    rwa = (
        balance_sheet.cash_mm * risk_weights[AssetRiskCategory.CASH]
        + balance_sheet.total_securities_mm * risk_weights[AssetRiskCategory.SECURITIES]
        + balance_sheet.other_assets_mm * risk_weights[AssetRiskCategory.OTHER_ASSETS]
    )
    net_loans_by_category = balance_sheet.loan_balance_mm - balance_sheet.allowance_mm
    for i, category in enumerate(balance_sheet.categories):
        risk_category = _LOAN_CATEGORY_TO_RISK_CATEGORY.get(category, AssetRiskCategory.OTHER_LOANS)
        rwa = rwa + net_loans_by_category[i] * risk_weights[risk_category]
    return rwa


def calibrate_rwa(
    balance_sheet: BalanceSheet, opening: BankOpeningBalance, config: BankConfig
) -> tuple[np.ndarray, float]:
    """Returns (calibrated rwa_mm array, calibration_factor). The jump-off
    (period 0) entry of the returned array equals `opening.reported_rwa_mm`
    exactly by construction."""
    uncalibrated = _bottom_up_rwa(balance_sheet, config.risk_weights)
    jumpoff_uncalibrated = uncalibrated[0]
    if jumpoff_uncalibrated <= 0:
        raise ValueError(
            f"bottom-up jump-off RWA was {jumpoff_uncalibrated}, cannot calibrate a scalar "
            "factor against it"
        )
    calibration_factor = opening.reported_rwa_mm / jumpoff_uncalibrated
    return uncalibrated * calibration_factor, calibration_factor


def compute_cet1_capital(
    balance_sheet: BalanceSheet, opening: BankOpeningBalance
) -> tuple[np.ndarray, float]:
    """Returns (cet1_capital_mm array, other_cet1_adjustments_mm). The
    jump-off (period 0) entry of the returned array equals
    `opening.reported_cet1_capital_mm` exactly by construction."""
    formula_capital = (
        balance_sheet.equity_mm - balance_sheet.goodwill_mm - balance_sheet.other_intangibles_mm
    )
    other_cet1_adjustments_mm = opening.reported_cet1_capital_mm - formula_capital[0]
    return formula_capital + other_cet1_adjustments_mm, other_cet1_adjustments_mm


def compute_capital(
    balance_sheet: BalanceSheet,
    opening: BankOpeningBalance,
    config: BankConfig,
    timeline: Timeline,
) -> CapitalResult:
    cet1_capital_mm, other_cet1_adjustments_mm = compute_cet1_capital(balance_sheet, opening)
    rwa_mm, rwa_calibration_factor = calibrate_rwa(balance_sheet, opening, config)
    return CapitalResult(
        timeline=timeline,
        cet1_capital_mm=cet1_capital_mm,
        rwa_mm=rwa_mm,
        average_assets_mm=balance_sheet.total_assets_mm,
        rwa_calibration_factor=rwa_calibration_factor,
        other_cet1_adjustments_mm=other_cet1_adjustments_mm,
    )
