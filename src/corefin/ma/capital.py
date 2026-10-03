"""Pro forma regulatory capital ratios AT CLOSE: RWA (acquirer's and
target's own calibrated RWA, adjusted for the loan/securities marks and
the Day-2 allowance), CET1 ratio, Tier 1 leverage ratio, and total
capital ratio.

RWA ADJUSTMENT: the credit mark, rate mark, and Day-2 allowance apply to
the target's WHOLE loan book, not broken out by `LoanCategory` (Stage 3's
deal config is a single blanket rate, not category-specific). To convert
that blanket dollar adjustment into an RWA impact, this module computes
the target's own CALIBRATED, loan-book-wide BLENDED risk weight (the
same category mix and calibration factor Stage 2 used to reproduce the
target's own reported RWA) and applies it uniformly -- more precise than
assuming a flat 100% weight for the whole adjustment, and exactly
consistent with how the target's own RWA was built in the first place.
Securities get their own (lower) calibrated weight the same way. PCD's
gross-up is excluded (confirmed net zero effect on net loans, so no RWA
impact -- see `purchase_accounting`'s module docstring); new goodwill and
CDI are excluded from RWA (same treatment as `corefin.bank.capital`'s own
RWA, which excludes capital-deducted intangibles).

TOTAL CAPITAL: Tier 2 (subordinated debt, allowance-add-back, etc.) isn't
separately modeled anywhere in this project -- each bank's own Tier 2 $
is backed out from its REPORTED total and Tier 1 risk-based ratios
(`corefin.bank.capital.compute_tier2_capital_mm`) and held static,
carried into the pro forma unchanged (a deal redeeming or assuming a
target's subordinated notes isn't modeled)."""

from __future__ import annotations

from dataclasses import dataclass

from corefin.bank.balance_sheet import BalanceSheet
from corefin.bank.capital import LOAN_CATEGORY_TO_RISK_CATEGORY, CapitalResult
from corefin.bank.schema import AssetRiskCategory, BankConfig
from corefin.ma.purchase_accounting import FairValueMarks


def compute_target_blended_loan_risk_weight(
    target_balance_sheet: BalanceSheet,
    target_capital: CapitalResult,
    target_bank_config: BankConfig,
) -> float:
    """The target's own calibrated, category-mix-weighted average risk
    weight across its WHOLE loan book at jump-off (period 0)."""
    net_loans_by_category = (
        target_balance_sheet.loan_balance_mm[:, 0] - target_balance_sheet.allowance_mm[:, 0]
    )
    total_net_loans_mm = float(net_loans_by_category.sum())
    if total_net_loans_mm <= 0:
        return 0.0
    uncalibrated_loan_rwa_mm = sum(
        net_loans_by_category[i]
        * target_bank_config.risk_weights[
            LOAN_CATEGORY_TO_RISK_CATEGORY.get(category, AssetRiskCategory.OTHER_LOANS)
        ]
        for i, category in enumerate(target_balance_sheet.categories)
    )
    calibrated_loan_rwa_mm = uncalibrated_loan_rwa_mm * target_capital.rwa_calibration_factor
    return calibrated_loan_rwa_mm / total_net_loans_mm


@dataclass(frozen=True)
class ProFormaCapitalRatios:
    pro_forma_rwa_mm: float
    loan_mark_rwa_delta_mm: float
    securities_mark_rwa_delta_mm: float
    pro_forma_average_assets_mm: float

    pro_forma_cet1_mm: float
    pro_forma_cet1_ratio: float

    pro_forma_preferred_stock_mm: float
    pro_forma_tier1_capital_mm: float
    pro_forma_tier1_leverage_ratio: float

    acquirer_tier2_mm: float | None
    target_tier2_mm: float | None
    pro_forma_tier2_mm: float | None
    pro_forma_total_capital_mm: float | None
    pro_forma_total_capital_ratio: float | None


def compute_pro_forma_capital_ratios(
    acquirer_capital: CapitalResult,
    target_capital: CapitalResult,
    target_balance_sheet: BalanceSheet,
    target_bank_config: BankConfig,
    pro_forma_cet1_mm: float,
    pro_forma_total_assets_mm: float,
    pro_forma_preferred_stock_mm: float,
    marks: FairValueMarks,
    acquirer_tier2_mm: float | None = None,
    target_tier2_mm: float | None = None,
) -> ProFormaCapitalRatios:
    """`pro_forma_cet1_mm`/`pro_forma_total_assets_mm`:
    `pro_forma.ProFormaCet1Bridge.pro_forma_cet1_mm` /
    `pro_forma.ProFormaBalanceSheet.total_assets_mm`.
    `pro_forma_preferred_stock_mm`: the acquirer's own preferred stock
    (target's is assumed redeemed at close -- see pro_forma.py).
    `acquirer_tier2_mm`/`target_tier2_mm`:
    `corefin.bank.capital.compute_tier2_capital_mm` for each bank; total
    capital fields are None if either is None (unavailable)."""
    blended_loan_weight = compute_target_blended_loan_risk_weight(
        target_balance_sheet, target_capital, target_bank_config
    )
    securities_weight = (
        target_bank_config.risk_weights[AssetRiskCategory.SECURITIES]
        * target_capital.rwa_calibration_factor
    )

    loan_mark_rwa_delta_mm = (
        marks.credit_mark_total_mm + marks.rate_mark_mm - marks.day2_allowance_non_pcd_mm
    ) * blended_loan_weight
    securities_mark_rwa_delta_mm = marks.securities_mark_mm * securities_weight

    pro_forma_rwa_mm = (
        float(acquirer_capital.rwa_mm[0])
        + float(target_capital.rwa_mm[0])
        + loan_mark_rwa_delta_mm
        + securities_mark_rwa_delta_mm
    )
    pro_forma_cet1_ratio = pro_forma_cet1_mm / pro_forma_rwa_mm
    pro_forma_tier1_capital_mm = pro_forma_cet1_mm + pro_forma_preferred_stock_mm
    pro_forma_tier1_leverage_ratio = pro_forma_tier1_capital_mm / pro_forma_total_assets_mm

    if acquirer_tier2_mm is None or target_tier2_mm is None:
        pro_forma_tier2_mm = None
        pro_forma_total_capital_mm = None
        pro_forma_total_capital_ratio = None
    else:
        pro_forma_tier2_mm = acquirer_tier2_mm + target_tier2_mm
        pro_forma_total_capital_mm = pro_forma_tier1_capital_mm + pro_forma_tier2_mm
        pro_forma_total_capital_ratio = pro_forma_total_capital_mm / pro_forma_rwa_mm

    return ProFormaCapitalRatios(
        pro_forma_rwa_mm=pro_forma_rwa_mm,
        loan_mark_rwa_delta_mm=loan_mark_rwa_delta_mm,
        securities_mark_rwa_delta_mm=securities_mark_rwa_delta_mm,
        pro_forma_average_assets_mm=pro_forma_total_assets_mm,
        pro_forma_cet1_mm=pro_forma_cet1_mm,
        pro_forma_cet1_ratio=pro_forma_cet1_ratio,
        pro_forma_preferred_stock_mm=pro_forma_preferred_stock_mm,
        pro_forma_tier1_capital_mm=pro_forma_tier1_capital_mm,
        pro_forma_tier1_leverage_ratio=pro_forma_tier1_leverage_ratio,
        acquirer_tier2_mm=acquirer_tier2_mm,
        target_tier2_mm=target_tier2_mm,
        pro_forma_tier2_mm=pro_forma_tier2_mm,
        pro_forma_total_capital_mm=pro_forma_total_capital_mm,
        pro_forma_total_capital_ratio=pro_forma_total_capital_ratio,
    )
