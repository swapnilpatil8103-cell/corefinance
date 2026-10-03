"""Orchestrates `corefin.ma`'s purchase accounting for one deal, plus the
Stage 3 integrity checks: the pro forma balance sheet balances; goodwill
equals consideration minus fair value of net assets acquired exactly;
PCD has no net effect on loans/equity at close; the pro forma CET1
bridge reconciles to the balance-sheet-derived CET1 figure.

Takes each bank's full Stage 2 `BankModelResult` (not a pile of
individually-passed floats) -- opening balance, calibrated capital, and
the jump-off `BalanceSheet` (for the target's per-category loan mix, used
by `corefin.ma.capital`'s RWA blend) all come from there."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from corefin.bank.capital import compute_tier2_capital_mm
from corefin.bank.model import BankModelResult
from corefin.bank.schema import BankConfig, BankOpeningBalance
from corefin.checks.framework import CheckResult, check_close_to_zero
from corefin.ma.accretion import (
    EpsAccretionResult,
    TbvEarnbackResult,
    compute_acquirer_irr,
    compute_eps_accretion_dilution,
    compute_tbv_dilution_and_earnback,
)
from corefin.ma.capital import ProFormaCapitalRatios, compute_pro_forma_capital_ratios
from corefin.ma.pro_forma import (
    ProFormaBalanceSheet,
    ProFormaCet1Bridge,
    compute_pro_forma_balance_sheet,
    compute_pro_forma_cet1_bridge,
)
from corefin.ma.projection import ProFormaProjection, compute_pro_forma_projection
from corefin.ma.purchase_accounting import (
    FairValueMarks,
    SourcesAndUses,
    compute_fair_value_marks,
    compute_sources_and_uses,
)
from corefin.ma.schema import DealConfig

BALANCE_SHEET_TOLERANCE_MM = 1e-6
GOODWILL_TOLERANCE_MM = 1e-6
CET1_BRIDGE_TOLERANCE_MM = 1e-6


@dataclass(frozen=True)
class DealResult:
    marks: FairValueMarks
    sources_and_uses: SourcesAndUses
    pro_forma_balance_sheet: ProFormaBalanceSheet
    pro_forma_cet1_bridge: ProFormaCet1Bridge
    pro_forma_capital_ratios: ProFormaCapitalRatios
    pro_forma_projection: ProFormaProjection
    eps_accretion: EpsAccretionResult
    tbv_earnback: TbvEarnbackResult
    acquirer_irr: float


def run_deal_model(
    acquirer_result: BankModelResult,
    target_result: BankModelResult,
    acquirer_bank_config: BankConfig,
    target_bank_config: BankConfig,
    config: DealConfig,
) -> DealResult:
    """`acquirer_result`/`target_result`: each bank's full Stage 2
    `corefin.bank.model.run_bank_model` output. `acquirer_bank_config`/
    `target_bank_config`: the `BankConfig` each bank was run with --
    needed separately (not stored on `BankModelResult`) for the
    acquirer's own `dividend_payout_ratio` (`projection.
    compute_pro_forma_projection`) and the target's RWA risk weights
    (`corefin.ma.capital`'s RWA blend, which must use the SAME weights
    Stage 2 calibrated the target's own RWA against)."""
    acquirer = acquirer_result.opening
    target = target_result.opening
    acquirer_cet1_mm = float(acquirer_result.capital.cet1_capital_mm[0])
    target_cet1_mm = float(target_result.capital.cet1_capital_mm[0])
    acquirer_net_loans_mm = float(
        acquirer_result.balance_sheet.total_loans_mm[0]
        - acquirer_result.balance_sheet.total_allowance_mm[0]
    )
    acquirer_cash_mm = float(acquirer_result.balance_sheet.cash_mm[0])
    target_cash_mm = float(target_result.balance_sheet.cash_mm[0])
    target_gross_loans_mm = float(target_result.balance_sheet.total_loans_mm[0])
    target_existing_allowance_mm = float(target_result.balance_sheet.total_allowance_mm[0])

    marks = compute_fair_value_marks(
        target, target_gross_loans_mm, target_existing_allowance_mm, config
    )
    sources_and_uses = compute_sources_and_uses(target, marks, config)
    pro_forma_balance_sheet = compute_pro_forma_balance_sheet(
        acquirer,
        target,
        acquirer_net_loans_mm,
        acquirer_cash_mm,
        target_cash_mm,
        marks,
        sources_and_uses,
        config,
    )
    pro_forma_cet1_bridge = compute_pro_forma_cet1_bridge(
        acquirer_cet1_mm, target_cet1_mm, target, marks, sources_and_uses, config
    )
    pro_forma_capital_ratios = compute_pro_forma_capital_ratios(
        acquirer_capital=acquirer_result.capital,
        target_capital=target_result.capital,
        target_balance_sheet=target_result.balance_sheet,
        target_bank_config=target_bank_config,
        pro_forma_cet1_mm=pro_forma_cet1_bridge.pro_forma_cet1_mm,
        pro_forma_total_assets_mm=pro_forma_balance_sheet.total_assets_mm,
        pro_forma_preferred_stock_mm=acquirer.preferred_stock_mm,
        marks=marks,
        acquirer_tier2_mm=compute_tier2_capital_mm(acquirer),
        target_tier2_mm=compute_tier2_capital_mm(target),
    )

    pro_forma_projection = compute_pro_forma_projection(
        acquirer_result=acquirer_result,
        target_result=target_result,
        marks=marks,
        sources_and_uses=sources_and_uses,
        pro_forma_equity_at_close_mm=pro_forma_balance_sheet.equity_mm,
        pro_forma_goodwill_at_close_mm=pro_forma_balance_sheet.goodwill_mm,
        pro_forma_other_intangibles_at_close_mm=pro_forma_balance_sheet.other_intangibles_mm,
        acquirer_bank_config=acquirer_bank_config,
        config=config,
    )
    eps_accretion = compute_eps_accretion_dilution(
        acquirer_result, pro_forma_projection, sources_and_uses, config
    )
    tbv_earnback = compute_tbv_dilution_and_earnback(
        acquirer_result, pro_forma_projection, sources_and_uses, config
    )
    acquirer_irr = compute_acquirer_irr(
        acquirer_result,
        pro_forma_projection,
        sources_and_uses,
        pro_forma_cet1_bridge.pro_forma_cet1_mm,
        pro_forma_capital_ratios.pro_forma_rwa_mm,
        config,
    )

    return DealResult(
        marks=marks,
        sources_and_uses=sources_and_uses,
        pro_forma_balance_sheet=pro_forma_balance_sheet,
        pro_forma_cet1_bridge=pro_forma_cet1_bridge,
        pro_forma_capital_ratios=pro_forma_capital_ratios,
        pro_forma_projection=pro_forma_projection,
        eps_accretion=eps_accretion,
        tbv_earnback=tbv_earnback,
        acquirer_irr=acquirer_irr,
    )


def check_pro_forma_balance_sheet_balances(
    result: DealResult, tolerance: float = BALANCE_SHEET_TOLERANCE_MM
) -> CheckResult:
    residual = np.array(
        [
            [
                result.pro_forma_balance_sheet.total_assets_mm
                - result.pro_forma_balance_sheet.total_liabilities_and_equity_mm
            ]
        ]
    )
    return check_close_to_zero("pro_forma_balance_sheet_balances", residual, tolerance)


def check_goodwill_equals_consideration_less_fair_value(
    result: DealResult, tolerance: float = GOODWILL_TOLERANCE_MM
) -> CheckResult:
    residual = np.array(
        [
            [
                result.sources_and_uses.goodwill_mm
                - (
                    result.sources_and_uses.consideration_mm
                    - result.sources_and_uses.fair_value_of_net_assets_acquired_mm
                )
            ]
        ]
    )
    return check_close_to_zero("goodwill_equals_consideration_less_fair_value", residual, tolerance)


def check_pcd_has_no_net_effect_at_close(
    result: DealResult, tolerance: float = BALANCE_SHEET_TOLERANCE_MM
) -> CheckResult:
    """The PCD gross-up and its offsetting day-1 allowance must be equal
    and opposite -- net zero effect on loans/equity/income at close."""
    residual = np.array([[result.marks.pcd_gross_up_mm + result.marks.credit_mark_pcd_mm]])
    return check_close_to_zero("pcd_has_no_net_effect_at_close", residual, tolerance)


def check_cet1_bridge_matches_balance_sheet(
    acquirer: BankOpeningBalance,
    result: DealResult,
    acquirer_unexplained_cet1_residual_mm: float = 0.0,
    tolerance: float = CET1_BRIDGE_TOLERANCE_MM,
) -> CheckResult:
    """The bridge's `pro_forma_cet1_mm` must match the SAME figure
    computed directly from the pro forma balance sheet, using the
    acquirer's own (unchanged) preferred stock/DTA-NOL/AOCI/other-
    deduction bridge items -- target's are eliminated at close, per
    `pro_forma.py`'s module docstring. New deal goodwill is assumed non-
    tax-deductible (no DTL); new CDI's DTL (`marks.cdi_dtl_mm`) is
    already embedded in the pro forma balance sheet's
    `other_liabilities_mm`, so the pro forma other-intangibles deduction
    uses CDI net of DTL directly, matching the bridge's own treatment.

    `acquirer_unexplained_cet1_residual_mm`: the acquirer's own Stage 2
    `capital.CapitalResult.unexplained_cet1_residual_mm` (default 0.0 for
    a synthetic acquirer whose explicit bridge already reconciles
    exactly). `acquirer_cet1_mm` fed into `compute_pro_forma_cet1_bridge`
    already has this baked in (it IS the acquirer's reported CET1
    capital, calibrated); this check's independent from-scratch
    recomputation must add the same residual back in for the two
    derivations to agree to the dollar, not just approximately -- for a
    real bank, that residual is itself tiny (see
    `model.check_cet1_bridge_reconciles`'s own materiality check), but
    omitting it here would otherwise show up as a spurious few-thousand-
    dollar mismatch on an otherwise-exact reconciliation."""
    bs = result.pro_forma_balance_sheet
    from_balance_sheet_mm = (
        bs.equity_mm
        - acquirer.preferred_stock_mm
        - (acquirer.goodwill_net_of_dtl_mm + result.sources_and_uses.goodwill_mm)
        - (acquirer.other_intangibles_net_of_dtl_mm + result.marks.cdi_net_of_dtl_mm)
        - acquirer.dta_nol_deduction_mm
        - (
            acquirer.aoci_afs_unrealized_mm
            + acquirer.aoci_cash_flow_hedge_mm
            + acquirer.aoci_pension_mm
            + acquirer.aoci_htm_mm
        )
        - acquirer.other_cet1_deductions_mm
        + acquirer_unexplained_cet1_residual_mm
    )
    residual = np.array([[result.pro_forma_cet1_bridge.pro_forma_cet1_mm - from_balance_sheet_mm]])
    return check_close_to_zero("cet1_bridge_matches_balance_sheet", residual, tolerance)
