"""Stage 5: stress test under a projection scenario (the Fed's severely
adverse scenario, or the baseline scenario for comparison) -- standalone
acquirer and the pro forma combined bank's minimum CET1 ratio over the
Fed's own 9-quarter DFAST reporting window
(`credit.projection.FED_COMPARISON_QUARTERS`), compared against the
Basel III 4.5% CET1 minimum (the BARE minimum -- NOT the +2.5% capital
conservation buffer; see `BASEL_III_CET1_MINIMUM_RATIO`'s own name), an
ILLUSTRATIVE stress capital buffer, cumulative PPNR/provisions/net
income, the peak-to-trough CET1 change, and how much the deal itself
changes stressed capital (standalone acquirer vs. the pro forma combined
entity).

ILLUSTRATIVE STRESS CAPITAL BUFFER: the Fed's real SCB = max(stress CET1
ratio depletion, 2.5%) + 4 quarters of planned common dividends (as a %
of RWA). This models the first (depletion-or-floor) term only -- this
project doesn't maintain a distinct "planned dividend under stress"
schedule separate from its normal payout-ratio-driven dividends -- hence
"illustrative": a lower bound on the real SCB, not a reproduction of it.

PEAK-TO-TROUGH CET1 CHANGE: the maximum DRAWDOWN in the CET1 ratio path
(the largest decline from any running peak to a later trough), NOT
simply starting-minus-minimum -- a bank whose CET1 ratio rises before
later falling would otherwise understate its own worst peak-to-trough
decline. Zero when the path never declines from its own running peak
(e.g. a bank whose CET1 ratio only ever rises under the scenario).

PPNR/PROVISION/NET INCOME: reported per-period and as 9-quarter
cumulative totals. "PPNR" (pre-provision net revenue) = pretax income
PLUS provision expense (standalone) or PLUS provision expense PLUS
`ProFormaProjection.one_time_charges_pretax_mm` (pro forma combined --
bucketing the Day-2 allowance/restructuring one-time items together
with provision as "below PPNR," a simple, documented convention rather
than decomposing that already-combined field further).

PRO FORMA COMBINED ENTITY UNDER STRESS: both banks' own standalone
`BankModelResult`s are rerun under the given scenario (the SAME credit-
engine machinery the baseline deal economics use, just a different
scenario, and with PPNR stress applied too when `bank_config.ppnr_stress`
is set -- see `corefin.bank.ppnr`), then combined via `ma.projection.
compute_pro_forma_projection` -- the SAME combination logic Stage 4
uses, fed stressed inputs. The regulatory CET1 bridge is mostly STATIC
(per `corefin.bank.schema`'s "WHICH ITEMS CHANGE IN A MERGER"), so the
stressed pro forma CET1 path only needs to track two things moving
quarter to quarter: (1) net income minus dividends (the GAAP equity
roll-forward `ProFormaProjection.equity_mm` already performs), and (2)
the new CDI intangible's own net-of-DTL balance as it amortizes --
which is REGULATORY-CAPITAL-NEUTRAL (a real accounting fact: the book
value decline and the regulatory deduction's own decline offset
exactly, confirmed algebraically here), so it is tracked separately
rather than naively carried through `equity_mm`'s own GAAP
amortization-EXPENSE effect, which would otherwise incorrectly let CDI
amortization drag CET1 down over time. RWA is held FLAT at its at-close
level for the whole stress horizon -- this project doesn't project RWA
growth beyond close anywhere (see `ma.capital`'s own module docstring),
the same already-established simplification."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from corefin.bank.model import BankModelResult
from corefin.bank.schema import BankConfig, BankOpeningBalance
from corefin.credit.interface import CreditLossProjection
from corefin.credit.projection import FED_COMPARISON_QUARTERS
from corefin.ma.horizon import build_stress_bank_result
from corefin.ma.model import DealResult
from corefin.ma.projection import compute_pro_forma_projection
from corefin.ma.schema import DealConfig

BASEL_III_CET1_MINIMUM_RATIO = 0.045
STRESS_CAPITAL_BUFFER_FLOOR = 0.025


@dataclass(frozen=True)
class StressedCet1Path:
    cet1_ratio: np.ndarray  # (n_periods,), period 0 = jump-off (not yet stressed)
    starting_cet1_ratio: float
    minimum_cet1_ratio: float
    peak_to_trough_cet1_change_pp: float  # max drawdown, in percentage points, >= 0
    breaches_4_5_pct_minimum: bool
    illustrative_stress_capital_buffer: float  # max(starting - minimum, 0.025)

    ppnr_mm: np.ndarray
    provision_mm: np.ndarray
    net_income_mm: np.ndarray

    @property
    def cumulative_ppnr_mm(self) -> float:
        return float(np.nansum(self.ppnr_mm[1:]))

    @property
    def cumulative_provision_mm(self) -> float:
        return float(np.nansum(self.provision_mm[1:]))

    @property
    def cumulative_net_income_mm(self) -> float:
        return float(np.nansum(self.net_income_mm[1:]))


def _summarize(
    cet1_ratio: np.ndarray,
    ppnr_mm: np.ndarray,
    provision_mm: np.ndarray,
    net_income_mm: np.ndarray,
) -> StressedCet1Path:
    starting = float(cet1_ratio[0])
    minimum = float(cet1_ratio.min())
    running_peak = np.maximum.accumulate(cet1_ratio)
    peak_to_trough_pp = float((running_peak - cet1_ratio).max()) * 100.0
    return StressedCet1Path(
        cet1_ratio=cet1_ratio,
        starting_cet1_ratio=starting,
        minimum_cet1_ratio=minimum,
        peak_to_trough_cet1_change_pp=peak_to_trough_pp,
        breaches_4_5_pct_minimum=minimum < BASEL_III_CET1_MINIMUM_RATIO,
        illustrative_stress_capital_buffer=max(starting - minimum, STRESS_CAPITAL_BUFFER_FLOOR),
        ppnr_mm=ppnr_mm,
        provision_mm=provision_mm,
        net_income_mm=net_income_mm,
    )


def compute_standalone_stressed_cet1_path(result: BankModelResult) -> StressedCet1Path:
    income = result.income_statement
    ppnr_mm = income.pretax_income_mm + income.provision_expense_mm
    return _summarize(
        result.capital.cet1_ratio, ppnr_mm, income.provision_expense_mm, income.net_income_mm
    )


def compute_pro_forma_stressed_cet1_path(
    acquirer_result_stressed: BankModelResult,
    target_result_stressed: BankModelResult,
    acquirer_opening: BankOpeningBalance,
    deal_result: DealResult,
    acquirer_bank_config: BankConfig,
    acquirer_unexplained_cet1_residual_mm: float,
    config: DealConfig,
) -> StressedCet1Path:
    """`deal_result`: the ORIGINAL (baseline-scenario) Stage 3/4
    `DealResult` -- its `marks`/`sources_and_uses`/
    `pro_forma_balance_sheet`/`pro_forma_capital_ratios` are all AT-CLOSE
    snapshots that don't depend on scenario, so they're reused directly
    rather than recomputed under stress."""
    stressed_projection = compute_pro_forma_projection(
        acquirer_result=acquirer_result_stressed,
        target_result=target_result_stressed,
        marks=deal_result.marks,
        sources_and_uses=deal_result.sources_and_uses,
        pro_forma_equity_at_close_mm=deal_result.pro_forma_balance_sheet.equity_mm,
        pro_forma_goodwill_at_close_mm=deal_result.pro_forma_balance_sheet.goodwill_mm,
        pro_forma_other_intangibles_at_close_mm=deal_result.pro_forma_balance_sheet.other_intangibles_mm,
        acquirer_bank_config=acquirer_bank_config,
        config=config,
    )

    # CDI's own net-of-DTL balance as it amortizes -- see module docstring for why this is
    # tracked separately rather than read off stressed_projection.other_intangibles_mm's own
    # GAAP (gross) figure directly.
    cdi_remaining_gross_mm = deal_result.marks.cdi_gross_mm - np.cumsum(
        stressed_projection.cdi_amortization_mm
    )
    other_intangibles_net_of_dtl_mm = (
        acquirer_opening.other_intangibles_net_of_dtl_mm
        + cdi_remaining_gross_mm * (1.0 - config.tax_rate)
    )
    acquirer_aoci_mm = (
        acquirer_opening.aoci_afs_unrealized_mm
        + acquirer_opening.aoci_cash_flow_hedge_mm
        + acquirer_opening.aoci_pension_mm
        + acquirer_opening.aoci_htm_mm
    )
    pro_forma_cet1_mm = (
        stressed_projection.equity_mm
        - acquirer_opening.preferred_stock_mm
        - (acquirer_opening.goodwill_net_of_dtl_mm + deal_result.sources_and_uses.goodwill_mm)
        - other_intangibles_net_of_dtl_mm
        - acquirer_opening.dta_nol_deduction_mm
        - acquirer_aoci_mm
        - acquirer_opening.other_cet1_deductions_mm
        + acquirer_unexplained_cet1_residual_mm
    )
    cet1_ratio = pro_forma_cet1_mm / deal_result.pro_forma_capital_ratios.pro_forma_rwa_mm

    provision_combined_mm = (
        acquirer_result_stressed.income_statement.provision_expense_mm
        + target_result_stressed.income_statement.provision_expense_mm
    )
    ppnr_combined_mm = (
        stressed_projection.pretax_income_mm
        + provision_combined_mm
        + stressed_projection.one_time_charges_pretax_mm
    )
    return _summarize(
        cet1_ratio, ppnr_combined_mm, provision_combined_mm, stressed_projection.net_income_mm
    )


@dataclass(frozen=True)
class StressTestResult:
    acquirer_standalone: StressedCet1Path
    pro_forma_combined: StressedCet1Path

    @property
    def minimum_cet1_ratio_change_pp(self) -> float:
        """Percentage-point change in the minimum stressed CET1 ratio,
        pro forma combined minus standalone acquirer -- how much the
        deal itself changes stressed capital adequacy, isolated from
        each bank's own standalone stress performance."""
        return (
            self.pro_forma_combined.minimum_cet1_ratio - self.acquirer_standalone.minimum_cet1_ratio
        )


def run_stress_test(
    acquirer_opening: BankOpeningBalance,
    target_opening: BankOpeningBalance,
    acquirer_scenario_projection: CreditLossProjection,
    target_scenario_projection: CreditLossProjection,
    acquirer_bank_config: BankConfig,
    target_bank_config: BankConfig,
    deal_result: DealResult,
    config: DealConfig,
    acquirer_unexplained_cet1_residual_mm: float,
    n_quarters: int = FED_COMPARISON_QUARTERS,
    jumpoff_rate_pp: float | None = None,
    projected_rate_path_pp: np.ndarray | None = None,
) -> StressTestResult:
    """Runs both banks' own scenario `CreditLossProjection` (built the
    same way the baseline deal's were, just against whichever scenario
    `acquirer_scenario_projection`/`target_scenario_projection` are --
    the Fed's severely adverse, or baseline for comparison) through the
    Fed's own 9-quarter DFAST window, and reports each entity's own
    `StressedCet1Path`. `jumpoff_rate_pp`/`projected_rate_path_pp`: that
    SAME scenario's own rate path -- see `build_stress_bank_result`'s
    own docstring; both banks see the same economy, so one rate path
    covers both."""
    acquirer_result_stressed = build_stress_bank_result(
        acquirer_opening,
        acquirer_scenario_projection,
        acquirer_bank_config,
        n_quarters,
        jumpoff_rate_pp=jumpoff_rate_pp,
        projected_rate_path_pp=projected_rate_path_pp,
    )
    target_result_stressed = build_stress_bank_result(
        target_opening,
        target_scenario_projection,
        target_bank_config,
        n_quarters,
        jumpoff_rate_pp=jumpoff_rate_pp,
        projected_rate_path_pp=projected_rate_path_pp,
    )
    acquirer_standalone = compute_standalone_stressed_cet1_path(acquirer_result_stressed)
    pro_forma_combined = compute_pro_forma_stressed_cet1_path(
        acquirer_result_stressed,
        target_result_stressed,
        acquirer_opening,
        deal_result,
        acquirer_bank_config,
        acquirer_unexplained_cet1_residual_mm,
        config,
    )
    return StressTestResult(
        acquirer_standalone=acquirer_standalone, pro_forma_combined=pro_forma_combined
    )
