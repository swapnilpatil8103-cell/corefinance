"""Stage 5: stress test under the Fed's severely adverse scenario --
standalone acquirer and the pro forma combined bank's minimum CET1
ratio over the Fed's own 9-quarter DFAST reporting window
(`credit.projection.FED_COMPARISON_QUARTERS`), compared against the
Basel III 4.5% CET1 minimum (the BARE minimum -- NOT the +2.5% capital
conservation buffer; see `BASEL_III_CET1_MINIMUM_RATIO`'s own name), an
ILLUSTRATIVE stress capital buffer, and how much the deal itself changes
stressed capital (standalone acquirer vs. the pro forma combined
entity).

ILLUSTRATIVE STRESS CAPITAL BUFFER: the Fed's real SCB = max(stress CET1
ratio depletion, 2.5%) + 4 quarters of planned common dividends (as a %
of RWA). This models the first (depletion-or-floor) term only -- this
project doesn't maintain a distinct "planned dividend under stress"
schedule separate from its normal payout-ratio-driven dividends -- hence
"illustrative": a lower bound on the real SCB, not a reproduction of it.

PRO FORMA COMBINED ENTITY UNDER STRESS: both banks' own standalone
`BankModelResult`s are rerun under the severely_adverse scenario (the
SAME credit-engine machinery the baseline deal economics use, just a
different scenario), then combined via `ma.projection.compute_pro_forma_
projection` -- the SAME combination logic Stage 4 uses, fed stressed
inputs. The regulatory CET1 bridge is mostly STATIC (per `corefin.bank.
schema`'s "WHICH ITEMS CHANGE IN A MERGER"), so the stressed pro forma
CET1 path only needs to track two things moving quarter to quarter: (1)
net income minus dividends (the GAAP equity roll-forward `ProFormaProjection.equity_mm`
already performs), and (2) the new CDI intangible's own net-of-DTL
balance as it amortizes -- which is REGULATORY-CAPITAL-NEUTRAL (a real
accounting fact: the book value decline and the regulatory deduction's
own decline offset exactly, confirmed algebraically here), so it is
tracked separately rather than naively carried through `equity_mm`'s
own GAAP amortization-EXPENSE effect, which would otherwise incorrectly
let CDI amortization drag CET1 down over time. RWA is held FLAT at its
at-close level for the whole stress horizon -- this project doesn't
project RWA growth beyond close anywhere (see `ma.capital`'s own module
docstring), the same already-established simplification."""

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
    breaches_4_5_pct_minimum: bool
    illustrative_stress_capital_buffer: float  # max(starting - minimum, 0.025)


def _summarize(cet1_ratio: np.ndarray) -> StressedCet1Path:
    starting = float(cet1_ratio[0])
    minimum = float(cet1_ratio.min())
    return StressedCet1Path(
        cet1_ratio=cet1_ratio,
        starting_cet1_ratio=starting,
        minimum_cet1_ratio=minimum,
        breaches_4_5_pct_minimum=minimum < BASEL_III_CET1_MINIMUM_RATIO,
        illustrative_stress_capital_buffer=max(starting - minimum, STRESS_CAPITAL_BUFFER_FLOOR),
    )


def compute_standalone_stressed_cet1_path(result: BankModelResult) -> StressedCet1Path:
    return _summarize(result.capital.cet1_ratio)


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
    return _summarize(cet1_ratio)


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
    acquirer_severely_adverse_projection: CreditLossProjection,
    target_severely_adverse_projection: CreditLossProjection,
    acquirer_bank_config: BankConfig,
    target_bank_config: BankConfig,
    deal_result: DealResult,
    config: DealConfig,
    acquirer_unexplained_cet1_residual_mm: float,
    n_quarters: int = FED_COMPARISON_QUARTERS,
) -> StressTestResult:
    """Runs both banks' own severely-adverse-scenario `CreditLossProjection`
    (built the same way the baseline deal's were, just against the
    severely_adverse scenario) through the Fed's own 9-quarter DFAST
    window, and reports each entity's own `StressedCet1Path`."""
    acquirer_result_stressed = build_stress_bank_result(
        acquirer_opening, acquirer_severely_adverse_projection, acquirer_bank_config, n_quarters
    )
    target_result_stressed = build_stress_bank_result(
        target_opening, target_severely_adverse_projection, target_bank_config, n_quarters
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
