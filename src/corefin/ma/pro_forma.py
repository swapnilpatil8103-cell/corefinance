"""The pro forma combined balance sheet and CET1 bridge AT DEAL CLOSE.

A NOTE ON THE CET1 BRIDGE'S SIGN CONVENTION (flagged explicitly, not
just in code comments, because it is a real judgment call): the
mathematically consistent walk, independently verified three ways here
(direct balance-sheet construction, a fair-value-net-assets substitution,
and the `test_ma_pro_forma.py` reconciliation test), is

    pro_forma_cet1 = acquirer_cet1
                     + new_common_stock_issued
                     - new_deal_goodwill
                     - new_cdi_net_of_dtl
                     - day2_allowance_after_tax
                     - restructuring_charge_after_tax

TARGET'S OWN STANDALONE CET1 DOES NOT APPEAR AS A SEPARATE ADDITIVE TERM.
This is not an oversight -- literally adding target's standalone CET1
capital alongside the NEW goodwill/CDI double-counts target's net worth:
the target's balance sheet is economically replaced by what the acquirer
PAYS for it (cash + new stock, split via `ConsiderationConfig.stock_pct`)
and by goodwill (the plug between that price and the fair value of what
was acquired) -- target's own capital doesn't carry over as a separate
line. Confirmed by a clean balance-sheet identity: for the pro forma
balance sheet to balance, equity changes by EXACTLY the stock
consideration issued, nothing else (cash consideration is funded by
assets leaving, not a capital-side item).

`target_contribution_mm` and its components are still computed and
reported, as a MEMO reconciliation (what a simple book-value "pooling"
of the two balance sheets, before any fair-value remeasurement, would
have contributed) -- useful context, but NOT part of the arithmetic that
sums to `pro_forma_cet1_mm`."""

from __future__ import annotations

from dataclasses import dataclass

from corefin.bank.schema import BankOpeningBalance
from corefin.ma.purchase_accounting import FairValueMarks, SourcesAndUses
from corefin.ma.schema import DealConfig


@dataclass(frozen=True)
class ProFormaBalanceSheet:
    cash_mm: float
    securities_mm: float
    net_loans_mm: float
    goodwill_mm: float
    other_intangibles_mm: float
    other_assets_mm: float
    deposits_mm: float
    borrowings_mm: float
    other_liabilities_mm: float
    equity_mm: float

    @property
    def total_assets_mm(self) -> float:
        return (
            self.cash_mm
            + self.securities_mm
            + self.net_loans_mm
            + self.goodwill_mm
            + self.other_intangibles_mm
            + self.other_assets_mm
        )

    @property
    def total_liabilities_mm(self) -> float:
        return self.deposits_mm + self.borrowings_mm + self.other_liabilities_mm

    @property
    def total_liabilities_and_equity_mm(self) -> float:
        return self.total_liabilities_mm + self.equity_mm


def compute_pro_forma_balance_sheet(
    acquirer: BankOpeningBalance,
    target: BankOpeningBalance,
    acquirer_net_loans_mm: float,
    acquirer_cash_mm: float,
    target_cash_mm: float,
    marks: FairValueMarks,
    sources_and_uses: SourcesAndUses,
    config: DealConfig,
) -> ProFormaBalanceSheet:
    """`acquirer_net_loans_mm`: the acquirer's own jump-off net loan total
    (gross minus allowance) from its `CreditLossProjection` -- same figure
    `sources.real_data.build_opening_balance_from_real_data`'s
    `bank_level_net_loans_mm` parameter supplies for Stage 2.

    `acquirer_cash_mm`/`target_cash_mm`: each bank's own jump-off cash,
    from `corefin.bank.balance_sheet.BalanceSheet.cash_mm[0]` (i.e. AFTER
    Stage 2's own cash-balancing-plug logic) -- NOT simply
    `BankOpeningBalance.cash_mm`. Each bank's own REPORTED total assets
    and REPORTED total liabilities + equity can differ by a tiny amount
    (confirmed against real 2025Q4 data -- roughly $0.1mm on a ~$20B
    balance sheet, an FFIEC/Y-9C filing-level rounding artifact, not a
    modeling error); Stage 2's own cash plug silently absorbs that gap for
    each bank standalone, and using its OUTPUT here (rather than the raw
    reported cash figure) carries that same absorption into the combined
    pro forma balance sheet instead of letting the gap leak back in."""
    restructuring_charge_mm = config.cost_saves.restructuring_charge_mm
    tax_benefit_restructuring_mm = restructuring_charge_mm * config.tax_rate
    tax_benefit_day2_mm = marks.day2_allowance_non_pcd_mm * config.tax_rate

    cash_mm = (
        acquirer_cash_mm
        + target_cash_mm
        - sources_and_uses.cash_consideration_mm
        - restructuring_charge_mm  # paid in cash at close -- a documented simplification
        - target.preferred_stock_mm  # assumed REDEEMED for cash at close -- see module
        # docstring. Consideration is priced off target's TANGIBLE COMMON equity (excludes
        # preferred), so target's preferred claim must be settled separately, not silently
        # dropped -- confirmed by a balance-sheet identity that otherwise leaves a residual of
        # exactly target.preferred_stock_mm. Zero for both example banks' target (no preferred
        # stock), so this doesn't change the example deal's numbers.
    )
    securities_mm = (
        acquirer.securities_afs_mm
        + acquirer.securities_htm_mm
        + target.securities_afs_mm
        + target.securities_htm_mm
        + marks.securities_mark_mm
    )
    target_loans_fair_value_mm = (
        marks.target_gross_loans_mm + marks.credit_mark_total_mm + marks.rate_mark_mm
    )
    net_loans_mm = (
        acquirer_net_loans_mm + target_loans_fair_value_mm - marks.day2_allowance_non_pcd_mm
    )
    goodwill_mm = acquirer.goodwill_mm + sources_and_uses.goodwill_mm
    other_intangibles_mm = acquirer.other_intangibles_mm + marks.cdi_gross_mm
    other_assets_mm = (
        acquirer.other_assets_mm
        + target.other_assets_mm
        + marks.net_dta_on_marks_mm
        + tax_benefit_day2_mm
        + tax_benefit_restructuring_mm
    )
    deposits_mm = acquirer.deposits_mm + target.deposits_mm
    borrowings_mm = acquirer.borrowings_mm + target.borrowings_mm
    other_liabilities_mm = (
        acquirer.other_liabilities_mm + target.other_liabilities_mm + marks.cdi_dtl_mm
    )
    equity_mm = (
        acquirer.equity_mm
        + sources_and_uses.stock_consideration_mm
        - (restructuring_charge_mm - tax_benefit_restructuring_mm)
        - (marks.day2_allowance_non_pcd_mm - tax_benefit_day2_mm)
    )
    return ProFormaBalanceSheet(
        cash_mm=cash_mm,
        securities_mm=securities_mm,
        net_loans_mm=net_loans_mm,
        goodwill_mm=goodwill_mm,
        other_intangibles_mm=other_intangibles_mm,
        other_assets_mm=other_assets_mm,
        deposits_mm=deposits_mm,
        borrowings_mm=borrowings_mm,
        other_liabilities_mm=other_liabilities_mm,
        equity_mm=equity_mm,
    )


@dataclass(frozen=True)
class ProFormaCet1Bridge:
    acquirer_cet1_mm: float

    # MEMO ONLY -- see module docstring. Not part of the pro_forma_cet1_mm arithmetic.
    target_cet1_mm: float
    target_goodwill_reversed_mm: float
    target_other_intangibles_reversed_mm: float
    target_aoci_reversed_mm: float
    target_contribution_memo_mm: float

    new_common_stock_issued_mm: float
    new_deal_goodwill_mm: float
    new_cdi_net_of_dtl_mm: float
    day2_allowance_after_tax_mm: float
    restructuring_charge_after_tax_mm: float
    pro_forma_cet1_mm: float


def compute_pro_forma_cet1_bridge(
    acquirer_cet1_mm: float,
    target_cet1_mm: float,
    target: BankOpeningBalance,
    marks: FairValueMarks,
    sources_and_uses: SourcesAndUses,
    config: DealConfig,
) -> ProFormaCet1Bridge:
    target_goodwill_reversed_mm = target.goodwill_net_of_dtl_mm
    target_other_intangibles_reversed_mm = target.other_intangibles_net_of_dtl_mm
    target_aoci_reversed_mm = (
        target.aoci_afs_unrealized_mm
        + target.aoci_cash_flow_hedge_mm
        + target.aoci_pension_mm
        + target.aoci_htm_mm
    )
    target_contribution_memo_mm = (
        target_cet1_mm
        + target_goodwill_reversed_mm
        + target_other_intangibles_reversed_mm
        + target_aoci_reversed_mm
    )

    day2_allowance_after_tax_mm = marks.day2_allowance_non_pcd_mm * (1.0 - config.tax_rate)
    restructuring_charge_after_tax_mm = config.cost_saves.restructuring_charge_mm * (
        1.0 - config.tax_rate
    )

    pro_forma_cet1_mm = (
        acquirer_cet1_mm
        + sources_and_uses.stock_consideration_mm
        - sources_and_uses.goodwill_mm
        - marks.cdi_net_of_dtl_mm
        - day2_allowance_after_tax_mm
        - restructuring_charge_after_tax_mm
    )

    return ProFormaCet1Bridge(
        acquirer_cet1_mm=acquirer_cet1_mm,
        target_cet1_mm=target_cet1_mm,
        target_goodwill_reversed_mm=target_goodwill_reversed_mm,
        target_other_intangibles_reversed_mm=target_other_intangibles_reversed_mm,
        target_aoci_reversed_mm=target_aoci_reversed_mm,
        target_contribution_memo_mm=target_contribution_memo_mm,
        new_common_stock_issued_mm=sources_and_uses.stock_consideration_mm,
        new_deal_goodwill_mm=sources_and_uses.goodwill_mm,
        new_cdi_net_of_dtl_mm=marks.cdi_net_of_dtl_mm,
        day2_allowance_after_tax_mm=day2_allowance_after_tax_mm,
        restructuring_charge_after_tax_mm=restructuring_charge_after_tax_mm,
        pro_forma_cet1_mm=pro_forma_cet1_mm,
    )
