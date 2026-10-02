"""Offline tests for corefin.ma.pro_forma -- synthetic data only."""

from __future__ import annotations

import numpy as np
import pytest

from corefin.bank.balance_sheet import BalanceSheet
from corefin.bank.capital import compute_cet1_capital
from corefin.bank.schema import BankOpeningBalance
from corefin.ma.pro_forma import compute_pro_forma_balance_sheet, compute_pro_forma_cet1_bridge
from corefin.ma.purchase_accounting import compute_fair_value_marks, compute_sources_and_uses
from corefin.ma.schema import (
    CdiConfig,
    ConsiderationConfig,
    CostSaveConfig,
    CreditMarkConfig,
    DealConfig,
    RateMarkConfig,
    SecuritiesMarkConfig,
)
from corefin.timeline import Timeline

_TIMELINE = Timeline.quarterly(1, n_historical=0)


def _balanced_opening(
    name: str,
    bank_id: str,
    hc_rssd_id: str,
    net_loans_mm: float,
    equity_mm: float = 1000.0,
    goodwill_mm: float = 50.0,
    other_intangibles_mm: float = 10.0,
    preferred_stock_mm: float = 0.0,
    deposits_mm: float = 3000.0,
    borrowings_mm: float = 200.0,
    other_liabilities_mm: float = 50.0,
    securities_mm: float = 500.0,
    other_assets_mm: float = 100.0,
) -> BankOpeningBalance:
    cash_mm = (deposits_mm + borrowings_mm + other_liabilities_mm + equity_mm) - (
        securities_mm + net_loans_mm + goodwill_mm + other_intangibles_mm + other_assets_mm
    )
    return BankOpeningBalance(
        name=name,
        bank_id=bank_id,
        hc_rssd_id=hc_rssd_id,
        cash_mm=cash_mm,
        securities_afs_mm=securities_mm / 2,
        securities_htm_mm=securities_mm / 2,
        other_assets_mm=other_assets_mm,
        goodwill_mm=goodwill_mm,
        other_intangibles_mm=other_intangibles_mm,
        deposits_mm=deposits_mm,
        borrowings_mm=borrowings_mm,
        other_liabilities_mm=other_liabilities_mm,
        equity_mm=equity_mm,
        net_interest_income_jumpoff_mm=40.0,
        noninterest_income_jumpoff_mm=10.0,
        noninterest_expense_jumpoff_mm=30.0,
        preferred_stock_mm=preferred_stock_mm,
        goodwill_net_of_dtl_mm=goodwill_mm,
        other_intangibles_net_of_dtl_mm=other_intangibles_mm * 0.7,
        dta_nol_deduction_mm=1.0,
        aoci_afs_unrealized_mm=-5.0,
        reported_cet1_capital_mm=(
            equity_mm - preferred_stock_mm - goodwill_mm - other_intangibles_mm * 0.7 - 1.0 + 5.0
        ),
        reported_cet1_ratio=0.12,
        reported_rwa_mm=5000.0,
        reported_tier1_leverage_ratio=0.09,
    )


def _cet1_of(bank: BankOpeningBalance) -> float:
    bs = BalanceSheet(
        timeline=_TIMELINE,
        categories=(),
        loan_balance_mm=np.zeros((0, 1)),
        allowance_mm=np.zeros((0, 1)),
        cash_mm=np.array([bank.cash_mm]),
        securities_afs_mm=np.array([bank.securities_afs_mm]),
        securities_htm_mm=np.array([bank.securities_htm_mm]),
        other_assets_mm=np.array([bank.other_assets_mm]),
        goodwill_mm=np.array([bank.goodwill_mm]),
        other_intangibles_mm=np.array([bank.other_intangibles_mm]),
        deposits_mm=np.array([bank.deposits_mm]),
        borrowings_mm=np.array([bank.borrowings_mm]),
        other_liabilities_mm=np.array([bank.other_liabilities_mm]),
        equity_mm=np.array([bank.equity_mm]),
    )
    cet1, _ = compute_cet1_capital(bs, bank)
    return float(cet1[0])


def _deal_config(**overrides) -> DealConfig:
    kwargs: dict = dict(
        consideration=ConsiderationConfig(price_to_tbv=1.5, stock_pct=0.8),
        credit_mark=CreditMarkConfig(credit_mark_pct=0.02, pcd_share=0.3),
    )
    kwargs.update(overrides)
    return DealConfig(**kwargs)


def test_pro_forma_balance_sheet_balances():
    acquirer = _balanced_opening(
        "Acquirer Bank A", "111", "222", net_loans_mm=2000.0, preferred_stock_mm=100.0
    )
    target = _balanced_opening(
        "Target Bank B",
        "333",
        "444",
        net_loans_mm=784.0,
        equity_mm=400.0,
        deposits_mm=1200.0,
        borrowings_mm=50.0,
        other_liabilities_mm=20.0,
        securities_mm=200.0,
        other_assets_mm=40.0,
    )
    config = _deal_config(
        rate_mark=RateMarkConfig(rate_mark_pct=-0.01),
        securities_mark=SecuritiesMarkConfig(securities_mark_pct=-0.005),
        cost_saves=CostSaveConfig(restructuring_charge_mm=20.0),
    )
    marks = compute_fair_value_marks(target, 800.0, 16.0, config)
    sources_and_uses = compute_sources_and_uses(target, marks, config)
    bs = compute_pro_forma_balance_sheet(
        acquirer, target, 2000.0, acquirer.cash_mm, target.cash_mm, marks, sources_and_uses, config
    )

    assert bs.total_assets_mm == pytest.approx(bs.total_liabilities_and_equity_mm, abs=1e-6)


def test_pro_forma_cet1_bridge_matches_balance_sheet_derivation():
    acquirer = _balanced_opening(
        "Acquirer Bank A", "111", "222", net_loans_mm=2000.0, preferred_stock_mm=100.0
    )
    target = _balanced_opening(
        "Target Bank B",
        "333",
        "444",
        net_loans_mm=784.0,
        equity_mm=400.0,
        deposits_mm=1200.0,
        borrowings_mm=50.0,
        other_liabilities_mm=20.0,
        securities_mm=200.0,
        other_assets_mm=40.0,
    )
    config = _deal_config(
        rate_mark=RateMarkConfig(rate_mark_pct=-0.01),
        securities_mark=SecuritiesMarkConfig(securities_mark_pct=-0.005),
        cost_saves=CostSaveConfig(restructuring_charge_mm=20.0),
    )
    acquirer_cet1_mm = _cet1_of(acquirer)
    target_cet1_mm = _cet1_of(target)
    marks = compute_fair_value_marks(target, 800.0, 16.0, config)
    sources_and_uses = compute_sources_and_uses(target, marks, config)
    bs = compute_pro_forma_balance_sheet(
        acquirer, target, 2000.0, acquirer.cash_mm, target.cash_mm, marks, sources_and_uses, config
    )
    bridge = compute_pro_forma_cet1_bridge(
        acquirer_cet1_mm, target_cet1_mm, target, marks, sources_and_uses, config
    )

    from_balance_sheet_mm = (
        bs.equity_mm
        - acquirer.preferred_stock_mm
        - (acquirer.goodwill_net_of_dtl_mm + sources_and_uses.goodwill_mm)
        - (acquirer.other_intangibles_net_of_dtl_mm + marks.cdi_net_of_dtl_mm)
        - acquirer.dta_nol_deduction_mm
        - (
            acquirer.aoci_afs_unrealized_mm
            + acquirer.aoci_cash_flow_hedge_mm
            + acquirer.aoci_pension_mm
            + acquirer.aoci_htm_mm
        )
        - acquirer.other_cet1_deductions_mm
    )
    assert bridge.pro_forma_cet1_mm == pytest.approx(from_balance_sheet_mm, abs=1e-6)


def test_all_stock_zero_mark_zero_premium_deal_leaves_pro_forma_cet1_near_acquirer_plus_tbv():
    # price_to_tbv=1.0, all-stock, every mark/CDI/restructuring at zero, and target's existing
    # allowance left at zero (so eliminating it has no effect either -- see
    # test_ma_purchase_accounting's equivalent case): goodwill should be ~0, so pro forma CET1
    # should be approximately acquirer_cet1 plus the new stock issued (= target's own TBV).
    acquirer = _balanced_opening("Acquirer Bank A", "111", "222", net_loans_mm=2000.0)
    target = _balanced_opening(
        "Target Bank B",
        "333",
        "444",
        net_loans_mm=800.0,
        equity_mm=400.0,
        deposits_mm=1200.0,
        borrowings_mm=50.0,
        other_liabilities_mm=20.0,
        securities_mm=200.0,
        other_assets_mm=40.0,
    )
    config = _deal_config(
        consideration=ConsiderationConfig(price_to_tbv=1.0, stock_pct=1.0),
        credit_mark=CreditMarkConfig(credit_mark_pct=0.0, pcd_share=0.0),
        cdi=CdiConfig(cdi_pct_of_core_deposits=0.0),
    )
    acquirer_cet1_mm = _cet1_of(acquirer)
    target_cet1_mm = _cet1_of(target)
    marks = compute_fair_value_marks(target, 800.0, 0.0, config)
    sources_and_uses = compute_sources_and_uses(target, marks, config)
    bridge = compute_pro_forma_cet1_bridge(
        acquirer_cet1_mm, target_cet1_mm, target, marks, sources_and_uses, config
    )

    assert sources_and_uses.goodwill_mm == pytest.approx(0.0, abs=1e-9)
    assert bridge.pro_forma_cet1_mm == pytest.approx(
        acquirer_cet1_mm + sources_and_uses.stock_consideration_mm
    )


def test_higher_price_lowers_pro_forma_cet1():
    acquirer = _balanced_opening("Acquirer Bank A", "111", "222", net_loans_mm=2000.0)
    target = _balanced_opening(
        "Target Bank B",
        "333",
        "444",
        net_loans_mm=784.0,
        equity_mm=400.0,
        deposits_mm=1200.0,
        borrowings_mm=50.0,
        other_liabilities_mm=20.0,
        securities_mm=200.0,
        other_assets_mm=40.0,
    )
    acquirer_cet1_mm = _cet1_of(acquirer)
    target_cet1_mm = _cet1_of(target)

    # Must include a CASH component -- an all-stock deal is capital-NEUTRAL to price (the extra
    # goodwill from a higher price is exactly offset by the extra stock issued to fund it; see
    # test_all_stock_price_is_capital_neutral_to_cet1 below), so a mixed/cash deal is needed for
    # a higher price to actually show up as lower pro forma CET1.
    low_config = _deal_config(consideration=ConsiderationConfig(price_to_tbv=1.2, stock_pct=0.5))
    high_config = _deal_config(consideration=ConsiderationConfig(price_to_tbv=1.8, stock_pct=0.5))

    marks_low = compute_fair_value_marks(target, 800.0, 16.0, low_config)
    su_low = compute_sources_and_uses(target, marks_low, low_config)
    bridge_low = compute_pro_forma_cet1_bridge(
        acquirer_cet1_mm, target_cet1_mm, target, marks_low, su_low, low_config
    )

    marks_high = compute_fair_value_marks(target, 800.0, 16.0, high_config)
    su_high = compute_sources_and_uses(target, marks_high, high_config)
    bridge_high = compute_pro_forma_cet1_bridge(
        acquirer_cet1_mm, target_cet1_mm, target, marks_high, su_high, high_config
    )

    assert bridge_high.pro_forma_cet1_mm < bridge_low.pro_forma_cet1_mm


def test_all_stock_price_is_capital_neutral_to_cet1():
    # A well-known real-world bank M&A result, confirmed here: in a 100%-stock deal, pro forma
    # CET1 is INVARIANT to the price paid -- the extra goodwill from paying more is exactly
    # offset by the extra stock issued to fund it (both scale with price_to_tbv identically).
    # Price still dilutes PER-SHARE metrics (more shares issued) -- that's Stage 4's concern,
    # not this capital-ratio-level result.
    acquirer = _balanced_opening("Acquirer Bank A", "111", "222", net_loans_mm=2000.0)
    target = _balanced_opening(
        "Target Bank B",
        "333",
        "444",
        net_loans_mm=784.0,
        equity_mm=400.0,
        deposits_mm=1200.0,
        borrowings_mm=50.0,
        other_liabilities_mm=20.0,
        securities_mm=200.0,
        other_assets_mm=40.0,
    )
    acquirer_cet1_mm = _cet1_of(acquirer)
    target_cet1_mm = _cet1_of(target)

    low_config = _deal_config(consideration=ConsiderationConfig(price_to_tbv=1.2, stock_pct=1.0))
    high_config = _deal_config(consideration=ConsiderationConfig(price_to_tbv=1.8, stock_pct=1.0))

    marks_low = compute_fair_value_marks(target, 800.0, 16.0, low_config)
    su_low = compute_sources_and_uses(target, marks_low, low_config)
    bridge_low = compute_pro_forma_cet1_bridge(
        acquirer_cet1_mm, target_cet1_mm, target, marks_low, su_low, low_config
    )

    marks_high = compute_fair_value_marks(target, 800.0, 16.0, high_config)
    su_high = compute_sources_and_uses(target, marks_high, high_config)
    bridge_high = compute_pro_forma_cet1_bridge(
        acquirer_cet1_mm, target_cet1_mm, target, marks_high, su_high, high_config
    )

    assert bridge_high.pro_forma_cet1_mm == pytest.approx(bridge_low.pro_forma_cet1_mm)


def test_pro_forma_balance_sheet_balances_when_target_has_preferred_stock():
    # Consideration is priced off target's TANGIBLE COMMON equity (excludes preferred), so
    # target's preferred stock claim must be settled separately (assumed redeemed for cash --
    # see pro_forma.py's module docstring) rather than silently dropped when target's GAAP
    # equity is eliminated at close. Both example banks' real target has zero preferred stock,
    # so this edge case isn't exercised by the other tests above.
    acquirer = _balanced_opening("Acquirer Bank A", "111", "222", net_loans_mm=2000.0)
    target = _balanced_opening(
        "Target Bank B",
        "333",
        "444",
        net_loans_mm=784.0,
        equity_mm=400.0,
        preferred_stock_mm=60.0,
        deposits_mm=1200.0,
        borrowings_mm=50.0,
        other_liabilities_mm=20.0,
        securities_mm=200.0,
        other_assets_mm=40.0,
    )
    config = _deal_config(
        rate_mark=RateMarkConfig(rate_mark_pct=-0.01),
        securities_mark=SecuritiesMarkConfig(securities_mark_pct=-0.005),
        cost_saves=CostSaveConfig(restructuring_charge_mm=20.0),
    )
    marks = compute_fair_value_marks(target, 800.0, 16.0, config)
    sources_and_uses = compute_sources_and_uses(target, marks, config)
    bs = compute_pro_forma_balance_sheet(
        acquirer, target, 2000.0, acquirer.cash_mm, target.cash_mm, marks, sources_and_uses, config
    )

    assert bs.total_assets_mm == pytest.approx(bs.total_liabilities_and_equity_mm, abs=1e-6)
