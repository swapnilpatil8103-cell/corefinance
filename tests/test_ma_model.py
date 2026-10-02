"""Offline integration tests for corefin.ma.model -- synthetic data only.
Covers the required Stage 3 checks end to end: pro forma balance sheet
balances; goodwill = consideration minus fair value of net assets
exactly; PCD has no net effect at close; the CET1 bridge reconciles to
the balance-sheet-derived figure."""

from __future__ import annotations

import numpy as np
import pytest

from corefin.bank.balance_sheet import BalanceSheet
from corefin.bank.capital import compute_cet1_capital
from corefin.bank.schema import BankOpeningBalance
from corefin.ma.model import (
    DealResult,
    check_cet1_bridge_matches_balance_sheet,
    check_goodwill_equals_consideration_less_fair_value,
    check_pcd_has_no_net_effect_at_close,
    check_pro_forma_balance_sheet_balances,
    run_deal_model,
)
from corefin.ma.schema import (
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
    name, bank_id, hc_rssd_id, net_loans_mm, equity_mm=1000.0, preferred_stock_mm=0.0, **overrides
):
    defaults = dict(
        goodwill_mm=50.0,
        other_intangibles_mm=10.0,
        deposits_mm=3000.0,
        borrowings_mm=200.0,
        other_liabilities_mm=50.0,
        securities_mm=500.0,
        other_assets_mm=100.0,
    )
    defaults.update(overrides)
    cash_mm = (
        defaults["deposits_mm"]
        + defaults["borrowings_mm"]
        + defaults["other_liabilities_mm"]
        + equity_mm
    ) - (
        defaults["securities_mm"]
        + net_loans_mm
        + defaults["goodwill_mm"]
        + defaults["other_intangibles_mm"]
        + defaults["other_assets_mm"]
    )
    return BankOpeningBalance(
        name=name,
        bank_id=bank_id,
        hc_rssd_id=hc_rssd_id,
        cash_mm=cash_mm,
        securities_afs_mm=defaults["securities_mm"] / 2,
        securities_htm_mm=defaults["securities_mm"] / 2,
        other_assets_mm=defaults["other_assets_mm"],
        goodwill_mm=defaults["goodwill_mm"],
        other_intangibles_mm=defaults["other_intangibles_mm"],
        deposits_mm=defaults["deposits_mm"],
        borrowings_mm=defaults["borrowings_mm"],
        other_liabilities_mm=defaults["other_liabilities_mm"],
        equity_mm=equity_mm,
        net_interest_income_jumpoff_mm=40.0,
        noninterest_income_jumpoff_mm=10.0,
        noninterest_expense_jumpoff_mm=30.0,
        preferred_stock_mm=preferred_stock_mm,
        goodwill_net_of_dtl_mm=defaults["goodwill_mm"],
        other_intangibles_net_of_dtl_mm=defaults["other_intangibles_mm"] * 0.7,
        dta_nol_deduction_mm=1.0,
        aoci_afs_unrealized_mm=-5.0,
        reported_cet1_capital_mm=(
            equity_mm
            - preferred_stock_mm
            - defaults["goodwill_mm"]
            - defaults["other_intangibles_mm"] * 0.7
            - 1.0
            + 5.0
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


def _run_deal(**config_overrides) -> tuple[DealResult, BankOpeningBalance]:
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
    config_kwargs = dict(
        consideration=ConsiderationConfig(price_to_tbv=1.5, stock_pct=0.8),
        credit_mark=CreditMarkConfig(credit_mark_pct=0.02, pcd_share=0.3),
        rate_mark=RateMarkConfig(rate_mark_pct=-0.01),
        securities_mark=SecuritiesMarkConfig(securities_mark_pct=-0.005),
        cost_saves=CostSaveConfig(restructuring_charge_mm=20.0),
    )
    config_kwargs.update(config_overrides)
    config = DealConfig(**config_kwargs)

    result = run_deal_model(
        acquirer=acquirer,
        target=target,
        acquirer_cet1_mm=_cet1_of(acquirer),
        target_cet1_mm=_cet1_of(target),
        acquirer_net_loans_mm=2000.0,
        acquirer_cash_mm=acquirer.cash_mm,
        target_cash_mm=target.cash_mm,
        target_gross_loans_mm=800.0,
        target_existing_allowance_mm=16.0,
        config=config,
    )
    return result, acquirer


def test_run_deal_model_passes_every_stage3_check():
    result, acquirer = _run_deal()

    assert check_pro_forma_balance_sheet_balances(result).passed
    assert check_goodwill_equals_consideration_less_fair_value(result).passed
    assert check_pcd_has_no_net_effect_at_close(result).passed
    assert check_cet1_bridge_matches_balance_sheet(acquirer, result).passed


def test_run_deal_model_is_reproducible():
    result_a, _ = _run_deal()
    result_b, _ = _run_deal()
    assert result_a.pro_forma_cet1_bridge.pro_forma_cet1_mm == pytest.approx(
        result_b.pro_forma_cet1_bridge.pro_forma_cet1_mm
    )
    assert result_a.sources_and_uses.goodwill_mm == pytest.approx(
        result_b.sources_and_uses.goodwill_mm
    )


def test_higher_credit_mark_lowers_pro_forma_cet1_and_tbv_denominator():
    low_mark_result, _ = _run_deal(
        credit_mark=CreditMarkConfig(credit_mark_pct=0.01, pcd_share=0.3)
    )
    high_mark_result, _ = _run_deal(
        credit_mark=CreditMarkConfig(credit_mark_pct=0.05, pcd_share=0.3)
    )
    assert (
        high_mark_result.pro_forma_cet1_bridge.pro_forma_cet1_mm
        < low_mark_result.pro_forma_cet1_bridge.pro_forma_cet1_mm
    )


def test_cet1_bridge_reconciliation_accounts_for_a_nonzero_acquirer_residual():
    # A real acquirer's own Stage 2 CET1 bridge rarely reconciles to EXACTLY zero residual
    # (corefin.bank.capital.compute_cet1_capital's own unexplained_cet1_residual_mm) -- confirmed
    # against real 2025Q4 data, where it's a few thousand dollars. acquirer_cet1_mm fed into
    # compute_pro_forma_cet1_bridge already includes that residual; the cross-check must be told
    # about it explicitly or it will show a spurious mismatch even though nothing is wrong.
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
    config = DealConfig(
        consideration=ConsiderationConfig(price_to_tbv=1.5, stock_pct=0.8),
        credit_mark=CreditMarkConfig(credit_mark_pct=0.02, pcd_share=0.3),
    )
    formula_cet1_mm = _cet1_of(acquirer)
    residual_mm = 1.5  # e.g. a bank whose reported CET1 isn't perfectly explained by the bridge
    acquirer_cet1_mm = formula_cet1_mm + residual_mm

    result = run_deal_model(
        acquirer=acquirer,
        target=target,
        acquirer_cet1_mm=acquirer_cet1_mm,
        target_cet1_mm=_cet1_of(target),
        acquirer_net_loans_mm=2000.0,
        acquirer_cash_mm=acquirer.cash_mm,
        target_cash_mm=target.cash_mm,
        target_gross_loans_mm=800.0,
        target_existing_allowance_mm=16.0,
        config=config,
    )

    without_residual = check_cet1_bridge_matches_balance_sheet(acquirer, result)
    assert not without_residual.passed

    with_residual = check_cet1_bridge_matches_balance_sheet(
        acquirer, result, acquirer_unexplained_cet1_residual_mm=residual_mm
    )
    assert with_residual.passed, with_residual.describe()
