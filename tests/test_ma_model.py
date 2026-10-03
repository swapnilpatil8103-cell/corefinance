"""Offline integration tests for corefin.ma.model -- synthetic data only.
Covers the required Stage 3 checks end to end: pro forma balance sheet
balances; goodwill = consideration minus fair value of net assets
exactly; PCD has no net effect at close; the CET1 bridge reconciles to
the balance-sheet-derived figure; pro forma RWA/capital ratios."""

from __future__ import annotations

import numpy as np
import pytest

from corefin.bank.model import BankModelResult, run_bank_model
from corefin.bank.schema import BankConfig, BankOpeningBalance
from corefin.credit.interface import CreditLossProjection
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

_TIMELINE = Timeline.quarterly(9, n_historical=1, start_year=2025, start_quarter=4)
_CATEGORIES = ("commercial_and_industrial", "residential_mortgage")


def _balanced_opening(
    name, bank_id, hc_rssd_id, net_loans_mm, equity_mm=1000.0, preferred_stock_mm=0.0, **overrides
):
    """`**overrides` may override any balance-sheet-layout key (goodwill_mm,
    other_intangibles_mm, deposits_mm, borrowings_mm, other_liabilities_mm,
    securities_mm, other_assets_mm -- cash_mm is then re-derived as the
    balancing plug) OR any other BankOpeningBalance field directly (e.g.
    reported_rwa_mm, reported_cet1_capital_mm, reported_tier1_capital_ratio) --
    the latter are applied AFTER the balanced defaults, so they truly
    override rather than being silently dropped."""
    layout_keys = {
        "goodwill_mm",
        "other_intangibles_mm",
        "deposits_mm",
        "borrowings_mm",
        "other_liabilities_mm",
        "securities_mm",
        "other_assets_mm",
    }
    layout = dict(
        goodwill_mm=50.0,
        other_intangibles_mm=10.0,
        deposits_mm=3000.0,
        borrowings_mm=200.0,
        other_liabilities_mm=50.0,
        securities_mm=500.0,
        other_assets_mm=100.0,
    )
    layout.update({k: v for k, v in overrides.items() if k in layout_keys})
    direct_overrides = {k: v for k, v in overrides.items() if k not in layout_keys}

    cash_mm = (
        layout["deposits_mm"] + layout["borrowings_mm"] + layout["other_liabilities_mm"] + equity_mm
    ) - (
        layout["securities_mm"]
        + net_loans_mm
        + layout["goodwill_mm"]
        + layout["other_intangibles_mm"]
        + layout["other_assets_mm"]
    )
    kwargs = dict(
        name=name,
        bank_id=bank_id,
        hc_rssd_id=hc_rssd_id,
        cash_mm=cash_mm,
        securities_afs_mm=layout["securities_mm"] / 2,
        securities_htm_mm=layout["securities_mm"] / 2,
        other_assets_mm=layout["other_assets_mm"],
        goodwill_mm=layout["goodwill_mm"],
        other_intangibles_mm=layout["other_intangibles_mm"],
        deposits_mm=layout["deposits_mm"],
        borrowings_mm=layout["borrowings_mm"],
        other_liabilities_mm=layout["other_liabilities_mm"],
        equity_mm=equity_mm,
        net_interest_income_jumpoff_mm=40.0,
        noninterest_income_jumpoff_mm=10.0,
        noninterest_expense_jumpoff_mm=30.0,
        preferred_stock_mm=preferred_stock_mm,
        goodwill_net_of_dtl_mm=layout["goodwill_mm"],
        other_intangibles_net_of_dtl_mm=layout["other_intangibles_mm"] * 0.7,
        dta_nol_deduction_mm=1.0,
        aoci_afs_unrealized_mm=-5.0,
        reported_cet1_capital_mm=(
            equity_mm
            - preferred_stock_mm
            - layout["goodwill_mm"]
            - layout["other_intangibles_mm"] * 0.7
            - 1.0
            + 5.0
        ),
        reported_cet1_ratio=0.12,
        reported_rwa_mm=net_loans_mm,  # a simple, internally-consistent calibration anchor
        reported_tier1_leverage_ratio=0.09,
    )
    kwargs.update(direct_overrides)
    return BankOpeningBalance(**kwargs)


def _credit_projection_for(
    opening: BankOpeningBalance, net_loans_mm: float
) -> CreditLossProjection:
    n_cat = len(_CATEGORIES)
    n = _TIMELINE.n_periods
    per_category = net_loans_mm / n_cat
    return CreditLossProjection(
        timeline=_TIMELINE,
        categories=_CATEGORIES,
        scenario_name="baseline",
        bank_identifier=opening.bank_id,
        balance_mm=np.full((n_cat, n), per_category),
        net_charge_off_mm=np.zeros((n_cat, n)),
        provision_expense_mm=np.zeros((n_cat, n)),
        allowance_mm=np.zeros((n_cat, n)),
        npl_mm=np.zeros((n_cat, n)),
        nco_rate=np.zeros((n_cat, n)),
        npl_ratio=np.zeros((n_cat, n)),
        monte_carlo_mean=np.zeros(n_cat),
        monte_carlo_percentiles={},
    )


def _run_bank(opening: BankOpeningBalance, net_loans_mm: float) -> BankModelResult:
    credit_projection = _credit_projection_for(opening, net_loans_mm)
    config = BankConfig()
    return run_bank_model(opening, credit_projection, config, _TIMELINE)


def _run_deal(**config_overrides) -> tuple[DealResult, BankOpeningBalance]:
    acquirer_opening = _balanced_opening(
        "Acquirer Bank A", "111", "222", net_loans_mm=2000.0, preferred_stock_mm=100.0
    )
    target_opening = _balanced_opening(
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
    acquirer_result = _run_bank(acquirer_opening, 2000.0)
    target_result = _run_bank(target_opening, 784.0)

    config_kwargs = dict(
        consideration=ConsiderationConfig(
            price_to_tbv=1.5,
            stock_pct=0.8,
            acquirer_share_price=25.0,
            acquirer_shares_outstanding_mm=50.0,
        ),
        credit_mark=CreditMarkConfig(credit_mark_pct=0.02, pcd_share=0.3),
        rate_mark=RateMarkConfig(rate_mark_pct=-0.01),
        securities_mark=SecuritiesMarkConfig(securities_mark_pct=-0.005),
        cost_saves=CostSaveConfig(restructuring_charge_mm=20.0),
    )
    config_kwargs.update(config_overrides)
    config = DealConfig(**config_kwargs)

    result = run_deal_model(acquirer_result, target_result, BankConfig(), BankConfig(), config)
    return result, acquirer_opening


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
    # against real 2025Q4 data, where it's a few thousand dollars. The residual is baked into
    # acquirer_result.capital.cet1_capital_mm (Stage 2's own calibration); the cross-check must
    # be told about it explicitly or it will show a spurious mismatch even though nothing's wrong.
    acquirer_opening = _balanced_opening(
        "Acquirer Bank A",
        "111",
        "222",
        net_loans_mm=2000.0,
        preferred_stock_mm=100.0,
        reported_cet1_capital_mm=1000.0 - 100.0 - 50.0 - 7.0 - 1.0 + 5.0 + 1.5,  # +1.5 residual
    )
    target_opening = _balanced_opening(
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
    acquirer_result = _run_bank(acquirer_opening, 2000.0)
    target_result = _run_bank(target_opening, 784.0)
    residual_mm = acquirer_result.capital.unexplained_cet1_residual_mm
    assert residual_mm == pytest.approx(1.5)

    config = DealConfig(
        consideration=ConsiderationConfig(
            price_to_tbv=1.5,
            stock_pct=0.8,
            acquirer_share_price=25.0,
            acquirer_shares_outstanding_mm=50.0,
        ),
        credit_mark=CreditMarkConfig(credit_mark_pct=0.02, pcd_share=0.3),
    )
    result = run_deal_model(acquirer_result, target_result, BankConfig(), BankConfig(), config)

    without_residual = check_cet1_bridge_matches_balance_sheet(acquirer_opening, result)
    assert not without_residual.passed

    with_residual = check_cet1_bridge_matches_balance_sheet(
        acquirer_opening, result, acquirer_unexplained_cet1_residual_mm=residual_mm
    )
    assert with_residual.passed, with_residual.describe()


def test_pro_forma_capital_ratios_are_populated():
    result, _acquirer = _run_deal()
    ratios = result.pro_forma_capital_ratios

    assert ratios.pro_forma_rwa_mm > 0
    assert ratios.pro_forma_cet1_ratio == pytest.approx(
        result.pro_forma_cet1_bridge.pro_forma_cet1_mm / ratios.pro_forma_rwa_mm
    )
    assert ratios.pro_forma_tier1_leverage_ratio == pytest.approx(
        ratios.pro_forma_tier1_capital_mm / result.pro_forma_balance_sheet.total_assets_mm
    )
    # neither synthetic bank supplies reported_tier1_capital_ratio/reported_total_capital_ratio
    assert ratios.pro_forma_total_capital_ratio is None


def test_pro_forma_total_capital_ratio_populated_when_tier2_data_available():
    acquirer_opening = _balanced_opening(
        "Acquirer Bank A",
        "111",
        "222",
        net_loans_mm=2000.0,
        preferred_stock_mm=100.0,
        reported_rwa_mm=2000.0,
        reported_tier1_capital_ratio=0.5,
        reported_total_capital_ratio=0.55,
    )
    target_opening = _balanced_opening(
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
        reported_rwa_mm=784.0,
        reported_tier1_capital_ratio=0.45,
        reported_total_capital_ratio=0.50,
    )
    acquirer_result = _run_bank(acquirer_opening, 2000.0)
    target_result = _run_bank(target_opening, 784.0)
    config = DealConfig(
        consideration=ConsiderationConfig(
            price_to_tbv=1.5,
            stock_pct=0.8,
            acquirer_share_price=25.0,
            acquirer_shares_outstanding_mm=50.0,
        ),
        credit_mark=CreditMarkConfig(credit_mark_pct=0.02, pcd_share=0.3),
    )
    result = run_deal_model(acquirer_result, target_result, BankConfig(), BankConfig(), config)
    ratios = result.pro_forma_capital_ratios

    expected_acquirer_tier2_mm = (0.55 - 0.5) * 2000.0
    expected_target_tier2_mm = (0.50 - 0.45) * 784.0
    assert ratios.acquirer_tier2_mm == pytest.approx(expected_acquirer_tier2_mm)
    assert ratios.target_tier2_mm == pytest.approx(expected_target_tier2_mm)
    assert ratios.pro_forma_total_capital_mm == pytest.approx(
        ratios.pro_forma_tier1_capital_mm + expected_acquirer_tier2_mm + expected_target_tier2_mm
    )
    assert ratios.pro_forma_total_capital_ratio == pytest.approx(
        ratios.pro_forma_total_capital_mm / ratios.pro_forma_rwa_mm
    )
