"""Offline tests for corefin.bank.model -- the orchestrator plus the
required Stage 2 integrity checks (balance sheet balances; capital
roll-forward ties) and the Call-Report-vs-FR-Y-9C level reconciliation.
Synthetic data only."""

from __future__ import annotations

import numpy as np
import pytest

from corefin.bank.model import (
    BankModelResult,
    check_balance_sheet_balances,
    check_capital_rollforward,
    check_cet1_bridge_reconciles,
    reconcile_levels,
    run_bank_model,
)
from corefin.bank.schema import BankConfig, BankOpeningBalance
from corefin.credit.interface import CreditLossProjection
from corefin.timeline import Timeline

_CATEGORIES = ("commercial_and_industrial", "residential_mortgage", "home_equity")


def _timeline(n_periods: int = 9) -> Timeline:
    return Timeline.quarterly(n_periods=n_periods, n_historical=1, start_year=2025, start_quarter=4)


def _synthetic_credit_projection(
    timeline: Timeline, bank_id: str = "99999", seed: int = 0
) -> CreditLossProjection:
    n_cat = len(_CATEGORIES)
    n = timeline.n_periods
    rng = np.random.default_rng(seed)
    balance_mm = np.full((n_cat, n), 1000.0) + rng.normal(0, 5, size=(n_cat, n)).cumsum(axis=1)
    allowance_mm = np.full((n_cat, n), 20.0)
    provision_mm = np.abs(rng.normal(2.0, 0.5, size=(n_cat, n)))
    return CreditLossProjection(
        timeline=timeline,
        categories=_CATEGORIES,
        scenario_name="baseline",
        bank_identifier=bank_id,
        balance_mm=balance_mm,
        net_charge_off_mm=provision_mm,
        provision_expense_mm=provision_mm,
        allowance_mm=allowance_mm,
        npl_mm=np.full((n_cat, n), 10.0),
        nco_rate=provision_mm / balance_mm,
        npl_ratio=np.full((n_cat, n), 0.01),
        monte_carlo_mean=np.zeros(n_cat),
        monte_carlo_percentiles={},
    )


def _synthetic_opening(**overrides) -> BankOpeningBalance:
    kwargs = dict(
        name="Acquirer Bank A",
        bank_id="99999",
        hc_rssd_id="88888",
        cash_mm=500.0,
        securities_afs_mm=300.0,
        securities_htm_mm=200.0,
        other_assets_mm=100.0,
        goodwill_mm=50.0,
        other_intangibles_mm=10.0,
        deposits_mm=3000.0,
        borrowings_mm=200.0,
        other_liabilities_mm=50.0,
        equity_mm=850.0,
        net_interest_income_jumpoff_mm=40.0,
        noninterest_income_jumpoff_mm=10.0,
        noninterest_expense_jumpoff_mm=30.0,
        goodwill_net_of_dtl_mm=50.0,
        reported_cet1_capital_mm=790.0,
        reported_cet1_ratio=0.12,
        reported_rwa_mm=6583.33,
        reported_tier1_leverage_ratio=0.09,
    )
    kwargs.update(overrides)
    return BankOpeningBalance(**kwargs)


def test_run_bank_model_returns_a_consistent_result():
    timeline = _timeline()
    opening = _synthetic_opening()
    credit_projection = _synthetic_credit_projection(timeline)
    config = BankConfig()

    result = run_bank_model(opening, credit_projection, config, timeline)

    assert isinstance(result, BankModelResult)
    assert result.balance_sheet.timeline is timeline
    assert result.income_statement.timeline is timeline
    assert result.capital.timeline is timeline


def test_run_bank_model_rejects_mismatched_bank_identifier():
    timeline = _timeline()
    opening = _synthetic_opening(bank_id="99999")
    credit_projection = _synthetic_credit_projection(timeline, bank_id="11111")
    config = BankConfig()

    with pytest.raises(ValueError, match="99999"):
        run_bank_model(opening, credit_projection, config, timeline)


def test_balance_sheet_balances_every_quarter_for_a_full_model_run():
    timeline = _timeline()
    opening = _synthetic_opening()
    credit_projection = _synthetic_credit_projection(timeline, seed=1)
    config = BankConfig(balance_sheet_growth_rate=0.015, dividend_payout_ratio=0.3)

    result = run_bank_model(opening, credit_projection, config, timeline)
    check = check_balance_sheet_balances(result.balance_sheet)
    assert check.passed, check.describe()


def test_capital_rollforward_ties_for_a_full_model_run():
    timeline = _timeline()
    opening = _synthetic_opening()
    credit_projection = _synthetic_credit_projection(timeline, seed=2)
    config = BankConfig(balance_sheet_growth_rate=0.02, dividend_payout_ratio=0.5)

    result = run_bank_model(opening, credit_projection, config, timeline)
    check = check_capital_rollforward(result)
    assert check.passed, check.describe()


def test_jumpoff_cet1_ratio_matches_reported_for_a_full_model_run():
    timeline = _timeline()
    opening = _synthetic_opening()
    credit_projection = _synthetic_credit_projection(timeline, seed=3)
    config = BankConfig()

    result = run_bank_model(opening, credit_projection, config, timeline)
    assert result.capital.cet1_ratio[0] == pytest.approx(opening.reported_cet1_ratio, abs=1e-6)


def test_cet1_bridge_reconciles_when_explicit_items_already_explain_reported_capital():
    # goodwill_net_of_dtl_mm=50.0 (the synthetic fixture's only nonzero bridge item) already
    # equals equity(850) - reported_cet1_capital_mm(790) = 60... not quite, so there IS a small
    # residual here, but it should still be well under the default 2% materiality threshold.
    timeline = _timeline()
    opening = _synthetic_opening()
    credit_projection = _synthetic_credit_projection(timeline, seed=4)
    config = BankConfig()

    result = run_bank_model(opening, credit_projection, config, timeline)
    check = check_cet1_bridge_reconciles(result)
    assert check.passed, check.describe()


def test_cet1_bridge_flags_a_large_unexplained_residual():
    # A reported CET1 capital wildly inconsistent with the bridge inputs (e.g. a data error, or
    # a bank with a material RC-R Part I item this bridge doesn't itemize) must fail the
    # materiality check rather than silently absorb an oversized residual.
    timeline = _timeline()
    opening = _synthetic_opening(reported_cet1_capital_mm=1.0)
    credit_projection = _synthetic_credit_projection(timeline, seed=5)
    config = BankConfig()

    result = run_bank_model(opening, credit_projection, config, timeline)
    check = check_cet1_bridge_reconciles(result)
    assert not check.passed


def test_model_is_reproducible_across_repeated_runs():
    timeline = _timeline()
    opening = _synthetic_opening()
    config = BankConfig(balance_sheet_growth_rate=0.01)

    credit_projection_a = _synthetic_credit_projection(timeline, seed=42)
    credit_projection_b = _synthetic_credit_projection(timeline, seed=42)
    result_a = run_bank_model(opening, credit_projection_a, config, timeline)
    result_b = run_bank_model(opening, credit_projection_b, config, timeline)

    assert np.allclose(result_a.capital.cet1_ratio, result_b.capital.cet1_ratio)
    assert np.allclose(
        result_a.balance_sheet.total_assets_mm, result_b.balance_sheet.total_assets_mm
    )


def test_reconcile_levels_reports_the_equity_gap():
    recon = reconcile_levels(
        bank_level_loans_mm=11395.3,
        bank_level_deposits_mm=16264.6,
        bank_level_equity_mm=3036.7,
        hc_level_assets_mm=20842.3,
        hc_level_equity_mm=3055.7,
        hc_level_deposits_mm=None,
    )
    assert recon.equity_gap_mm == pytest.approx(19.0, abs=0.1)
    assert recon.equity_gap_pct == pytest.approx(0.00626, abs=1e-4)
    assert recon.deposits_gap_mm is None


def test_reconcile_levels_computes_deposits_gap_when_both_levels_available():
    recon = reconcile_levels(
        bank_level_loans_mm=1000.0,
        bank_level_deposits_mm=5000.0,
        bank_level_equity_mm=500.0,
        hc_level_assets_mm=6000.0,
        hc_level_equity_mm=520.0,
        hc_level_deposits_mm=5100.0,
    )
    assert recon.deposits_gap_mm == pytest.approx(100.0)
