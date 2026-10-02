"""Offline tests for corefin.bank.balance_sheet -- synthetic data only."""

from __future__ import annotations

import numpy as np
import pytest

from corefin.bank.balance_sheet import project_balance_sheet, roll_forward_from_jumpoff
from corefin.bank.schema import BankConfig, BankOpeningBalance
from corefin.credit.interface import CreditLossProjection
from corefin.timeline import Timeline

_CATEGORIES = ("commercial_and_industrial", "residential_mortgage")


def _synthetic_credit_projection(
    timeline: Timeline, bank_id: str = "99999"
) -> CreditLossProjection:
    n_cat = len(_CATEGORIES)
    n = timeline.n_periods
    balance_mm = np.full((n_cat, n), 1000.0)
    allowance_mm = np.full((n_cat, n), 20.0)
    return CreditLossProjection(
        timeline=timeline,
        categories=_CATEGORIES,
        scenario_name="baseline",
        bank_identifier=bank_id,
        balance_mm=balance_mm,
        net_charge_off_mm=np.full((n_cat, n), 2.0),
        provision_expense_mm=np.full((n_cat, n), 2.0),
        allowance_mm=allowance_mm,
        npl_mm=np.full((n_cat, n), 10.0),
        nco_rate=np.full((n_cat, n), 0.002),
        npl_ratio=np.full((n_cat, n), 0.01),
        monte_carlo_mean=np.zeros(n_cat),
        monte_carlo_percentiles={},
    )


def _synthetic_opening(**overrides) -> BankOpeningBalance:
    kwargs = dict(
        name="Test Bank",
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


def _timeline(n_periods: int = 9) -> Timeline:
    return Timeline.quarterly(n_periods=n_periods, n_historical=1, start_year=2025, start_quarter=4)


def test_roll_forward_from_jumpoff_period_zero_equals_opening_exactly():
    is_projection = np.array([False, True, True, True])
    deltas = np.array([999.0, 10.0, 10.0, 10.0])  # period 0's own delta must be ignored
    level = roll_forward_from_jumpoff(100.0, deltas, is_projection)
    assert level[0] == pytest.approx(100.0)
    assert level[1] == pytest.approx(110.0)
    assert level[3] == pytest.approx(130.0)


def test_project_balance_sheet_is_static_at_zero_growth_except_equity():
    timeline = _timeline()
    opening = _synthetic_opening()
    credit_projection = _synthetic_credit_projection(timeline)
    config = BankConfig(balance_sheet_growth_rate=0.0)
    net_income_mm = np.full(timeline.n_periods, 5.0)
    dividends_mm = np.zeros(timeline.n_periods)

    bs = project_balance_sheet(
        opening, credit_projection, net_income_mm, dividends_mm, config, timeline
    )

    assert np.allclose(bs.securities_afs_mm, opening.securities_afs_mm)
    assert np.allclose(bs.deposits_mm, opening.deposits_mm)
    assert np.allclose(bs.goodwill_mm, opening.goodwill_mm)
    assert np.allclose(bs.total_loans_mm, 2000.0)  # 2 categories * 1000, static
    # equity grows with retained earnings (net income - 0 dividends)
    assert bs.equity_mm[0] == pytest.approx(opening.equity_mm)
    assert bs.equity_mm[-1] == pytest.approx(opening.equity_mm + 5.0 * (timeline.n_periods - 1))


def test_project_balance_sheet_balances_every_period():
    timeline = _timeline()
    opening = _synthetic_opening()
    credit_projection = _synthetic_credit_projection(timeline)
    config = BankConfig(balance_sheet_growth_rate=0.01, dividend_payout_ratio=0.3)
    net_income_mm = np.linspace(5.0, 8.0, timeline.n_periods)
    dividends_mm = net_income_mm * config.dividend_payout_ratio

    bs = project_balance_sheet(
        opening, credit_projection, net_income_mm, dividends_mm, config, timeline
    )

    residual = bs.total_assets_mm - bs.total_liabilities_and_equity_mm
    assert np.allclose(residual, 0.0, atol=1e-8)


def test_project_balance_sheet_jumpoff_matches_opening_totals():
    timeline = _timeline()
    opening = _synthetic_opening()
    credit_projection = _synthetic_credit_projection(timeline)
    config = BankConfig()
    net_income_mm = np.full(timeline.n_periods, 5.0)
    dividends_mm = np.zeros(timeline.n_periods)

    bs = project_balance_sheet(
        opening, credit_projection, net_income_mm, dividends_mm, config, timeline
    )

    expected_jumpoff_liab_and_equity = (
        opening.deposits_mm
        + opening.borrowings_mm
        + opening.other_liabilities_mm
        + opening.equity_mm
    )
    assert bs.total_liabilities_and_equity_mm[0] == pytest.approx(expected_jumpoff_liab_and_equity)
    assert bs.total_assets_mm[0] == pytest.approx(expected_jumpoff_liab_and_equity)


def test_balance_sheet_rejects_wrong_shape_array():
    from corefin.bank.balance_sheet import BalanceSheet

    timeline = _timeline(n_periods=4)
    n = timeline.n_periods
    with pytest.raises(ValueError, match="cash_mm"):
        BalanceSheet(
            timeline=timeline,
            categories=_CATEGORIES,
            loan_balance_mm=np.zeros((2, n)),
            allowance_mm=np.zeros((2, n)),
            cash_mm=np.zeros(n + 1),  # wrong shape
            securities_afs_mm=np.zeros(n),
            securities_htm_mm=np.zeros(n),
            other_assets_mm=np.zeros(n),
            goodwill_mm=np.zeros(n),
            other_intangibles_mm=np.zeros(n),
            deposits_mm=np.zeros(n),
            borrowings_mm=np.zeros(n),
            other_liabilities_mm=np.zeros(n),
            equity_mm=np.zeros(n),
        )
