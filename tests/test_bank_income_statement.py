"""Offline tests for corefin.bank.income_statement -- synthetic data only."""

from __future__ import annotations

import numpy as np
import pytest

from corefin.bank.income_statement import compute_income_statement
from corefin.bank.schema import BankConfig, PpnrStressConfig
from corefin.timeline import Timeline


def _timeline(n_periods: int = 9) -> Timeline:
    return Timeline.quarterly(n_periods=n_periods, n_historical=1, start_year=2025, start_quarter=4)


def test_compute_income_statement_pretax_bridges_correctly():
    timeline = _timeline()
    n = timeline.n_periods
    provision_mm = np.full(n, 2.0)
    config = BankConfig(tax_rate=0.25, balance_sheet_growth_rate=0.0)

    result = compute_income_statement(
        net_interest_income_jumpoff_mm=40.0,
        noninterest_income_jumpoff_mm=10.0,
        noninterest_expense_jumpoff_mm=30.0,
        provision_expense_total_mm=provision_mm,
        config=config,
        timeline=timeline,
    )
    expected_pretax = 40.0 + 10.0 - 30.0 - 2.0
    assert np.allclose(result.pretax_income_mm, expected_pretax)
    assert np.allclose(result.tax_expense_mm, expected_pretax * 0.25)
    assert np.allclose(result.net_income_mm, expected_pretax * 0.75)


def test_compute_income_statement_no_tax_benefit_on_pretax_loss():
    timeline = _timeline()
    n = timeline.n_periods
    provision_mm = np.full(n, 1000.0)  # drives a pretax loss
    config = BankConfig(tax_rate=0.25)

    result = compute_income_statement(
        net_interest_income_jumpoff_mm=40.0,
        noninterest_income_jumpoff_mm=10.0,
        noninterest_expense_jumpoff_mm=30.0,
        provision_expense_total_mm=provision_mm,
        config=config,
        timeline=timeline,
    )
    assert np.all(result.pretax_income_mm < 0)
    assert np.allclose(result.tax_expense_mm, 0.0)
    assert np.allclose(result.net_income_mm, result.pretax_income_mm)


def test_compute_income_statement_dividends_scale_with_payout_ratio():
    timeline = _timeline()
    n = timeline.n_periods
    provision_mm = np.zeros(n)
    config = BankConfig(tax_rate=0.25, dividend_payout_ratio=0.4)

    result = compute_income_statement(
        net_interest_income_jumpoff_mm=40.0,
        noninterest_income_jumpoff_mm=10.0,
        noninterest_expense_jumpoff_mm=30.0,
        provision_expense_total_mm=provision_mm,
        config=config,
        timeline=timeline,
    )
    assert np.allclose(result.dividends_mm, result.net_income_mm * 0.4)


def test_compute_income_statement_growth_applied_from_period_zero():
    timeline = _timeline(n_periods=3)
    provision_mm = np.zeros(3)
    config = BankConfig(balance_sheet_growth_rate=0.10)

    result = compute_income_statement(
        net_interest_income_jumpoff_mm=100.0,
        noninterest_income_jumpoff_mm=0.0,
        noninterest_expense_jumpoff_mm=0.0,
        provision_expense_total_mm=provision_mm,
        config=config,
        timeline=timeline,
    )
    assert result.net_interest_income_mm[0] == pytest.approx(100.0)
    assert result.net_interest_income_mm[1] == pytest.approx(110.0)
    assert result.net_interest_income_mm[2] == pytest.approx(121.0)


def test_preferred_dividends_reduce_net_income_available_to_common_not_net_income():
    timeline = _timeline()
    n = timeline.n_periods
    provision_mm = np.zeros(n)
    config = BankConfig(tax_rate=0.25)

    result = compute_income_statement(
        net_interest_income_jumpoff_mm=40.0,
        noninterest_income_jumpoff_mm=10.0,
        noninterest_expense_jumpoff_mm=30.0,
        provision_expense_total_mm=provision_mm,
        config=config,
        timeline=timeline,
        preferred_dividends_jumpoff_mm=3.0,
    )
    assert np.allclose(result.preferred_dividends_mm, 3.0)
    assert np.allclose(result.net_income_available_to_common_mm, result.net_income_mm - 3.0)
    # net_income_mm itself is unaffected by preferred dividends -- they're a distribution,
    # not an expense.
    assert np.allclose(result.net_income_mm, result.pretax_income_mm * 0.75)


def test_preferred_dividends_stay_static_even_with_balance_sheet_growth():
    timeline = _timeline(n_periods=3)
    provision_mm = np.zeros(3)
    config = BankConfig(balance_sheet_growth_rate=0.10)

    result = compute_income_statement(
        net_interest_income_jumpoff_mm=100.0,
        noninterest_income_jumpoff_mm=0.0,
        noninterest_expense_jumpoff_mm=0.0,
        provision_expense_total_mm=provision_mm,
        config=config,
        timeline=timeline,
        preferred_dividends_jumpoff_mm=5.0,
    )
    # unlike net_interest_income_mm (which grows 10%/period), preferred dividends stay flat
    assert np.allclose(result.preferred_dividends_mm, 5.0)


def test_total_dividends_mm_sums_common_and_preferred():
    timeline = _timeline()
    n = timeline.n_periods
    provision_mm = np.zeros(n)
    config = BankConfig(dividend_payout_ratio=0.5)

    result = compute_income_statement(
        net_interest_income_jumpoff_mm=40.0,
        noninterest_income_jumpoff_mm=10.0,
        noninterest_expense_jumpoff_mm=30.0,
        provision_expense_total_mm=provision_mm,
        config=config,
        timeline=timeline,
        preferred_dividends_jumpoff_mm=4.0,
    )
    assert np.allclose(result.total_dividends_mm, result.dividends_mm + 4.0)
    # common dividend payout applies to earnings AVAILABLE TO COMMON, not total net income
    assert np.allclose(result.dividends_mm, result.net_income_available_to_common_mm * 0.5)


def test_ppnr_stress_moves_nii_with_the_rate_path_when_configured():
    timeline = _timeline(n_periods=3)
    provision_mm = np.zeros(3)
    config = BankConfig(
        ppnr_stress=PpnrStressConfig(nim_beta=0.3, noninterest_income_decline_pct=0.10)
    )
    rate_path_pp = np.array([3.7, 0.1, 0.1])

    result = compute_income_statement(
        net_interest_income_jumpoff_mm=100.0,
        noninterest_income_jumpoff_mm=20.0,
        noninterest_expense_jumpoff_mm=30.0,
        provision_expense_total_mm=provision_mm,
        config=config,
        timeline=timeline,
        rate_path_pp=rate_path_pp,
        earning_assets_jumpoff_mm=10_000.0,
    )
    assert result.net_interest_income_mm[0] == pytest.approx(100.0)
    expected_delta = (0.3 * (0.1 - 3.7) / 100.0) * 10_000.0 / 4.0
    assert result.net_interest_income_mm[1] == pytest.approx(100.0 + expected_delta)
    # noninterest income steps down 10% from period 1; expense is held flat, not grown
    assert result.noninterest_income_mm[0] == pytest.approx(20.0)
    assert result.noninterest_income_mm[1] == pytest.approx(18.0)
    assert np.allclose(result.noninterest_expense_mm, 30.0)


def test_ppnr_stress_raises_without_rate_path_or_earning_assets():
    timeline = _timeline(n_periods=3)
    provision_mm = np.zeros(3)
    config = BankConfig(ppnr_stress=PpnrStressConfig())

    with pytest.raises(ValueError, match="ppnr_stress"):
        compute_income_statement(
            net_interest_income_jumpoff_mm=100.0,
            noninterest_income_jumpoff_mm=20.0,
            noninterest_expense_jumpoff_mm=30.0,
            provision_expense_total_mm=provision_mm,
            config=config,
            timeline=timeline,
        )


def test_income_statement_rejects_wrong_shape_array():
    from corefin.bank.income_statement import IncomeStatement

    timeline = _timeline(n_periods=4)
    n = timeline.n_periods
    with pytest.raises(ValueError, match="net_interest_income_mm"):
        IncomeStatement(
            timeline=timeline,
            net_interest_income_mm=np.zeros(n + 1),
            noninterest_income_mm=np.zeros(n),
            noninterest_expense_mm=np.zeros(n),
            provision_expense_mm=np.zeros(n),
            pretax_income_mm=np.zeros(n),
            tax_expense_mm=np.zeros(n),
            net_income_mm=np.zeros(n),
            preferred_dividends_mm=np.zeros(n),
            net_income_available_to_common_mm=np.zeros(n),
            dividends_mm=np.zeros(n),
        )
