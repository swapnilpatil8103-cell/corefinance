"""Offline tests for corefin.bank.income_statement -- synthetic data only."""

from __future__ import annotations

import numpy as np
import pytest

from corefin.bank.income_statement import compute_income_statement
from corefin.bank.schema import BankConfig
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
            dividends_mm=np.zeros(n),
        )
