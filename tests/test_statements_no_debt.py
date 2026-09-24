import numpy as np

from corefin.assumptions.schema import RootConfig
from corefin.checks.corporate_checks import check_balance_sheet_balances, check_cash_ties_to_cfs
from corefin.checks.framework import run_checks
from corefin.scenarios.generator import (
    broadcast_drivers,
    deterministic_drivers,
    generate_stochastic_drivers,
)
from corefin.statements.corporate_model import run_corporate_model_no_debt
from corefin.timeline import Timeline
from tests.conftest import minimal_config_dict


def _hand_checked_config_dict() -> dict:
    data = minimal_config_dict(n_periods=2)
    data["company"].update(
        {
            "revenue_base_mm": 100.0,
            "revenue_growth": 0.10,
            "ebitda_margin": 0.20,
            "da_pct_revenue": 0.05,
            "capex_pct_revenue": 0.05,
            "nwc_pct_revenue": 0.10,
            "tax_rate": 0.25,
            "nol_beginning_balance_mm": 0.0,
            "dividend_pct_of_ni": 0.0,
        }
    )
    data["opening_balance_sheet"] = {
        "cash_mm": 10.0,
        "nwc_mm": 10.0,
        "ppe_mm": 50.0,
        "goodwill_mm": 0.0,
        "other_liabilities_mm": 0.0,
        "equity_mm": 70.0,
    }
    return data


def test_hand_checked_deterministic_case():
    data = _hand_checked_config_dict()
    config = RootConfig.model_validate(data)
    timeline = Timeline.annual(n_periods=2)
    drivers = deterministic_drivers(config, timeline)

    result = run_corporate_model_no_debt(
        config.company, config.opening_balance_sheet, timeline, drivers
    )

    expected_revenue = [110.0, 121.0]
    expected_net_income = [12.375, 13.6125]
    expected_cash = [21.375, 33.8875]
    expected_equity = [82.375, 95.9875]

    np.testing.assert_allclose(result.income_statement.revenue[0], expected_revenue)
    np.testing.assert_allclose(result.income_statement.net_income[0], expected_net_income)
    np.testing.assert_allclose(result.balance_sheet.cash[0], expected_cash)
    np.testing.assert_allclose(result.balance_sheet.equity[0], expected_equity)


def test_balance_sheet_balances_every_period_and_scenario():
    data = minimal_config_dict(n_periods=6)
    data["scenario"]["n_scenarios"] = 25
    data["scenario"]["driver_vol"] = {"revenue_growth_std": 0.05, "ebitda_margin_std": 0.02}
    config = RootConfig.model_validate(data)
    timeline = Timeline.annual(n_periods=6)
    drivers = generate_stochastic_drivers(config, timeline, seed=3)
    result = run_corporate_model_no_debt(
        config.company, config.opening_balance_sheet, timeline, drivers
    )
    run_checks([check_balance_sheet_balances(result.balance_sheet)])


def test_cash_ties_to_cash_flow_statement():
    config = RootConfig.model_validate(minimal_config_dict(n_periods=5))
    timeline = Timeline.annual(n_periods=5)
    drivers = broadcast_drivers(deterministic_drivers(config, timeline), n_scenarios=10)
    result = run_corporate_model_no_debt(
        config.company, config.opening_balance_sheet, timeline, drivers
    )
    check = check_cash_ties_to_cfs(
        result.balance_sheet, result.cash_flow_statement, config.opening_balance_sheet.cash_mm
    )
    assert check.passed


def test_nol_carryforward_prevents_negative_tax():
    data = minimal_config_dict(n_periods=3)
    data["company"]["revenue_growth"] = -0.5
    data["company"]["ebitda_margin"] = -0.5
    config = RootConfig.model_validate(data)
    timeline = Timeline.annual(n_periods=3)
    drivers = deterministic_drivers(config, timeline)
    result = run_corporate_model_no_debt(
        config.company, config.opening_balance_sheet, timeline, drivers
    )
    assert np.all(result.income_statement.tax >= 0)
    assert np.all(result.income_statement.nol_balance >= 0)
    assert result.income_statement.nol_balance[0, -1] > 0


def test_n_scenarios_one_matches_broadcast_average():
    config = RootConfig.model_validate(minimal_config_dict(n_periods=4))
    timeline = Timeline.annual(n_periods=4)
    single = deterministic_drivers(config, timeline)
    tiled = broadcast_drivers(single, n_scenarios=8)
    result_single = run_corporate_model_no_debt(
        config.company, config.opening_balance_sheet, timeline, single
    )
    result_tiled = run_corporate_model_no_debt(
        config.company, config.opening_balance_sheet, timeline, tiled
    )
    for s in range(8):
        np.testing.assert_allclose(
            result_tiled.balance_sheet.cash[s], result_single.balance_sheet.cash[0]
        )
