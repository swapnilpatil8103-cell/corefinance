import numpy as np

from corefin.assumptions.schema import RootConfig
from corefin.checks.corporate_checks import check_balance_sheet_balances, check_cash_ties_to_cfs
from corefin.checks.debt_checks import (
    check_debt_rollforward,
    check_revolver_bounds,
    check_sweep_priority_respected,
)
from corefin.checks.framework import run_checks
from corefin.scenarios.generator import (
    broadcast_drivers,
    deterministic_drivers,
    generate_stochastic_drivers,
)
from corefin.statements.corporate_model import (
    run_corporate_model_no_debt,
    run_corporate_model_with_debt,
)
from corefin.timeline import Timeline


def debt_config_dict(n_periods: int = 5) -> dict:
    return {
        "timeline": {"n_periods": n_periods, "n_historical": 0, "start_year": 2024},
        "company": {
            "revenue_base_mm": 500.0,
            "revenue_growth": 0.06,
            "ebitda_margin": 0.20,
            "da_pct_revenue": 0.03,
            "capex_pct_revenue": 0.04,
            "nwc_pct_revenue": 0.10,
            "tax_rate": 0.25,
            "nol_beginning_balance_mm": 0.0,
            "dividend_pct_of_ni": 0.0,
        },
        "opening_balance_sheet": {
            "cash_mm": 10.0,
            "nwc_mm": 50.0,
            "ppe_mm": 200.0,
            "goodwill_mm": 240.0,
            "deferred_financing_costs_mm": 0.0,
            "other_liabilities_mm": 0.0,
            "equity_mm": 150.0,
        },
        "transaction": {
            "entry_multiple": 8.0,
            "transaction_fees_pct": 0.0,
            "exit_year_index": n_periods - 1,
        },
        "tranches": [
            {
                "name": "Revolver",
                "tranche_type": "revolver",
                "size_mm": 50.0,
                "rate_type": "floating",
                "spread": 0.04,
                "rate_floor": 0.0,
                "commitment_fee_pct": 0.005,
                "cash_sweep_eligible": False,
            },
            {
                "name": "TLB",
                "tranche_type": "term_loan_b",
                "size_mm": 250.0,
                "rate_type": "floating",
                "spread": 0.05,
                "mandatory_amort_pct_of_original": 0.01,
                "cash_sweep_eligible": True,
                "sweep_priority": 1,
            },
            {
                "name": "Notes",
                "tranche_type": "senior_notes",
                "size_mm": 100.0,
                "rate_type": "fixed",
                "fixed_rate": 0.08,
                "cash_sweep_eligible": True,
                "sweep_priority": 2,
            },
        ],
        "covenants": [],
        "scenario": {
            "n_scenarios": 1,
            "random_seed": 42,
            "exit_multiple": 8.0,
            "base_rate": 0.045,
        },
        "waterfall": {
            "minimum_cash_mm": 10.0,
            "sweep_pct": 1.0,
            "interest_mode": "average_balance",
        },
    }


def test_opening_balance_sheet_is_debt_inclusive_and_balances():
    data = debt_config_dict()
    ob = data["opening_balance_sheet"]
    assets = ob["cash_mm"] + ob["nwc_mm"] + ob["ppe_mm"] + ob["goodwill_mm"]
    initial_debt = sum(t["size_mm"] for t in data["tranches"] if t["tranche_type"] != "revolver")
    assert assets == ob["other_liabilities_mm"] + ob["equity_mm"] + initial_debt


def test_balance_sheet_balances_with_debt_across_scenarios():
    data = debt_config_dict(n_periods=6)
    data["scenario"]["n_scenarios"] = 40
    data["scenario"]["driver_vol"] = {
        "revenue_growth_std": 0.03,
        "ebitda_margin_std": 0.02,
        "base_rate_std": 0.01,
    }
    config = RootConfig.model_validate(data)
    timeline = Timeline.annual(n_periods=6)
    drivers = generate_stochastic_drivers(config, timeline, seed=11)
    result = run_corporate_model_with_debt(
        config.company,
        config.opening_balance_sheet,
        config.tranches,
        config.waterfall,
        timeline,
        drivers,
    )
    run_checks([check_balance_sheet_balances(result.balance_sheet, tolerance=1e-6)])


def test_cash_ties_to_cfs_with_debt():
    config = RootConfig.model_validate(debt_config_dict(n_periods=5))
    timeline = Timeline.annual(n_periods=5)
    drivers = broadcast_drivers(deterministic_drivers(config, timeline), n_scenarios=15)
    result = run_corporate_model_with_debt(
        config.company,
        config.opening_balance_sheet,
        config.tranches,
        config.waterfall,
        timeline,
        drivers,
    )
    check = check_cash_ties_to_cfs(
        result.balance_sheet, result.cash_flow_statement, config.opening_balance_sheet.cash_mm
    )
    assert check.passed


def test_debt_rollforward_net_of_pik():
    data = debt_config_dict(n_periods=6)
    data["tranches"][2]["pik_fraction"] = 0.5
    config = RootConfig.model_validate(data)
    timeline = Timeline.annual(n_periods=6)
    drivers = deterministic_drivers(config, timeline)
    result = run_corporate_model_with_debt(
        config.company,
        config.opening_balance_sheet,
        config.tranches,
        config.waterfall,
        timeline,
        drivers,
    )
    run_checks([check_debt_rollforward(result.debt_schedule)])
    assert np.any(sum(result.debt_schedule.pik_interest.values()) > 0)


def test_revolver_bounds_respected_under_stress():
    data = debt_config_dict(n_periods=6)
    data["company"]["revenue_growth"] = -0.2
    data["company"]["ebitda_margin"] = 0.05
    config = RootConfig.model_validate(data)
    timeline = Timeline.annual(n_periods=6)
    drivers = deterministic_drivers(config, timeline)
    result = run_corporate_model_with_debt(
        config.company,
        config.opening_balance_sheet,
        config.tranches,
        config.waterfall,
        timeline,
        drivers,
    )
    revolver = next(t for t in config.tranches if t.is_revolver)
    check = check_revolver_bounds(result.debt_schedule, revolver)
    assert check.passed
    assert np.any(result.debt_schedule.shortfall_flag)


def test_sweep_priority_respected():
    config = RootConfig.model_validate(debt_config_dict(n_periods=6))
    timeline = Timeline.annual(n_periods=6)
    drivers = deterministic_drivers(config, timeline)
    result = run_corporate_model_with_debt(
        config.company,
        config.opening_balance_sheet,
        config.tranches,
        config.waterfall,
        timeline,
        drivers,
    )
    check = check_sweep_priority_respected(result.debt_schedule, config.tranches)
    assert check.passed
    assert np.any(sum(result.debt_schedule.sweep_amort.values())[0] > 0)


def test_zero_rate_debt_matches_no_debt_net_income():
    data = debt_config_dict(n_periods=5)
    for tranche in data["tranches"]:
        if tranche["rate_type"] == "floating":
            tranche["spread"] = 0.0
        else:
            tranche["fixed_rate"] = 0.0
    data["tranches"][0]["commitment_fee_pct"] = 0.0
    data["scenario"]["base_rate"] = 0.0
    config = RootConfig.model_validate(data)
    timeline = Timeline.annual(n_periods=5)
    drivers = deterministic_drivers(config, timeline)

    with_debt = run_corporate_model_with_debt(
        config.company,
        config.opening_balance_sheet,
        config.tranches,
        config.waterfall,
        timeline,
        drivers,
    )
    no_debt = run_corporate_model_no_debt(
        config.company, config.opening_balance_sheet, timeline, drivers
    )

    np.testing.assert_allclose(
        with_debt.income_statement.net_income, no_debt.income_statement.net_income, atol=1e-8
    )


def test_n_scenarios_one_matches_broadcast_with_debt():
    config = RootConfig.model_validate(debt_config_dict(n_periods=4))
    timeline = Timeline.annual(n_periods=4)
    single = deterministic_drivers(config, timeline)
    tiled = broadcast_drivers(single, n_scenarios=6)
    result_single = run_corporate_model_with_debt(
        config.company,
        config.opening_balance_sheet,
        config.tranches,
        config.waterfall,
        timeline,
        single,
    )
    result_tiled = run_corporate_model_with_debt(
        config.company,
        config.opening_balance_sheet,
        config.tranches,
        config.waterfall,
        timeline,
        tiled,
    )
    for s in range(6):
        np.testing.assert_allclose(
            result_tiled.balance_sheet.total_debt[s], result_single.balance_sheet.total_debt[0]
        )
