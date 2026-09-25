import time

import pytest

from corefin.assumptions.schema import RootConfig
from corefin.checks.corporate_checks import check_balance_sheet_balances
from corefin.checks.framework import run_checks
from corefin.optimize.search import generate_search_drivers, grid_search
from corefin.scenarios.generator import deterministic_drivers, generate_stochastic_drivers
from corefin.statements.corporate_model import run_corporate_model_with_debt
from corefin.timeline import Timeline
from tests.test_debt_integration import debt_config_dict

TIME_BUDGET_SECONDS = 15.0
GRID_SEARCH_TIME_BUDGET_SECONDS = 60.0


@pytest.mark.slow
def test_ten_thousand_scenarios_seven_periods_within_time_budget():
    data = debt_config_dict(n_periods=7)
    data["scenario"]["n_scenarios"] = 10_000
    data["scenario"]["driver_vol"] = {
        "revenue_growth_std": 0.03,
        "ebitda_margin_std": 0.015,
        "capex_pct_revenue_std": 0.005,
        "nwc_pct_revenue_std": 0.01,
        "exit_multiple_std": 0.5,
        "base_rate_std": 0.005,
    }
    config = RootConfig.model_validate(data)
    timeline = Timeline.annual(n_periods=7)

    start = time.perf_counter()
    drivers = generate_stochastic_drivers(config, timeline, seed=1)
    result = run_corporate_model_with_debt(
        config.company,
        config.opening_balance_sheet,
        config.tranches,
        config.waterfall,
        timeline,
        drivers,
    )
    elapsed = time.perf_counter() - start

    assert elapsed < TIME_BUDGET_SECONDS, f"10k scenarios x 7 periods took {elapsed:.2f}s"
    assert result.balance_sheet.cash.shape == (10_000, 7)
    run_checks([check_balance_sheet_balances(result.balance_sheet)])


@pytest.mark.slow
def test_fifteen_by_fifteen_grid_two_thousand_scenarios_within_time_budget():
    data = debt_config_dict(n_periods=7)
    data["scenario"]["driver_vol"] = {
        "revenue_growth_std": 0.03,
        "ebitda_margin_std": 0.015,
        "capex_pct_revenue_std": 0.005,
        "nwc_pct_revenue_std": 0.01,
        "exit_multiple_std": 0.5,
        "base_rate_std": 0.005,
    }
    data["optimizer"] = {
        "decision_variables": [
            {"tranche_name": "TLB", "min_multiple": 1.0, "max_multiple": 5.0},
            {"tranche_name": "Notes", "min_multiple": 0.0, "max_multiple": 2.5},
        ],
        "search": {"grid_points_per_dimension": 15, "n_scenarios_search": 2000},
    }
    config = RootConfig.model_validate(data)
    timeline = Timeline.annual(n_periods=7)
    stochastic_drivers = generate_search_drivers(config, timeline, n_scenarios=2000, seed=1)
    det_drivers = deterministic_drivers(config, timeline)

    result = grid_search(config, timeline, stochastic_drivers, det_drivers)

    assert result.n_evaluated == 15 * 15
    assert result.elapsed_seconds < GRID_SEARCH_TIME_BUDGET_SECONDS, (
        f"15x15 grid @ 2,000 scenarios took {result.elapsed_seconds:.2f}s"
    )
