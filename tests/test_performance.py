import time

import pytest

from corefin.assumptions.schema import RootConfig
from corefin.checks.corporate_checks import check_balance_sheet_balances
from corefin.checks.framework import run_checks
from corefin.scenarios.generator import generate_stochastic_drivers
from corefin.statements.corporate_model import run_corporate_model_with_debt
from corefin.timeline import Timeline
from tests.test_debt_integration import debt_config_dict

TIME_BUDGET_SECONDS = 15.0


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
