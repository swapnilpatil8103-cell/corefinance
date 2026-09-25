import time

import numpy as np
import pytest

from corefin.assumptions.schema import RootConfig
from corefin.checks.corporate_checks import check_balance_sheet_balances, check_cash_ties_to_cfs
from corefin.checks.debt_checks import check_debt_rollforward, check_revolver_bounds
from corefin.checks.framework import run_checks
from corefin.optimize.structure import build_structure
from corefin.simulate.engine import run_simulation
from corefin.timeline import Timeline
from tests.conftest import minimal_config_dict


def _config(n_periods: int = 6, **scenario_overrides) -> RootConfig:
    data = minimal_config_dict(n_periods=n_periods)
    data["optimizer"] = {
        "decision_variables": [{"tranche_name": "TLB", "min_multiple": 1.0, "max_multiple": 3.0}],
    }
    data["scenario"].update(scenario_overrides)
    return RootConfig.model_validate(data)


def _candidate(config: RootConfig, tlb_multiple: float = 2.0):
    return build_structure(config, {"TLB": tlb_multiple})


def test_run_simulation_reproducible_with_fixed_seed():
    config = _config(driver_vol={"revenue_growth_std": 0.04, "base_rate_std": 0.01})
    timeline = Timeline.annual(n_periods=6)
    candidate = _candidate(config)
    a = run_simulation(config, candidate, timeline, n_scenarios=500, seed=11)
    b = run_simulation(config, candidate, timeline, n_scenarios=500, seed=11)
    np.testing.assert_array_equal(a.exit_result.irr, b.exit_result.irr)
    np.testing.assert_array_equal(
        a.credit_metrics.total_net_leverage, b.credit_metrics.total_net_leverage
    )


def test_run_simulation_shapes_match_requested_scenario_count():
    config = _config(driver_vol={"revenue_growth_std": 0.03})
    timeline = Timeline.annual(n_periods=6)
    candidate = _candidate(config)
    result = run_simulation(config, candidate, timeline, n_scenarios=250, seed=3)
    assert result.exit_result.irr.shape == (250,)
    assert result.model_result.balance_sheet.cash.shape == (250, 6)
    assert result.distress_by_year.shape == (250, 6)


def test_run_simulation_balance_sheet_and_debt_integrity_simple_mode():
    config = _config(
        driver_vol={
            "revenue_growth_std": 0.05,
            "ebitda_margin_std": 0.02,
            "base_rate_std": 0.01,
            "exit_multiple_std": 0.5,
        }
    )
    timeline = Timeline.annual(n_periods=6)
    candidate = _candidate(config)
    result = run_simulation(config, candidate, timeline, n_scenarios=400, seed=5)
    _assert_integrity(result, candidate)


def test_run_simulation_balance_sheet_and_debt_integrity_advanced_mode():
    config = _config(
        driver_vol={"revenue_growth_std": 0.05, "ebitda_margin_std": 0.02, "base_rate_std": 0.01},
        advanced={
            "persistence": {"revenue_growth_phi": 0.5, "ebitda_margin_phi": 0.4},
            "regime": {"annual_probability": 0.2, "duration_years": 2},
            "fat_tails": {"degrees_of_freedom": 5.0},
            "rate_mean_reversion": {"kappa": 0.3, "long_run_mean": 0.05},
            "exit_multiple_link": {"beta_growth": 3.0, "beta_rate": -2.0},
        },
    )
    timeline = Timeline.annual(n_periods=6)
    candidate = _candidate(config)
    result = run_simulation(config, candidate, timeline, n_scenarios=400, seed=5)
    _assert_integrity(result, candidate)


def _assert_integrity(result, candidate) -> None:
    run_checks(
        [
            check_balance_sheet_balances(result.model_result.balance_sheet),
            check_cash_ties_to_cfs(
                result.model_result.balance_sheet,
                result.model_result.cash_flow_statement,
                candidate.opening_balance_sheet.cash_mm,
            ),
            check_debt_rollforward(result.model_result.debt_schedule),
        ]
    )
    revolver = next(t for t in candidate.tranches if t.is_revolver)
    assert check_revolver_bounds(result.model_result.debt_schedule, revolver).passed


@pytest.mark.slow
def test_ten_thousand_scenarios_with_regimes_and_fat_tails_within_time_budget():
    config = _config(
        n_periods=7,
        driver_vol={
            "revenue_growth_std": 0.04,
            "ebitda_margin_std": 0.02,
            "base_rate_std": 0.01,
            "exit_multiple_std": 0.5,
        },
        advanced={
            "regime": {"annual_probability": 0.15, "duration_years": 2},
            "fat_tails": {"degrees_of_freedom": 5.0},
        },
    )
    timeline = Timeline.annual(n_periods=7)
    candidate = _candidate(config)

    start = time.perf_counter()
    result = run_simulation(config, candidate, timeline, n_scenarios=10_000, seed=1)
    elapsed = time.perf_counter() - start

    assert elapsed < 20.0, f"10k scenarios with regimes/fat tails took {elapsed:.2f}s"
    assert result.exit_result.irr.shape == (10_000,)
    run_checks([check_balance_sheet_balances(result.model_result.balance_sheet)])
