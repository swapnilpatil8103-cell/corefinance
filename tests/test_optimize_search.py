import numpy as np
import pytest

from corefin.assumptions.schema import RootConfig
from corefin.optimize.search import (
    decision_variable_grid_points,
    generate_search_drivers,
    grid_search,
)
from corefin.scenarios.generator import deterministic_drivers
from corefin.timeline import Timeline
from tests.test_debt_integration import debt_config_dict


def _optimizer_data(n_periods: int = 6, grid_points: int = 5, **optimizer_overrides) -> dict:
    data = debt_config_dict(n_periods=n_periods)
    data["optimizer"] = {
        "decision_variables": [
            {"tranche_name": "TLB", "min_multiple": 0.5, "max_multiple": 4.0},
            {"tranche_name": "Notes", "min_multiple": 0.0, "max_multiple": 1.5},
        ],
        "search": {"grid_points_per_dimension": grid_points, "n_scenarios_search": 400},
        **optimizer_overrides,
    }
    return data


class TestGridPoints:
    def test_no_step_uses_linspace(self):
        data = _optimizer_data()
        config = RootConfig.model_validate(data)
        dv = config.optimizer.decision_variables[0]
        points = decision_variable_grid_points(dv, grid_points_per_dimension=5)
        np.testing.assert_allclose(points, np.linspace(0.5, 4.0, 5))

    def test_with_step_covers_bounds(self):
        data = _optimizer_data()
        data["optimizer"]["decision_variables"][0]["step_multiple"] = 0.5
        config = RootConfig.model_validate(data)
        dv = config.optimizer.decision_variables[0]
        points = decision_variable_grid_points(dv, grid_points_per_dimension=999)
        assert points[0] == pytest.approx(0.5)
        assert points[-1] <= 4.0 + 1e-9
        diffs = np.diff(points)
        assert np.allclose(diffs, 0.5)


def _setup(config: RootConfig, n_periods: int):
    timeline = Timeline.annual(n_periods=n_periods)
    stochastic = generate_search_drivers(
        config,
        timeline,
        config.optimizer.search.n_scenarios_search,
        seed=config.optimizer.search.random_seed,
    )
    det = deterministic_drivers(config, timeline)
    return timeline, stochastic, det


def test_grid_search_evaluates_full_cartesian_product():
    config = RootConfig.model_validate(_optimizer_data(grid_points=4))
    timeline, stochastic, det = _setup(config, 6)
    result = grid_search(config, timeline, stochastic, det)
    assert result.n_evaluated == 4 * 4
    assert len(result.evaluations) == 16
    assert result.elapsed_seconds > 0


def test_common_random_numbers_identical_operating_results_across_grid():
    config = RootConfig.model_validate(_optimizer_data(grid_points=4))
    timeline, stochastic, det = _setup(config, 6)
    result = grid_search(config, timeline, stochastic, det)
    ran = [e for e in result.evaluations if e.result is not None]
    assert len(ran) > 1
    first_revenue = ran[0].result.income_statement.revenue
    for evaluation in ran[1:]:
        np.testing.assert_array_equal(evaluation.result.income_statement.revenue, first_revenue)


def test_grid_search_reports_evaluation_count_and_runtime():
    config = RootConfig.model_validate(_optimizer_data(n_periods=5, grid_points=3))
    timeline, stochastic, det = _setup(config, 5)
    result = grid_search(config, timeline, stochastic, det)
    assert result.n_evaluated == 9
    assert isinstance(result.elapsed_seconds, float)
    assert result.elapsed_seconds < 30.0


def test_grid_search_all_infeasible_returns_no_best():
    data = _optimizer_data(n_periods=5, grid_points=3)
    data["optimizer"]["deterministic_constraints"] = {"max_total_leverage": 0.01}
    config = RootConfig.model_validate(data)
    timeline, stochastic, det = _setup(config, 5)
    result = grid_search(config, timeline, stochastic, det)
    assert result.best_feasible is None


def test_flat_pricing_no_constraints_optimum_at_max_leverage_bound():
    """Sanity check: with no pricing penalty and no risk constraints, more
    leverage (less equity, same purchase price) should mechanically improve
    sponsor IRR, so the optimum should sit at the top of the decision
    variable bounds."""
    config = RootConfig.model_validate(_optimizer_data(grid_points=5))
    assert config.optimizer.pricing.tranches == []
    timeline, stochastic, det = _setup(config, 6)
    result = grid_search(config, timeline, stochastic, det)
    assert result.best_feasible is not None
    assert result.best_feasible.candidate.decision_values["TLB"] == pytest.approx(4.0)
    assert result.best_feasible.candidate.decision_values["Notes"] == pytest.approx(1.5)


def test_steep_pricing_grid_pulls_optimum_interior():
    data = _optimizer_data(
        grid_points=7,
        pricing={
            "tranches": [
                {
                    "tranche_name": "TLB",
                    "basis": "total_leverage",
                    "leverage_threshold": 1.0,
                    "spread_bps_per_turn": 0.15,  # very steep: 15% per turn above 1.0x
                }
            ]
        },
    )
    config = RootConfig.model_validate(data)
    timeline, stochastic, det = _setup(config, 6)
    result = grid_search(config, timeline, stochastic, det)
    assert result.best_feasible is not None
    # Should NOT be pinned at the upper bound once pricing punishes leverage hard.
    assert result.best_feasible.candidate.decision_values["TLB"] < 4.0


def test_tightening_stochastic_constraint_never_increases_optimal_leverage():
    base_data = _optimizer_data(grid_points=6)
    base_data["scenario"]["driver_vol"] = {"revenue_growth_std": 0.08, "ebitda_margin_std": 0.05}

    loose_data = {
        **base_data,
        "optimizer": {**base_data["optimizer"], "stochastic_constraints": {}},
    }
    tight_data = {
        **base_data,
        "optimizer": {
            **base_data["optimizer"],
            "stochastic_constraints": {"max_loss_of_capital_probability": 0.02},
        },
    }

    loose_config = RootConfig.model_validate(loose_data)
    tight_config = RootConfig.model_validate(tight_data)

    timeline, stochastic, det = _setup(loose_config, 6)
    loose_result = grid_search(loose_config, timeline, stochastic, det)
    tight_result = grid_search(tight_config, timeline, stochastic, det)

    assert loose_result.best_feasible is not None
    assert tight_result.best_feasible is not None
    loose_leverage = loose_result.best_feasible.candidate.leverage.total_leverage
    tight_leverage = tight_result.best_feasible.candidate.leverage.total_leverage
    assert tight_leverage <= loose_leverage + 1e-9
