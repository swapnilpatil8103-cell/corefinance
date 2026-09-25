import numpy as np
import pytest

from corefin.assumptions.schema import RootConfig
from corefin.optimize.evaluate import binding_constraints
from corefin.optimize.search import (
    NoFeasibleStructureError,
    confirm,
    generate_search_drivers,
    grid_search,
    refine,
    run_optimization,
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
        "search": {
            "grid_points_per_dimension": grid_points,
            "n_scenarios_search": 500,
            "n_scenarios_confirm": 2000,
            "refine_grid_points_per_dimension": 5,
        },
        **optimizer_overrides,
    }
    return data


def _setup(config: RootConfig):
    timeline = Timeline.annual(n_periods=config.timeline.n_periods)
    stochastic = generate_search_drivers(
        config,
        timeline,
        config.optimizer.search.n_scenarios_search,
        seed=config.optimizer.search.random_seed,
    )
    det = deterministic_drivers(config, timeline)
    return timeline, stochastic, det


def test_refine_never_regresses_grid_best():
    config = RootConfig.model_validate(_optimizer_data(grid_points=5))
    timeline, stochastic, det = _setup(config)
    grid_result = grid_search(config, timeline, stochastic, det)
    refinement = refine(config, grid_result, timeline, stochastic, det)
    assert refinement.best.objective_value >= grid_result.best_feasible.objective_value - 1e-9


def test_refine_stays_within_decision_variable_bounds():
    config = RootConfig.model_validate(_optimizer_data(grid_points=5))
    timeline, stochastic, det = _setup(config)
    grid_result = grid_search(config, timeline, stochastic, det)
    refinement = refine(config, grid_result, timeline, stochastic, det)
    dv_by_name = {dv.tranche_name: dv for dv in config.optimizer.decision_variables}
    for name, value in refinement.best.candidate.decision_values.items():
        dv = dv_by_name[name]
        assert dv.min_multiple - 1e-9 <= value <= dv.max_multiple + 1e-9


def test_grid_and_refined_optimum_agree_within_tolerance_small_case():
    config = RootConfig.model_validate(_optimizer_data(n_periods=5, grid_points=4))
    timeline, stochastic, det = _setup(config)
    grid_result = grid_search(config, timeline, stochastic, det)
    refinement = refine(config, grid_result, timeline, stochastic, det)
    grid_irr = grid_result.best_feasible.objective_value
    refined_irr = refinement.best.objective_value
    # They're evaluating the same underlying smooth-ish surface with the same
    # common random numbers, just at different resolutions -- should be close.
    assert abs(refined_irr - grid_irr) < 0.02


def test_refine_raises_when_grid_found_nothing_feasible():
    data = _optimizer_data(n_periods=5, grid_points=3)
    data["optimizer"]["deterministic_constraints"] = {"max_total_leverage": 0.01}
    config = RootConfig.model_validate(data)
    timeline, stochastic, det = _setup(config)
    grid_result = grid_search(config, timeline, stochastic, det)
    with pytest.raises(NoFeasibleStructureError):
        refine(config, grid_result, timeline, stochastic, det)


def test_confirm_reevaluates_at_larger_scenario_count():
    config = RootConfig.model_validate(_optimizer_data(grid_points=3))
    timeline = Timeline.annual(n_periods=6)
    evaluation = confirm(config, {"TLB": 2.0, "Notes": 0.5}, timeline)
    assert evaluation.result is not None
    assert (
        evaluation.result.income_statement.revenue.shape[0]
        == config.optimizer.search.n_scenarios_confirm
    )


def test_run_optimization_end_to_end_recommended_and_confirmation():
    config = RootConfig.model_validate(_optimizer_data(grid_points=4))
    timeline = Timeline.annual(n_periods=6)
    result = run_optimization(config, timeline)
    assert result.recommended.feasible
    assert result.confirmation.result is not None
    assert (
        result.confirmation.result.income_statement.revenue.shape[0]
        == config.optimizer.search.n_scenarios_confirm
    )


def test_confirmation_satisfies_every_active_constraint():
    data = _optimizer_data(grid_points=5)
    data["optimizer"]["deterministic_constraints"] = {"min_equity_pct_of_sources": 0.20}
    data["optimizer"]["stochastic_constraints"] = {"max_loss_of_capital_probability": 0.10}
    config = RootConfig.model_validate(data)
    timeline = Timeline.annual(n_periods=6)
    result = run_optimization(config, timeline)
    assert result.confirmation.feasible
    assert result.confirmation.violations == []


def test_run_optimization_requires_optimizer_config():
    config = RootConfig.model_validate(debt_config_dict(n_periods=5))
    timeline = Timeline.annual(n_periods=5)
    with pytest.raises(ValueError, match="optimizer"):
        run_optimization(config, timeline)


def test_run_optimization_no_feasible_raises_clear_error():
    data = _optimizer_data(grid_points=3)
    data["optimizer"]["deterministic_constraints"] = {"max_total_leverage": 0.01}
    config = RootConfig.model_validate(data)
    timeline = Timeline.annual(n_periods=6)
    with pytest.raises(NoFeasibleStructureError):
        run_optimization(config, timeline)


def test_reproducibility_same_seed_gives_same_answer():
    config = RootConfig.model_validate(_optimizer_data(grid_points=4))
    timeline = Timeline.annual(n_periods=6)
    result_a = run_optimization(config, timeline)
    result_b = run_optimization(config, timeline)

    assert (
        result_a.recommended.candidate.decision_values
        == result_b.recommended.candidate.decision_values
    )
    assert result_a.recommended.objective_value == pytest.approx(
        result_b.recommended.objective_value
    )
    np.testing.assert_array_equal(
        result_a.confirmation.exit_result.irr, result_b.confirmation.exit_result.irr
    )


def test_binding_constraints_available_for_recommended_structure():
    data = _optimizer_data(grid_points=5)
    data["optimizer"]["deterministic_constraints"] = {"max_total_leverage": 3.5}
    config = RootConfig.model_validate(data)
    timeline = Timeline.annual(n_periods=6)
    result = run_optimization(config, timeline)
    names = binding_constraints(result.confirmation, config)
    assert isinstance(names, list)
