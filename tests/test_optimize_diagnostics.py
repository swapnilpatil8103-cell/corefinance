import dataclasses

import numpy as np
import pytest

from corefin.assumptions.schema import RootConfig
from corefin.optimize.diagnostics import (
    BindingItem,
    binding_items,
    compute_diagnostics,
    limiting_constraints_from_neighbors,
    relaxation_sensitivity,
)
from corefin.optimize.evaluate import CandidateEvaluation, ConstraintViolation
from corefin.optimize.pricing import build_priced_structure
from corefin.optimize.search import (
    GridSearchResult,
    OptimizationResult,
    generate_search_drivers,
    grid_search,
    run_optimization,
)
from corefin.scenarios.generator import deterministic_drivers, generate_stochastic_drivers
from corefin.timeline import Timeline
from tests.test_debt_integration import debt_config_dict


def _optimizer_data(n_periods: int = 5, grid_points: int = 4, **optimizer_overrides) -> dict:
    data = debt_config_dict(n_periods=n_periods)
    data["optimizer"] = {
        "decision_variables": [
            {"tranche_name": "TLB", "min_multiple": 0.5, "max_multiple": 4.0},
            {"tranche_name": "Notes", "min_multiple": 0.0, "max_multiple": 1.5},
        ],
        "search": {
            "grid_points_per_dimension": grid_points,
            "n_scenarios_search": 300,
            "n_scenarios_confirm": 600,
            "refine_grid_points_per_dimension": 3,
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


def test_probability_constraint_within_tolerance_is_binding_outside_is_not():
    data = _optimizer_data()
    data["optimizer"]["stochastic_constraints"] = {"max_covenant_breach_probability": 0.10}
    config = RootConfig.model_validate(data)
    timeline, stochastic, det = _setup(config)
    grid_result = grid_search(config, timeline, stochastic, det)
    base_evaluation = grid_result.best_feasible
    assert base_evaluation is not None

    # Default probability tolerance is 2pp -- 0.096 is 0.4pp below the 0.10
    # limit (within tolerance), 0.05 is 5pp below (not close).
    within = dataclasses.replace(
        base_evaluation,
        constraint_values={
            **base_evaluation.constraint_values,
            "covenant_breach_probability": 0.096,
        },
    )
    outside = dataclasses.replace(
        base_evaluation,
        constraint_values={
            **base_evaluation.constraint_values,
            "covenant_breach_probability": 0.05,
        },
    )

    items_within = binding_items(within, config, grid_result, None)
    items_outside = binding_items(outside, config, grid_result, None)

    assert any(i.name == "max_covenant_breach_probability" for i in items_within)
    assert not any(i.name == "max_covenant_breach_probability" for i in items_outside)


def test_decision_variable_at_its_bound_is_reported_as_a_binding_bound():
    data = _optimizer_data(grid_points=4)
    config = RootConfig.model_validate(data)
    timeline, stochastic, det = _setup(config)
    grid_result = grid_search(config, timeline, stochastic, det)
    base_evaluation = grid_result.best_feasible
    assert base_evaluation is not None

    dv_by_name = {dv.tranche_name: dv for dv in config.optimizer.decision_variables}
    at_bound_values = dict(base_evaluation.candidate.decision_values)
    at_bound_values["TLB"] = dv_by_name["TLB"].max_multiple
    candidate = dataclasses.replace(base_evaluation.candidate, decision_values=at_bound_values)
    evaluation = dataclasses.replace(base_evaluation, candidate=candidate)

    items = binding_items(evaluation, config, grid_result, None)
    bound_item = next(i for i in items if i.kind == "bound" and i.decision_variable == "TLB")
    assert bound_item.direction == "max"
    assert bound_item.limit == pytest.approx(dv_by_name["TLB"].max_multiple)


def test_limiting_constraint_analysis_identifies_the_only_possible_binder():
    data = debt_config_dict(n_periods=4)
    data["optimizer"] = {
        "decision_variables": [{"tranche_name": "TLB", "min_multiple": 1.0, "max_multiple": 3.0}],
    }
    config = RootConfig.model_validate(data)

    def _make(multiple, feasible, violations):
        candidate = build_priced_structure(config, {"TLB": multiple})
        return CandidateEvaluation(
            candidate=candidate,
            feasible=feasible,
            violations=violations,
            constraint_values={},
            objective_value=multiple,
        )

    low = _make(1.0, True, [])
    center = _make(2.0, True, [])
    high = _make(
        3.0,
        False,
        [
            ConstraintViolation(
                "max_total_leverage", "total leverage exceeds max", limit=2.5, actual=3.0
            )
        ],
    )

    grid_result = GridSearchResult(
        evaluations=[low, center, high],
        decision_variable_names=["TLB"],
        grid_values={"TLB": np.array([1.0, 2.0, 3.0])},
        best_feasible=center,
        n_evaluated=3,
        elapsed_seconds=0.0,
    )

    timeline = Timeline.annual(n_periods=4)
    stochastic = generate_stochastic_drivers(config, timeline, seed=1)
    det = deterministic_drivers(config, timeline)
    optimization_result = OptimizationResult(
        grid_result=grid_result,
        refinement_result=None,
        recommended=center,
        confirmation=center,
        fallback=None,
        search_stochastic_drivers=stochastic,
        search_deterministic_drivers=det,
    )

    summaries = limiting_constraints_from_neighbors(optimization_result)
    assert len(summaries) == 1
    assert summaries[0].name == "max_total_leverage"
    assert summaries[0].count == 1


def test_relaxation_never_lowers_objective_and_is_noop_when_not_binding():
    data = _optimizer_data(grid_points=5)
    data["optimizer"]["search"]["refine"] = False
    # Tight enough to actually constrain the feasible region (TLB alone can
    # reach 4.0x, Notes 1.5x -- 2.0x total leverage sits inside that range).
    data["optimizer"]["deterministic_constraints"] = {"max_total_leverage": 2.0}
    # Effectively unconstrained -- can never be binding.
    data["optimizer"]["stochastic_constraints"] = {"max_loss_of_capital_probability": 1.0}
    config = RootConfig.model_validate(data)
    timeline = Timeline.annual(n_periods=config.timeline.n_periods)
    result = run_optimization(config, timeline)

    binding_item = BindingItem(
        name="max_total_leverage",
        kind="constraint",
        direction="max",
        limit=2.0,
        actual=result.recommended.candidate.leverage.total_leverage,
        gap=0.0,
    )
    non_binding_item = BindingItem(
        name="max_loss_of_capital_probability",
        kind="constraint",
        direction="max",
        limit=1.0,
        actual=0.0,
        gap=100.0,
    )

    relaxed = relaxation_sensitivity(config, timeline, result, [binding_item, non_binding_item])
    by_name = {r.item_name: r for r in relaxed}

    # Relaxing max_total_leverage only ever adds newly-feasible candidates to
    # the search (same grid, same common random numbers) -- the best
    # objective over a superset can't be worse.
    assert by_name["max_total_leverage"].objective_after >= (
        by_name["max_total_leverage"].objective_before - 1e-9
    )
    # max_loss_of_capital_probability was never close to binding -- loosening
    # it further changes no candidate's feasibility, so the best objective is
    # unchanged.
    assert by_name["max_loss_of_capital_probability"].delta_objective == pytest.approx(
        0.0, abs=1e-9
    )


def test_compute_diagnostics_tolerates_hard_infeasibility_reasons_in_limiting_neighbors():
    """A neighbor can be infeasible for a "hard" reason (market capacity,
    negative sponsor equity) that has no single relaxable config field --
    those aren't in the seven-constraint spec table, so compute_diagnostics
    must not crash trying to look up a limit or relax them; they should
    still show up in the limiting-constraints listing, just not the
    relaxation table."""
    data = debt_config_dict(n_periods=4)
    data["optimizer"] = {
        "decision_variables": [{"tranche_name": "TLB", "min_multiple": 1.0, "max_multiple": 8.0}],
        "pricing": {
            "tranches": [
                {"tranche_name": "TLB", "leverage_threshold": 0.0, "market_capacity_mm": 250.0}
            ]
        },
        "search": {"grid_points_per_dimension": 5, "n_scenarios_search": 100},
    }
    config = RootConfig.model_validate(data)

    def _make(multiple, feasible, violations):
        candidate = build_priced_structure(config, {"TLB": multiple})
        return CandidateEvaluation(
            candidate=candidate,
            feasible=feasible,
            violations=violations,
            constraint_values={},
            objective_value=multiple,
        )

    low = _make(1.0, True, [])
    # entry EBITDA is 100mm -- 3.0x TLB = 300mm, over the 250mm cap.
    high = _make(
        3.0,
        False,
        [
            ConstraintViolation(
                "market_capacity", "TLB size 300.0mm exceeds market capacity 250.0mm"
            )
        ],
    )

    grid_result = GridSearchResult(
        evaluations=[low, high],
        decision_variable_names=["TLB"],
        grid_values={"TLB": np.array([1.0, 3.0])},
        best_feasible=low,
        n_evaluated=2,
        elapsed_seconds=0.0,
    )

    timeline = Timeline.annual(n_periods=4)
    stochastic = generate_stochastic_drivers(config, timeline, seed=1)
    det = deterministic_drivers(config, timeline)
    optimization_result = OptimizationResult(
        grid_result=grid_result,
        refinement_result=None,
        recommended=low,
        confirmation=low,
        fallback=None,
        search_stochastic_drivers=stochastic,
        search_deterministic_drivers=det,
    )

    diagnostics = compute_diagnostics(config, timeline, optimization_result)
    assert any(s.name == "market_capacity" for s in diagnostics.limiting_constraints)
    assert not any(r.item_name == "market_capacity" for r in diagnostics.relaxation)
