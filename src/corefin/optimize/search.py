"""Grid search over decision-variable combinations.

Common random numbers: `stochastic_drivers` and `deterministic_drivers` are
generated once by the caller (see `generate_search_drivers`) and reused for
every candidate evaluated here, so differences in the objective between
candidates reflect the structure, not fresh sampling noise for each one.
"""

from __future__ import annotations

import itertools
import time
from dataclasses import dataclass

import numpy as np

from corefin.assumptions.schema import DecisionVariableConfig, RootConfig
from corefin.optimize.evaluate import CandidateEvaluation, evaluate_candidate
from corefin.optimize.pricing import build_priced_structure
from corefin.scenarios.drivers import DriverSet
from corefin.scenarios.generator import deterministic_drivers, generate_stochastic_drivers
from corefin.timeline import Timeline


class NoFeasibleStructureError(RuntimeError):
    pass


def generate_search_drivers(
    root_config: RootConfig, timeline: Timeline, n_scenarios: int, seed: int
) -> DriverSet:
    """A DriverSet with the given scenario count/seed, without needing to touch
    scenarios/generator.py: pydantic's model_copy overrides just the one field
    generate_stochastic_drivers reads (config.scenario.n_scenarios)."""
    overridden = root_config.model_copy(
        update={"scenario": root_config.scenario.model_copy(update={"n_scenarios": n_scenarios})}
    )
    return generate_stochastic_drivers(overridden, timeline, seed=seed)


def decision_variable_grid_points(
    decision_variable: DecisionVariableConfig, grid_points_per_dimension: int
) -> np.ndarray:
    if decision_variable.step_multiple is None:
        return np.linspace(
            decision_variable.min_multiple,
            decision_variable.max_multiple,
            grid_points_per_dimension,
        )
    span = decision_variable.max_multiple - decision_variable.min_multiple
    n_points = int(np.floor(span / decision_variable.step_multiple + 1e-9)) + 1
    points = decision_variable.min_multiple + decision_variable.step_multiple * np.arange(n_points)
    return np.clip(points, decision_variable.min_multiple, decision_variable.max_multiple)


def _evaluate_grid(
    root_config: RootConfig,
    timeline: Timeline,
    stochastic_drivers: DriverSet,
    deterministic_drivers: DriverSet,
    grid_values: dict[str, np.ndarray],
) -> tuple[list[CandidateEvaluation], float]:
    names = list(grid_values.keys())
    start = time.perf_counter()
    evaluations: list[CandidateEvaluation] = []
    for combo in itertools.product(*(grid_values[name] for name in names)):
        decision_values = dict(zip(names, (float(v) for v in combo), strict=True))
        candidate = build_priced_structure(root_config, decision_values)
        evaluation = evaluate_candidate(
            root_config, candidate, timeline, stochastic_drivers, deterministic_drivers
        )
        evaluations.append(evaluation)
    elapsed_seconds = time.perf_counter() - start
    return evaluations, elapsed_seconds


def _best_feasible(evaluations: list[CandidateEvaluation]) -> CandidateEvaluation | None:
    feasible = [e for e in evaluations if e.feasible]
    return max(feasible, key=lambda e: e.objective_value) if feasible else None


@dataclass(frozen=True)
class GridSearchResult:
    evaluations: list[CandidateEvaluation]
    decision_variable_names: list[str]
    grid_values: dict[str, np.ndarray]
    best_feasible: CandidateEvaluation | None
    n_evaluated: int
    elapsed_seconds: float


def grid_search(
    root_config: RootConfig,
    timeline: Timeline,
    stochastic_drivers: DriverSet,
    deterministic_drivers: DriverSet,
) -> GridSearchResult:
    optimizer = root_config.optimizer
    names = [dv.tranche_name for dv in optimizer.decision_variables]
    grid_values = {
        dv.tranche_name: decision_variable_grid_points(
            dv, optimizer.search.grid_points_per_dimension
        )
        for dv in optimizer.decision_variables
    }

    evaluations, elapsed_seconds = _evaluate_grid(
        root_config, timeline, stochastic_drivers, deterministic_drivers, grid_values
    )

    return GridSearchResult(
        evaluations=evaluations,
        decision_variable_names=names,
        grid_values=grid_values,
        best_feasible=_best_feasible(evaluations),
        n_evaluated=len(evaluations),
        elapsed_seconds=elapsed_seconds,
    )


@dataclass(frozen=True)
class RefinementResult:
    evaluations: list[CandidateEvaluation]
    best: CandidateEvaluation
    n_evaluated: int
    elapsed_seconds: float
    improved: bool


def refine(
    root_config: RootConfig,
    grid_result: GridSearchResult,
    timeline: Timeline,
    stochastic_drivers: DriverSet,
    deterministic_drivers: DriverSet,
) -> RefinementResult:
    """A finer grid in a shrunk neighborhood (+/- one coarse grid step, clipped
    to the decision variable's own bounds) around the best coarse-grid point.
    Falls back to the coarse best if nothing in the finer neighborhood is both
    feasible and at least as good -- refinement can only improve or match,
    never regress, the reported recommendation."""
    if grid_result.best_feasible is None:
        raise NoFeasibleStructureError("cannot refine: grid search found no feasible candidate")

    optimizer = root_config.optimizer
    dv_by_name = {dv.tranche_name: dv for dv in optimizer.decision_variables}
    best_values = grid_result.best_feasible.candidate.decision_values

    neighborhoods: dict[str, np.ndarray] = {}
    for name, coarse_points in grid_result.grid_values.items():
        dv = dv_by_name[name]
        coarse_step = float(coarse_points[1] - coarse_points[0]) if len(coarse_points) > 1 else 0.0
        center = best_values[name]
        low = max(dv.min_multiple, center - coarse_step)
        high = min(dv.max_multiple, center + coarse_step)
        if high <= low:
            neighborhoods[name] = np.array([center])
        else:
            neighborhoods[name] = np.linspace(
                low, high, optimizer.search.refine_grid_points_per_dimension
            )

    evaluations, elapsed_seconds = _evaluate_grid(
        root_config, timeline, stochastic_drivers, deterministic_drivers, neighborhoods
    )
    best_refined = _best_feasible(evaluations)

    grid_objective = grid_result.best_feasible.objective_value
    if best_refined is not None and best_refined.objective_value >= grid_objective:
        return RefinementResult(
            evaluations=evaluations,
            best=best_refined,
            n_evaluated=len(evaluations),
            elapsed_seconds=elapsed_seconds,
            improved=best_refined.objective_value > grid_objective,
        )
    return RefinementResult(
        evaluations=evaluations,
        best=grid_result.best_feasible,
        n_evaluated=len(evaluations),
        elapsed_seconds=elapsed_seconds,
        improved=False,
    )


def confirm(
    root_config: RootConfig, decision_values: dict[str, float], timeline: Timeline
) -> CandidateEvaluation:
    """Re-evaluates one structure at a larger scenario count, for a more
    robust read on its risk/return profile than the (smaller, faster)
    search count. Same seed as the search for reproducibility -- a
    different n_scenarios naturally draws a different stream even with an
    identical seed, which is expected, not a bug."""
    optimizer = root_config.optimizer
    confirm_stochastic = generate_search_drivers(
        root_config,
        timeline,
        optimizer.search.n_scenarios_confirm,
        seed=optimizer.search.random_seed,
    )
    confirm_deterministic = deterministic_drivers(root_config, timeline)
    candidate = build_priced_structure(root_config, decision_values)
    return evaluate_candidate(
        root_config, candidate, timeline, confirm_stochastic, confirm_deterministic
    )


@dataclass(frozen=True)
class OptimizationResult:
    grid_result: GridSearchResult
    refinement_result: RefinementResult | None
    recommended: CandidateEvaluation
    confirmation: CandidateEvaluation


def run_optimization(root_config: RootConfig, timeline: Timeline) -> OptimizationResult:
    if root_config.optimizer is None:
        raise ValueError("root_config.optimizer must be set to run the optimizer")
    optimizer = root_config.optimizer

    stochastic_drivers = generate_search_drivers(
        root_config,
        timeline,
        optimizer.search.n_scenarios_search,
        seed=optimizer.search.random_seed,
    )
    det_drivers = deterministic_drivers(root_config, timeline)

    grid_result = grid_search(root_config, timeline, stochastic_drivers, det_drivers)
    if grid_result.best_feasible is None:
        raise NoFeasibleStructureError(
            "no candidate in the search grid satisfied every configured constraint; "
            "widen the decision variable bounds or loosen a constraint"
        )

    if optimizer.search.refine:
        refinement_result = refine(
            root_config, grid_result, timeline, stochastic_drivers, det_drivers
        )
        recommended = refinement_result.best
    else:
        refinement_result = None
        recommended = grid_result.best_feasible

    confirmation = confirm(root_config, recommended.candidate.decision_values, timeline)

    return OptimizationResult(
        grid_result=grid_result,
        refinement_result=refinement_result,
        recommended=recommended,
        confirmation=confirmation,
    )
