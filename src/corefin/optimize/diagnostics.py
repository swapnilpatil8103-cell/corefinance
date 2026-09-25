"""Binding-constraint diagnostics: what's actually stopping the optimizer
from adding more debt, distinct from "the reported value happened to be
within some tolerance of a limit."

Three layers, from cheapest/least informative to most:
1. `binding_items` -- is the recommended structure's OWN value close to one
   of its configured limits, or close to a decision-variable bound? A bound
   is reported separately from a constraint, since "the grid was too narrow"
   and "an economic limit was hit" are different problems with different
   fixes.
2. `limiting_constraints_from_neighbors` -- among the grid/refinement points
   immediately adjacent to the recommendation, which constraints made the
   *infeasible* ones fail, and how often? This is the direct answer to
   "what stops us adding more debt:" it's not just what's numerically close
   to binding at the optimum, it's what actually excluded the neighbors that
   would have had a better objective.
3. `relaxation_sensitivity` -- for each binding/limiting item, actually
   loosen it by a configured amount and re-run a full grid search on the
   same common random numbers, reporting the objective/leverage delta. A
   shadow-price table: which constraint, if loosened, would actually help?
"""

from __future__ import annotations

from dataclasses import dataclass

from corefin.assumptions.schema import RootConfig
from corefin.optimize.evaluate import CandidateEvaluation
from corefin.optimize.search import (
    GridSearchResult,
    OptimizationResult,
    RefinementResult,
    grid_search,
)
from corefin.timeline import Timeline

PROBABILITY_CONSTRAINT_NAMES = frozenset(
    {
        "max_covenant_breach_probability",
        "max_revolver_shortfall_probability",
        "max_loss_of_capital_probability",
    }
)

# (name, kind ["max"|"min"], constraint_values key)
_CONSTRAINT_SPECS = (
    ("max_total_leverage", "max", "total_leverage"),
    ("max_secured_leverage", "max", "secured_leverage"),
    ("min_equity_pct_of_sources", "min", "equity_pct_of_sources"),
    ("min_interest_coverage_at_close", "min", "interest_coverage_at_close"),
    ("max_covenant_breach_probability", "max", "covenant_breach_probability"),
    ("max_revolver_shortfall_probability", "max", "revolver_shortfall_probability"),
    ("max_loss_of_capital_probability", "max", "loss_of_capital_probability"),
)
_CONSTRAINT_DIRECTION = {name: direction for name, direction, _ in _CONSTRAINT_SPECS}
_CONSTRAINT_VALUE_KEY = {name: value_key for name, _, value_key in _CONSTRAINT_SPECS}


def _constraint_limit(root_config: RootConfig, name: str) -> float | None:
    """None for names outside the fixed set of seven config-scalar constraints
    -- including hard-infeasibility reasons like "market_capacity" and
    "sponsor_equity_non_negative" (see evaluate.py), which aren't a single
    relaxable config field and so are reported in the limiting-constraints
    listing but skipped by the relaxation table."""
    dc = root_config.optimizer.deterministic_constraints
    sc = root_config.optimizer.stochastic_constraints
    return {
        "max_total_leverage": dc.max_total_leverage,
        "max_secured_leverage": dc.max_secured_leverage,
        "min_equity_pct_of_sources": dc.min_equity_pct_of_sources,
        "min_interest_coverage_at_close": dc.min_interest_coverage_at_close,
        "max_covenant_breach_probability": sc.max_covenant_breach_probability,
        "max_revolver_shortfall_probability": sc.max_revolver_shortfall_probability,
        "max_loss_of_capital_probability": sc.max_loss_of_capital_probability,
    }.get(name)


@dataclass(frozen=True)
class BindingItem:
    name: str
    kind: str  # "constraint" or "bound"
    direction: str  # "max" or "min"
    limit: float
    actual: float
    gap: float  # signed distance to the limit (percentage points for probabilities,
    #             a % for other relative checks, turns of EBITDA for bounds); <= the
    #             configured tolerance means binding, <= 0 means violated
    decision_variable: str | None = None  # set only when kind == "bound"


def _refine_step_sizes(
    root_config: RootConfig,
    grid_result: GridSearchResult,
    refinement_result: RefinementResult | None,
) -> dict[str, float]:
    optimizer = root_config.optimizer
    steps: dict[str, float] = {}
    for name, coarse_points in grid_result.grid_values.items():
        coarse_step = float(coarse_points[1] - coarse_points[0]) if len(coarse_points) > 1 else 0.0
        if refinement_result is not None and optimizer.search.refine_grid_points_per_dimension > 1:
            # refine() spans [center-coarse_step, center+coarse_step] across
            # refine_grid_points_per_dimension points -- an approximation used
            # only for "is this close to its bound" reporting, not exact math.
            steps[name] = (2.0 * coarse_step) / (
                optimizer.search.refine_grid_points_per_dimension - 1
            )
        else:
            steps[name] = coarse_step
    return steps


def binding_items(
    evaluation: CandidateEvaluation,
    root_config: RootConfig,
    grid_result: GridSearchResult,
    refinement_result: RefinementResult | None,
) -> list[BindingItem]:
    optimizer = root_config.optimizer
    tol = optimizer.tolerances
    cv = evaluation.constraint_values
    items: list[BindingItem] = []

    for name, direction, value_key in _CONSTRAINT_SPECS:
        limit = _constraint_limit(root_config, name)
        actual = cv.get(value_key)
        if limit is None or actual is None:
            continue
        if name in PROBABILITY_CONSTRAINT_NAMES:
            gap = ((limit - actual) if direction == "max" else (actual - limit)) * 100.0
            if gap <= tol.probability_tolerance_pp:
                items.append(BindingItem(name, "constraint", direction, limit, actual, gap))
        else:
            denom = abs(limit) if limit != 0 else 1.0
            gap = ((limit - actual) if direction == "max" else (actual - limit)) / denom * 100.0
            if gap <= tol.relative_tolerance_pct:
                items.append(BindingItem(name, "constraint", direction, limit, actual, gap))

    refine_steps = _refine_step_sizes(root_config, grid_result, refinement_result)
    dv_by_name = {dv.tranche_name: dv for dv in optimizer.decision_variables}
    for dv_name, value in evaluation.candidate.decision_values.items():
        dv = dv_by_name[dv_name]
        step = refine_steps.get(dv_name, 0.0)
        if value - dv.min_multiple <= step:
            items.append(
                BindingItem(
                    f"{dv_name} (min bound)",
                    "bound",
                    "min",
                    dv.min_multiple,
                    value,
                    value - dv.min_multiple,
                    decision_variable=dv_name,
                )
            )
        if dv.max_multiple - value <= step:
            items.append(
                BindingItem(
                    f"{dv_name} (max bound)",
                    "bound",
                    "max",
                    dv.max_multiple,
                    value,
                    dv.max_multiple - value,
                    decision_variable=dv_name,
                )
            )
    return items


@dataclass(frozen=True)
class LimitingConstraintSummary:
    name: str
    count: int
    example_message: str


def limiting_constraints_from_neighbors(
    optimization_result: OptimizationResult,
) -> list[LimitingConstraintSummary]:
    grid_result = optimization_result.grid_result
    recommended_values = optimization_result.recommended.candidate.decision_values
    steps = {
        name: float(pts[1] - pts[0]) if len(pts) > 1 else 0.0
        for name, pts in grid_result.grid_values.items()
    }
    pool = list(grid_result.evaluations)
    if optimization_result.refinement_result is not None:
        pool += optimization_result.refinement_result.evaluations

    counts: dict[str, int] = {}
    examples: dict[str, str] = {}
    for evaluation in pool:
        if evaluation.feasible:
            continue
        dv = evaluation.candidate.decision_values
        if dv.keys() != recommended_values.keys():
            continue
        is_neighbor = all(
            abs(dv[name] - recommended_values[name]) <= 1.5 * steps.get(name, 0.0) + 1e-9
            for name in recommended_values
        )
        if not is_neighbor:
            continue
        for v in evaluation.violations:
            counts[v.name] = counts.get(v.name, 0) + 1
            examples.setdefault(v.name, v.message)

    return sorted(
        (
            LimitingConstraintSummary(name=name, count=count, example_message=examples[name])
            for name, count in counts.items()
        ),
        key=lambda s: -s.count,
    )


def _relax_root_config(root_config: RootConfig, item: BindingItem) -> tuple[RootConfig, float]:
    """Returns (relaxed config, the item's new limit value)."""
    optimizer = root_config.optimizer
    relax = optimizer.relaxation

    if item.kind == "bound":
        new_dvs = []
        new_limit = item.limit
        for dv in optimizer.decision_variables:
            if dv.tranche_name == item.decision_variable:
                if item.direction == "max":
                    new_limit = dv.max_multiple + relax.leverage_relax_turns
                    dv = dv.model_copy(update={"max_multiple": new_limit})
                else:
                    new_limit = max(0.0, dv.min_multiple - relax.leverage_relax_turns)
                    dv = dv.model_copy(update={"min_multiple": new_limit})
            new_dvs.append(dv)
        new_optimizer = optimizer.model_copy(update={"decision_variables": new_dvs})
        return root_config.model_copy(update={"optimizer": new_optimizer}), new_limit

    dc = optimizer.deterministic_constraints
    sc = optimizer.stochastic_constraints
    new_limit: float

    if item.name == "max_total_leverage":
        new_limit = dc.max_total_leverage + relax.leverage_relax_turns
        dc = dc.model_copy(update={"max_total_leverage": new_limit})
    elif item.name == "max_secured_leverage":
        new_limit = dc.max_secured_leverage + relax.leverage_relax_turns
        dc = dc.model_copy(update={"max_secured_leverage": new_limit})
    elif item.name == "min_equity_pct_of_sources":
        new_limit = max(0.0, dc.min_equity_pct_of_sources - relax.equity_pct_relax_pp / 100.0)
        dc = dc.model_copy(update={"min_equity_pct_of_sources": new_limit})
    elif item.name == "min_interest_coverage_at_close":
        new_limit = max(0.0, dc.min_interest_coverage_at_close - relax.coverage_relax_turns)
        dc = dc.model_copy(update={"min_interest_coverage_at_close": new_limit})
    elif item.name == "max_covenant_breach_probability":
        new_limit = min(
            1.0, sc.max_covenant_breach_probability + relax.probability_relax_pp / 100.0
        )
        sc = sc.model_copy(update={"max_covenant_breach_probability": new_limit})
    elif item.name == "max_revolver_shortfall_probability":
        new_limit = min(
            1.0, sc.max_revolver_shortfall_probability + relax.probability_relax_pp / 100.0
        )
        sc = sc.model_copy(update={"max_revolver_shortfall_probability": new_limit})
    elif item.name == "max_loss_of_capital_probability":
        new_limit = min(
            1.0, sc.max_loss_of_capital_probability + relax.probability_relax_pp / 100.0
        )
        sc = sc.model_copy(update={"max_loss_of_capital_probability": new_limit})
    else:
        raise ValueError(f"unhandled binding item for relaxation: {item.name}")

    new_optimizer = optimizer.model_copy(
        update={"deterministic_constraints": dc, "stochastic_constraints": sc}
    )
    return root_config.model_copy(update={"optimizer": new_optimizer}), new_limit


@dataclass(frozen=True)
class RelaxationResult:
    item_name: str
    kind: str
    original_limit: float
    relaxed_limit: float
    objective_before: float
    objective_after: float
    delta_objective: float
    leverage_before: float
    leverage_after: float


def relaxation_sensitivity(
    root_config: RootConfig,
    timeline: Timeline,
    optimization_result: OptimizationResult,
    items: list[BindingItem],
) -> list[RelaxationResult]:
    baseline_objective = optimization_result.recommended.objective_value
    baseline_leverage = optimization_result.recommended.candidate.leverage.total_leverage

    results: list[RelaxationResult] = []
    seen: set[str] = set()
    for item in items:
        if item.name in seen:
            continue
        seen.add(item.name)

        relaxed_config, new_limit = _relax_root_config(root_config, item)
        relaxed_result = grid_search(
            relaxed_config,
            timeline,
            optimization_result.search_stochastic_drivers,
            optimization_result.search_deterministic_drivers,
        )
        best = relaxed_result.best_feasible
        objective_after = best.objective_value if best is not None else float("nan")
        leverage_after = (
            best.candidate.leverage.total_leverage if best is not None else float("nan")
        )

        results.append(
            RelaxationResult(
                item_name=item.name,
                kind=item.kind,
                original_limit=item.limit,
                relaxed_limit=new_limit,
                objective_before=baseline_objective,
                objective_after=objective_after,
                delta_objective=(
                    objective_after - baseline_objective if best is not None else float("nan")
                ),
                leverage_before=baseline_leverage,
                leverage_after=leverage_after,
            )
        )
    return results


@dataclass(frozen=True)
class OptimizationDiagnostics:
    binding: list[BindingItem]
    limiting_constraints: list[LimitingConstraintSummary]
    relaxation: list[RelaxationResult]


def compute_diagnostics(
    root_config: RootConfig, timeline: Timeline, optimization_result: OptimizationResult
) -> OptimizationDiagnostics:
    binding = binding_items(
        optimization_result.confirmation,
        root_config,
        optimization_result.grid_result,
        optimization_result.refinement_result,
    )
    limiting = limiting_constraints_from_neighbors(optimization_result)

    relax_items_by_name = {item.name: item for item in binding}
    for summary in limiting:
        if summary.name in relax_items_by_name or summary.name not in _CONSTRAINT_DIRECTION:
            continue
        limit = _constraint_limit(root_config, summary.name)
        value_key = _CONSTRAINT_VALUE_KEY.get(summary.name)
        actual = (
            optimization_result.confirmation.constraint_values.get(value_key) if value_key else None
        )
        if limit is not None and actual is not None:
            relax_items_by_name[summary.name] = BindingItem(
                summary.name,
                "constraint",
                _CONSTRAINT_DIRECTION[summary.name],
                limit,
                actual,
                gap=float("nan"),
            )
    relaxation = relaxation_sensitivity(
        root_config, timeline, optimization_result, list(relax_items_by_name.values())
    )

    return OptimizationDiagnostics(
        binding=binding, limiting_constraints=limiting, relaxation=relaxation
    )
