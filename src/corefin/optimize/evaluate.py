"""Constraint and objective evaluation for a single candidate structure.

Two hard infeasibilities are checked before running the (expensive) model at
all, since there's no sensible model to run for them: market capacity
(debt simply isn't available above it) and negative sponsor equity (debt
sources exceeding total uses -- not a fundable structure). Every other
constraint still runs the model and reports the actual value even when
violated, since the grid/frontier reporting wants to show near-misses, not
just pass/fail.

Two DriverSets are used per candidate: `stochastic_drivers` (shared via
common random numbers across every candidate in a search) for the
objective and the Monte Carlo constraints, and `deterministic_drivers`
(also shared, n_scenarios=1) for the "at close" deterministic constraint,
matching how the covenant headroom check already treats "at close" as the
zero-vol base case.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from corefin.assumptions.schema import ObjectiveConfig, ObjectiveKind, RootConfig
from corefin.metrics.covenants import evaluate_covenants
from corefin.metrics.credit_metrics import compute_credit_metrics
from corefin.optimize.pricing import check_market_capacity
from corefin.optimize.structure import CandidateStructure
from corefin.scenarios.drivers import DriverSet
from corefin.statements.corporate_model import (
    CorporateModelWithDebtResult,
    run_corporate_model_with_debt,
)
from corefin.timeline import Timeline
from corefin.transaction.returns import ExitResult, compute_exit_and_returns


@dataclass(frozen=True)
class ConstraintViolation:
    name: str
    message: str
    limit: float | None = None
    actual: float | None = None


@dataclass(frozen=True)
class CandidateEvaluation:
    candidate: CandidateStructure
    feasible: bool
    violations: list[ConstraintViolation]
    constraint_values: dict[str, float] = field(default_factory=dict)
    result: CorporateModelWithDebtResult | None = None
    exit_result: ExitResult | None = None
    objective_value: float = float("nan")


def compute_objective(objective: ObjectiveConfig, irr: np.ndarray) -> float:
    mean_irr = float(np.mean(irr))
    if objective.kind is ObjectiveKind.MEAN_IRR:
        return mean_irr
    if objective.kind is ObjectiveKind.MEDIAN_IRR:
        return float(np.median(irr))
    percentile_irr = float(np.percentile(irr, objective.percentile))
    if objective.kind is ObjectiveKind.PERCENTILE_IRR:
        return percentile_irr
    if objective.kind is ObjectiveKind.MEAN_DOWNSIDE_BLEND:
        return mean_irr - objective.downside_lambda * (mean_irr - percentile_irr)
    raise ValueError(f"unhandled objective kind: {objective.kind}")


def evaluate_candidate(
    root_config: RootConfig,
    candidate: CandidateStructure,
    timeline: Timeline,
    stochastic_drivers: DriverSet,
    deterministic_drivers: DriverSet,
) -> CandidateEvaluation:
    optimizer = root_config.optimizer
    sau = candidate.sources_and_uses
    lev = candidate.leverage
    violations: list[ConstraintViolation] = []

    capacity = check_market_capacity(candidate, optimizer.pricing)
    hard_infeasible = not capacity.feasible or sau.sponsor_equity_mm < 0
    if hard_infeasible:
        for msg in capacity.violations:
            violations.append(ConstraintViolation("market_capacity", msg))
        if sau.sponsor_equity_mm < 0:
            violations.append(
                ConstraintViolation(
                    "sponsor_equity_non_negative",
                    f"sponsor equity {sau.sponsor_equity_mm:.1f}mm is negative -- debt "
                    "sources exceed total uses; not a fundable structure",
                    limit=0.0,
                    actual=sau.sponsor_equity_mm,
                )
            )
        return CandidateEvaluation(candidate=candidate, feasible=False, violations=violations)

    dc = optimizer.deterministic_constraints
    equity_pct = sau.sponsor_equity_mm / sau.total_uses_mm
    constraint_values: dict[str, float] = {
        "total_leverage": lev.total_leverage,
        "secured_leverage": lev.secured_leverage,
        "equity_pct_of_sources": equity_pct,
    }

    def _check_max(name: str, limit: float | None, actual: float, message: str) -> None:
        if limit is not None and actual > limit:
            violations.append(ConstraintViolation(name, message, limit=limit, actual=actual))

    def _check_min(name: str, limit: float | None, actual: float, message: str) -> None:
        if limit is not None and actual < limit:
            violations.append(ConstraintViolation(name, message, limit=limit, actual=actual))

    _check_max(
        "max_total_leverage",
        dc.max_total_leverage,
        lev.total_leverage,
        f"total leverage {lev.total_leverage:.2f}x exceeds max {(dc.max_total_leverage or 0):.2f}x",
    )
    _check_max(
        "max_secured_leverage",
        dc.max_secured_leverage,
        lev.secured_leverage,
        f"secured leverage {lev.secured_leverage:.2f}x exceeds "
        f"max {(dc.max_secured_leverage or 0):.2f}x",
    )
    _check_min(
        "min_equity_pct_of_sources",
        dc.min_equity_pct_of_sources,
        equity_pct,
        f"equity {equity_pct:.1%} of sources is below the "
        f"{(dc.min_equity_pct_of_sources or 0):.1%} minimum",
    )

    result = run_corporate_model_with_debt(
        root_config.company,
        candidate.opening_balance_sheet,
        candidate.tranches,
        root_config.waterfall,
        timeline,
        stochastic_drivers,
    )
    base_result = run_corporate_model_with_debt(
        root_config.company,
        candidate.opening_balance_sheet,
        candidate.tranches,
        root_config.waterfall,
        timeline,
        deterministic_drivers,
    )
    credit_metrics = compute_credit_metrics(
        result.debt_schedule,
        candidate.tranches,
        result.income_statement.ebitda,
        result.cash_flow_statement.capex,
        result.balance_sheet.cash,
        result.balance_sheet.total_debt,
    )
    base_credit_metrics = compute_credit_metrics(
        base_result.debt_schedule,
        candidate.tranches,
        base_result.income_statement.ebitda,
        base_result.cash_flow_statement.capex,
        base_result.balance_sheet.cash,
        base_result.balance_sheet.total_debt,
    )
    interest_coverage_at_close = float(base_credit_metrics.interest_coverage[0, 0])
    constraint_values["interest_coverage_at_close"] = interest_coverage_at_close
    _check_min(
        "min_interest_coverage_at_close",
        dc.min_interest_coverage_at_close,
        interest_coverage_at_close,
        f"interest coverage at close {interest_coverage_at_close:.2f}x is below the "
        f"{(dc.min_interest_coverage_at_close or 0):.2f}x minimum",
    )

    covenant_results = evaluate_covenants(
        root_config.covenants, credit_metrics, timeline, result.debt_schedule, candidate.tranches
    )
    combined_breach = np.zeros(result.income_statement.ebitda.shape, dtype=bool)
    for cov in covenant_results:
        combined_breach |= cov.breach
    covenant_breach_probability = float(np.mean(np.any(combined_breach, axis=1)))
    constraint_values["covenant_breach_probability"] = covenant_breach_probability

    revolver_shortfall_probability = float(
        np.mean(np.any(result.debt_schedule.shortfall_flag, axis=1))
    )
    constraint_values["revolver_shortfall_probability"] = revolver_shortfall_probability

    exit_result = compute_exit_and_returns(
        sau.sponsor_equity_mm,
        root_config.transaction.exit_year_index,
        stochastic_drivers.exit_multiple,
        result.income_statement.ebitda,
        result.balance_sheet.total_debt,
        result.balance_sheet.cash,
        result.cash_flow_statement.dividends,
    )
    loss_of_capital_probability = float(np.mean(exit_result.moic < 1.0))
    constraint_values["loss_of_capital_probability"] = loss_of_capital_probability
    constraint_values["mean_irr"] = float(np.mean(exit_result.irr))
    constraint_values["median_irr"] = float(np.median(exit_result.irr))
    constraint_values["p10_irr"] = float(np.percentile(exit_result.irr, 10))
    constraint_values["mean_moic"] = float(np.mean(exit_result.moic))

    sc = optimizer.stochastic_constraints
    _check_max(
        "max_covenant_breach_probability",
        sc.max_covenant_breach_probability,
        covenant_breach_probability,
        f"covenant breach probability {covenant_breach_probability:.1%} exceeds max "
        f"{(sc.max_covenant_breach_probability or 0):.1%}",
    )
    _check_max(
        "max_revolver_shortfall_probability",
        sc.max_revolver_shortfall_probability,
        revolver_shortfall_probability,
        f"revolver shortfall probability {revolver_shortfall_probability:.1%} exceeds max "
        f"{(sc.max_revolver_shortfall_probability or 0):.1%}",
    )
    _check_max(
        "max_loss_of_capital_probability",
        sc.max_loss_of_capital_probability,
        loss_of_capital_probability,
        f"loss-of-capital probability {loss_of_capital_probability:.1%} exceeds max "
        f"{(sc.max_loss_of_capital_probability or 0):.1%}",
    )

    objective_value = compute_objective(optimizer.objective, exit_result.irr)

    return CandidateEvaluation(
        candidate=candidate,
        feasible=not violations,
        violations=violations,
        constraint_values=constraint_values,
        result=result,
        exit_result=exit_result,
        objective_value=objective_value,
    )
