"""Core Monte Carlo runner: given a structure (typically the optimizer's
recommendation) and a scenario count, generates driver paths (simple or
advanced -- see scenarios/generator.py), runs the exact same corporate
model/covenant/returns pipeline used everywhere else in corefin, and
packages every downstream analytics module needs into one SimulationResult.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from corefin.assumptions.schema import RootConfig
from corefin.metrics.covenants import CovenantResult, evaluate_covenants
from corefin.metrics.credit_metrics import CreditMetrics, compute_credit_metrics
from corefin.optimize.search import generate_search_drivers
from corefin.optimize.structure import CandidateStructure
from corefin.scenarios.drivers import DriverSet
from corefin.simulate.distress import compute_distress_flags
from corefin.statements.corporate_model import (
    CorporateModelWithDebtResult,
    run_corporate_model_with_debt,
)
from corefin.timeline import Timeline
from corefin.transaction.returns import ExitResult, compute_exit_and_returns


@dataclass(frozen=True)
class SimulationResult:
    candidate: CandidateStructure
    timeline: Timeline
    drivers: DriverSet
    model_result: CorporateModelWithDebtResult
    credit_metrics: CreditMetrics
    covenant_results: list[CovenantResult]
    exit_result: ExitResult
    distress_by_year: np.ndarray  # (n_scenarios, n_periods) bool


def run_simulation(
    root_config: RootConfig,
    candidate: CandidateStructure,
    timeline: Timeline,
    n_scenarios: int,
    seed: int | None = None,
) -> SimulationResult:
    """`generate_search_drivers` (optimize/search.py) is reused rather than
    reimplemented -- it's a generic "n_scenarios-overridden DriverSet"
    helper with no search-specific logic, already handling the
    simple/advanced dispatch via generate_stochastic_drivers."""
    drivers = generate_search_drivers(
        root_config,
        timeline,
        n_scenarios,
        seed=seed if seed is not None else root_config.scenario.random_seed,
    )
    model_result = run_corporate_model_with_debt(
        root_config.company,
        candidate.opening_balance_sheet,
        candidate.tranches,
        root_config.waterfall,
        timeline,
        drivers,
    )
    credit_metrics = compute_credit_metrics(
        model_result.debt_schedule,
        candidate.tranches,
        model_result.income_statement.ebitda,
        model_result.cash_flow_statement.capex,
        model_result.balance_sheet.cash,
        model_result.balance_sheet.total_debt,
    )
    covenant_results = evaluate_covenants(
        root_config.covenants,
        credit_metrics,
        timeline,
        model_result.debt_schedule,
        candidate.tranches,
    )
    exit_result = compute_exit_and_returns(
        candidate.sources_and_uses.sponsor_equity_mm,
        root_config.transaction.exit_year_index,
        drivers.exit_multiple,
        model_result.income_statement.ebitda,
        model_result.balance_sheet.total_debt,
        model_result.balance_sheet.cash,
        model_result.cash_flow_statement.dividends,
    )
    distress_by_year = compute_distress_flags(model_result.debt_schedule, credit_metrics)

    return SimulationResult(
        candidate=candidate,
        timeline=timeline,
        drivers=drivers,
        model_result=model_result,
        credit_metrics=credit_metrics,
        covenant_results=covenant_results,
        exit_result=exit_result,
        distress_by_year=distress_by_year,
    )
