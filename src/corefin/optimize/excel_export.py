"""Excel export for an optimization run: every candidate evaluated (grid +
refinement), the efficient frontier among feasible candidates, and the
recommended structure's full statements/debt schedule/metrics -- the last
of these reusing io.excel_export's per-statement sheet writer directly, so
it's produced by the exact same code as a standalone `corefin run` export.
"""

from __future__ import annotations

import pandas as pd

from corefin.io.excel_export import write_scenario_sheets
from corefin.metrics.credit_metrics import CreditMetrics
from corefin.optimize.evaluate import CandidateEvaluation
from corefin.optimize.search import OptimizationResult
from corefin.timeline import Timeline

_RISK_COLUMNS = (
    "total_leverage",
    "senior_leverage",
    "equity_pct_of_sources",
    "interest_coverage_at_close",
    "covenant_breach_probability",
    "revolver_shortfall_probability",
    "loss_of_capital_probability",
    "mean_irr",
    "median_irr",
    "p10_irr",
    "mean_moic",
)


def _evaluation_row(evaluation: CandidateEvaluation, source: str) -> dict[str, object]:
    row: dict[str, object] = dict(evaluation.candidate.decision_values)
    row["Source"] = source
    row["Feasible"] = evaluation.feasible
    row["Objective"] = evaluation.objective_value
    for key in _RISK_COLUMNS:
        row[key] = evaluation.constraint_values.get(key)
    row["Violations"] = "; ".join(v.message for v in evaluation.violations)
    return row


def _grid_results_frame(optimization_result: OptimizationResult) -> pd.DataFrame:
    rows = [_evaluation_row(e, "grid") for e in optimization_result.grid_result.evaluations]
    if optimization_result.refinement_result is not None:
        rows += [
            _evaluation_row(e, "refine") for e in optimization_result.refinement_result.evaluations
        ]
    return pd.DataFrame(rows)


def _efficient_frontier_frame(optimization_result: OptimizationResult) -> pd.DataFrame:
    pool = list(optimization_result.grid_result.evaluations)
    if optimization_result.refinement_result is not None:
        pool += optimization_result.refinement_result.evaluations
    feasible = [e for e in pool if e.feasible]
    recommended_values = optimization_result.recommended.candidate.decision_values

    rows = []
    for e in feasible:
        row = _evaluation_row(e, source="")
        del row["Source"]
        del row["Violations"]
        row["Recommended"] = e.candidate.decision_values == recommended_values
        rows.append(row)
    frame = pd.DataFrame(rows)
    if not frame.empty:
        frame = frame.sort_values("mean_irr", ascending=False).reset_index(drop=True)
    return frame


def export_optimization_to_excel(
    path: str,
    timeline: Timeline,
    optimization_result: OptimizationResult,
    recommended_credit_metrics: CreditMetrics,
) -> None:
    confirmation = optimization_result.confirmation
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        _grid_results_frame(optimization_result).to_excel(
            writer, sheet_name="Grid Results", index=False
        )
        _efficient_frontier_frame(optimization_result).to_excel(
            writer, sheet_name="Efficient Frontier", index=False
        )
        write_scenario_sheets(
            writer,
            timeline,
            confirmation.result.income_statement,
            confirmation.result.balance_sheet,
            confirmation.result.cash_flow_statement,
            confirmation.result.debt_schedule,
            recommended_credit_metrics,
            scenario=0,
        )
