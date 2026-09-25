"""CLI: `corefin run --config <path.yaml> [--output out.xlsx] [--scenario N]`."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import typer

from corefin.assumptions.loader import expand_series, load_config
from corefin.assumptions.schema import RateType
from corefin.io.excel_export import export_scenario_to_excel
from corefin.metrics.covenants import (
    DEFAULT_MIN_HEADROOM_PCT,
    compute_base_case_headroom,
    evaluate_covenants,
    low_headroom_warnings,
)
from corefin.metrics.credit_metrics import compute_credit_metrics
from corefin.optimize.evaluate import binding_constraints, evaluate_candidate
from corefin.optimize.excel_export import export_optimization_to_excel
from corefin.optimize.heatmap import render_objective_heatmap
from corefin.optimize.pricing import build_priced_structure
from corefin.optimize.search import (
    NoFeasibleStructureError,
    generate_search_drivers,
    run_optimization,
)
from corefin.scenarios.generator import deterministic_drivers, generate_stochastic_drivers
from corefin.statements.corporate_model import run_corporate_model_with_debt
from corefin.timeline import Timeline
from corefin.transaction.returns import compute_exit_and_returns
from corefin.transaction.sources_uses import compute_sources_and_uses

app = typer.Typer(add_completion=False)


@app.command()
def version() -> None:
    """Print the corefin package version."""
    from importlib.metadata import version as pkg_version

    typer.echo(pkg_version("corefin"))


@app.command()
def run(
    config: Path = typer.Option(
        ..., "--config", exists=True, dir_okay=False, help="Path to a YAML assumptions file."
    ),
    output: Path = typer.Option(
        Path("output.xlsx"), "--output", help="Where to export the detailed scenario workbook."
    ),
    scenario: int = typer.Option(
        0, "--scenario", help="Scenario index to detail in the summary and export to Excel."
    ),
    min_headroom_pct: float = typer.Option(
        DEFAULT_MIN_HEADROOM_PCT,
        "--min-headroom-pct",
        help="Warn if a covenant's base-case cushion at its first tested period falls below this.",
    ),
) -> None:
    """Run the corporate LBO model, print a summary, and export one scenario's
    statements, debt schedule and metrics to Excel."""
    root_config = load_config(config)
    timeline = Timeline.annual(
        n_periods=root_config.timeline.n_periods,
        n_historical=root_config.timeline.n_historical,
        start_year=root_config.timeline.start_year,
    )
    drivers = generate_stochastic_drivers(
        root_config, timeline, seed=root_config.scenario.random_seed
    )
    result = run_corporate_model_with_debt(
        root_config.company,
        root_config.opening_balance_sheet,
        root_config.tranches,
        root_config.waterfall,
        timeline,
        drivers,
    )
    credit_metrics = compute_credit_metrics(
        result.debt_schedule,
        root_config.tranches,
        result.income_statement.ebitda,
        result.cash_flow_statement.capex,
        result.balance_sheet.cash,
        result.balance_sheet.total_debt,
    )
    covenant_results = evaluate_covenants(
        root_config.covenants, credit_metrics, timeline, result.debt_schedule, root_config.tranches
    )

    base_drivers = deterministic_drivers(root_config, timeline)
    base_result = run_corporate_model_with_debt(
        root_config.company,
        root_config.opening_balance_sheet,
        root_config.tranches,
        root_config.waterfall,
        timeline,
        base_drivers,
    )
    base_metrics = compute_credit_metrics(
        base_result.debt_schedule,
        root_config.tranches,
        base_result.income_statement.ebitda,
        base_result.cash_flow_statement.capex,
        base_result.balance_sheet.cash,
        base_result.balance_sheet.total_debt,
    )
    base_covenant_results = evaluate_covenants(
        root_config.covenants,
        base_metrics,
        timeline,
        base_result.debt_schedule,
        root_config.tranches,
    )
    headrooms = compute_base_case_headroom(base_covenant_results, timeline)
    warnings = low_headroom_warnings(headrooms, min_headroom_pct)

    entry_ebitda_margin = expand_series(root_config.company.ebitda_margin, timeline.n_periods)[0]
    sau = compute_sources_and_uses(
        root_config.transaction,
        root_config.tranches,
        root_config.company.revenue_base_mm,
        entry_ebitda_margin,
    )
    exit_result = compute_exit_and_returns(
        sau.sponsor_equity_mm,
        root_config.transaction.exit_year_index,
        drivers.exit_multiple,
        result.income_statement.ebitda,
        result.balance_sheet.total_debt,
        result.balance_sheet.cash,
        result.cash_flow_statement.dividends,
    )

    exit_label = timeline.year_labels[root_config.transaction.exit_year_index]
    typer.echo(f"corefin -- {config.name}")
    typer.echo(f"  Scenarios: {drivers.n_scenarios}   Periods: {timeline.n_periods}")
    typer.echo("")
    typer.echo("Sources & Uses (at entry, deal-fixed):")
    typer.echo(f"  Entry EBITDA:        {sau.entry_ebitda_mm:>10.1f}")
    typer.echo(f"  Purchase Price:      {sau.purchase_price_mm:>10.1f}")
    typer.echo(f"  Transaction Fees:    {sau.transaction_fees_mm:>10.1f}")
    typer.echo(f"  Financing Fees/OID:  {sau.financing_fees_and_oid_mm:>10.1f}")
    typer.echo(f"  Total Uses:          {sau.total_uses_mm:>10.1f}")
    typer.echo(f"  Debt Sources:        {sau.total_debt_sources_mm:>10.1f}")
    typer.echo(f"  Sponsor Equity:      {sau.sponsor_equity_mm:>10.1f}")
    typer.echo("")
    typer.echo(f"Sponsor returns across {drivers.n_scenarios} scenario(s), exit at {exit_label}:")
    typer.echo(
        f"  IRR   mean={np.mean(exit_result.irr):.1%}  median={np.median(exit_result.irr):.1%}  "
        f"p5={np.percentile(exit_result.irr, 5):.1%}  p95={np.percentile(exit_result.irr, 95):.1%}"
    )
    moic_mean, moic_median = np.mean(exit_result.moic), np.median(exit_result.moic)
    moic_p5, moic_p95 = np.percentile(exit_result.moic, 5), np.percentile(exit_result.moic, 95)
    typer.echo(
        f"  MOIC  mean={moic_mean:.2f}x  median={moic_median:.2f}x  "
        f"p5={moic_p5:.2f}x  p95={moic_p95:.2f}x"
    )
    typer.echo("")

    if covenant_results:
        typer.echo("Covenant breach probability (any tested period) and test frequency:")
        for cov in covenant_results:
            breach_prob = np.mean(np.any(cov.breach, axis=1))
            tested_prob = np.mean(cov.tested)
            typer.echo(
                f"  {cov.name}: breach={breach_prob:.1%}  "
                f"tested={tested_prob:.1%} of scenario-periods"
            )
        typer.echo("")

        typer.echo("Base-case headroom by covenant (deterministic, zero-vol run):")
        for h in headrooms:
            if not h.tested:
                typer.echo(f"  {h.name}: not tested in base case")
                continue
            typer.echo(
                f"  {h.name}: {h.cushion_pct:.1%} cushion at {h.period_label} "
                f"(value={h.metric_value:.2f}, threshold={h.threshold:.2f})"
            )
        typer.echo("")

        if warnings:
            for w in warnings:
                typer.echo(
                    f"WARNING: {w.name} has only {w.cushion_pct:.1%} headroom at {w.period_label} "
                    f"(value={w.metric_value:.2f} vs threshold={w.threshold:.2f}); "
                    f"below the {min_headroom_pct:.0%} minimum cushion."
                )
            typer.echo("")

    typer.echo(f"Detail for scenario {scenario}:")
    typer.echo(
        f"  Revenue: {result.income_statement.revenue[scenario, 0]:.1f} "
        f"-> {result.income_statement.revenue[scenario, -1]:.1f}"
    )
    typer.echo(
        f"  Total debt at exit: "
        f"{result.balance_sheet.total_debt[scenario, root_config.transaction.exit_year_index]:.1f}"
    )
    typer.echo(f"  IRR: {exit_result.irr[scenario]:.1%}   MOIC: {exit_result.moic[scenario]:.2f}x")

    export_scenario_to_excel(
        str(output),
        timeline,
        result.income_statement,
        result.balance_sheet,
        result.cash_flow_statement,
        result.debt_schedule,
        credit_metrics,
        scenario=scenario,
    )
    typer.echo("")
    typer.echo(f"Exported scenario {scenario} to {output}")


@app.command()
def optimize(
    config: Path = typer.Option(
        ..., "--config", exists=True, dir_okay=False, help="Path to a YAML assumptions file."
    ),
    output: Path = typer.Option(
        Path("optimization.xlsx"), "--output", help="Where to export the grid/frontier workbook."
    ),
    heatmap: Path = typer.Option(
        Path("heatmap.png"), "--heatmap", help="Where to export the objective heatmap PNG."
    ),
) -> None:
    """Search for the debt structure that maximizes sponsor returns subject to
    the config's financing constraints (config must have an `optimizer:`
    section), and export the results."""
    root_config = load_config(config)
    if root_config.optimizer is None:
        typer.echo(f"Error: {config} has no 'optimizer:' section.", err=True)
        raise typer.Exit(code=1)

    timeline = Timeline.annual(
        n_periods=root_config.timeline.n_periods,
        n_historical=root_config.timeline.n_historical,
        start_year=root_config.timeline.start_year,
    )

    try:
        result = run_optimization(root_config, timeline)
    except NoFeasibleStructureError as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    optimizer = root_config.optimizer
    confirmation = result.confirmation
    candidate = confirmation.candidate
    cv = confirmation.constraint_values

    recommended_credit_metrics = compute_credit_metrics(
        confirmation.result.debt_schedule,
        candidate.tranches,
        confirmation.result.income_statement.ebitda,
        confirmation.result.cash_flow_statement.capex,
        confirmation.result.balance_sheet.cash,
        confirmation.result.balance_sheet.total_debt,
    )

    typer.echo(f"corefin optimize -- {config.name}")
    typer.echo(
        f"  Grid search: {result.grid_result.n_evaluated} candidates in "
        f"{result.grid_result.elapsed_seconds:.2f}s"
    )
    if result.refinement_result is not None:
        typer.echo(
            f"  Refinement:  {result.refinement_result.n_evaluated} candidates in "
            f"{result.refinement_result.elapsed_seconds:.2f}s "
            f"(improved={result.refinement_result.improved})"
        )
    typer.echo("")

    typer.echo("Recommended structure:")
    for name, multiple in candidate.decision_values.items():
        size_mm = multiple * candidate.entry_ebitda_mm
        typer.echo(f"  {name}: {size_mm:>8.1f}mm  ({multiple:.2f}x EBITDA)")
    typer.echo(
        f"  Total leverage: {candidate.leverage.total_leverage:.2f}x   "
        f"Senior leverage: {candidate.leverage.senior_leverage:.2f}x"
    )
    equity_pct = (
        candidate.sources_and_uses.sponsor_equity_mm / candidate.sources_and_uses.total_uses_mm
    )
    typer.echo(
        f"  Sponsor equity:  {candidate.sources_and_uses.sponsor_equity_mm:>8.1f}mm  "
        f"({equity_pct:.1%} of sources)"
    )
    typer.echo("")

    typer.echo("Priced tranches:")
    for t in candidate.tranches:
        rate_desc = (
            f"{t.fixed_rate:.2%} fixed" if t.rate_type is RateType.FIXED else f"base+{t.spread:.2%}"
        )
        typer.echo(f"  {t.name}: {t.size_mm:>8.1f}mm, {rate_desc}")
    typer.echo("")

    search_obj, confirm_obj = result.recommended.objective_value, confirmation.objective_value
    n_search, n_confirm = optimizer.search.n_scenarios_search, optimizer.search.n_scenarios_confirm
    typer.echo(
        f"Objective ({optimizer.objective.kind}): {search_obj:.1%} (search, n={n_search})  "
        f"-> {confirm_obj:.1%} (confirmed, n={n_confirm})"
    )
    typer.echo(
        f"  IRR   mean={cv['mean_irr']:.1%}  median={cv['median_irr']:.1%}  p10={cv['p10_irr']:.1%}"
    )
    typer.echo(f"  MOIC  mean={cv['mean_moic']:.2f}x")
    typer.echo(f"  Covenant breach probability:     {cv['covenant_breach_probability']:.1%}")
    typer.echo(f"  Revolver shortfall probability:  {cv['revolver_shortfall_probability']:.1%}")
    typer.echo(f"  Loss-of-capital probability:     {cv['loss_of_capital_probability']:.1%}")
    typer.echo("")

    binding = binding_constraints(confirmation, root_config)
    typer.echo(f"Binding constraints: {', '.join(binding) if binding else 'none'}")
    if not confirmation.feasible:
        typer.echo(
            "WARNING: the confirmation run (larger sample) found this structure infeasible "
            "-- the search sample missed this:"
        )
        for v in confirmation.violations:
            typer.echo(f"  - {v.message}")
    typer.echo("")

    input_decision_values = {
        name: next(t for t in root_config.tranches if t.name == name).size_mm
        / candidate.entry_ebitda_mm
        for name in candidate.decision_values
    }
    input_candidate = build_priced_structure(root_config, input_decision_values)
    compare_stochastic = generate_search_drivers(
        root_config,
        timeline,
        optimizer.search.n_scenarios_confirm,
        seed=optimizer.search.random_seed,
    )
    compare_deterministic = deterministic_drivers(root_config, timeline)
    input_evaluation = evaluate_candidate(
        root_config, input_candidate, timeline, compare_stochastic, compare_deterministic
    )
    input_irr = input_evaluation.constraint_values.get("mean_irr", float("nan"))
    input_moic = input_evaluation.constraint_values.get("mean_moic", float("nan"))

    typer.echo("Comparison vs input config structure:")
    typer.echo(f"  {'':<20}{'Input':>12}{'Recommended':>14}")
    typer.echo(
        f"  {'Total leverage':<20}"
        f"{input_candidate.leverage.total_leverage:>11.2f}x"
        f"{candidate.leverage.total_leverage:>13.2f}x"
    )
    typer.echo(f"  {'Mean IRR':<20}{input_irr:>12.1%}{cv['mean_irr']:>14.1%}")
    typer.echo(f"  {'Mean MOIC':<20}{input_moic:>11.2f}x{cv['mean_moic']:>13.2f}x")
    typer.echo("")

    export_optimization_to_excel(str(output), timeline, result, recommended_credit_metrics)
    typer.echo(f"Exported grid/frontier/recommended structure to {output}")

    render_objective_heatmap(result, str(heatmap))
    typer.echo(f"Exported objective heatmap to {heatmap}")


if __name__ == "__main__":
    app()
