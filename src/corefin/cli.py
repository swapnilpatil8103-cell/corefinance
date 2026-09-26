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
from corefin.optimize.diagnostics import compute_diagnostics
from corefin.optimize.evaluate import evaluate_candidate
from corefin.optimize.excel_export import export_optimization_to_excel
from corefin.optimize.heatmap import render_leverage_frontier, render_objective_heatmap
from corefin.optimize.pricing import build_priced_structure, pricing_sanity_warnings
from corefin.optimize.search import (
    NoConfirmedStructureError,
    NoFeasibleStructureError,
    generate_search_drivers,
    run_optimization,
)
from corefin.scenarios.generator import deterministic_drivers, generate_stochastic_drivers
from corefin.simulate.attribution import (
    compute_value_creation_bridge,
    summarize_value_creation_bridge,
)
from corefin.simulate.charts import (
    render_breach_distress_chart,
    render_irr_histogram,
    render_leverage_fan_chart,
    render_structure_comparison_chart,
    render_tornado_chart,
    render_value_bridge_waterfall,
)
from corefin.simulate.compare import (
    bump_decision_values,
    compare_structures,
    input_config_decision_values,
)
from corefin.simulate.excel_export import export_simulation_to_excel
from corefin.simulate.importance import compute_driver_importance, compute_tornado_chart
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
    pricing_warnings = pricing_sanity_warnings(
        root_config.tranches, float(base_drivers.base_rate[0, 0])
    )

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

    if pricing_warnings:
        for w in pricing_warnings:
            typer.echo(f"WARNING: {w}")
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
    leverage_frontier: Path = typer.Option(
        Path("leverage_frontier.png"),
        "--leverage-frontier",
        help="Where to export the best-objective-by-leverage PNG.",
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
    except (NoFeasibleStructureError, NoConfirmedStructureError) as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    diagnostics = compute_diagnostics(root_config, timeline, result)

    optimizer = root_config.optimizer
    confirmation = result.confirmation
    candidate = confirmation.candidate
    cv = confirmation.constraint_values
    search_cv = result.recommended.constraint_values

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

    if result.fallback is not None:
        status = "succeeded" if result.fallback.succeeded else "FAILED"
        typer.echo(
            f"NOTE: the original search recommendation failed on confirmation; fell back "
            f"to the next-best candidate after {result.fallback.attempts} attempt(s) ({status})."
        )
        typer.echo("")

    typer.echo("Recommended structure:")
    for name, multiple in candidate.decision_values.items():
        size_mm = multiple * candidate.entry_ebitda_mm
        typer.echo(f"  {name}: {size_mm:>8.1f}mm  ({multiple:.2f}x EBITDA)")
    typer.echo(
        f"  Total leverage: {candidate.leverage.total_leverage:.2f}x   "
        f"Secured leverage: {candidate.leverage.secured_leverage:.2f}x"
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

    pricing_warnings = pricing_sanity_warnings(
        candidate.tranches, float(result.search_deterministic_drivers.base_rate[0, 0])
    )
    if pricing_warnings:
        for w in pricing_warnings:
            typer.echo(f"WARNING: {w}")
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
    typer.echo("")

    typer.echo("Stochastic constraints, search sample -> confirmation sample:")
    for label, key in (
        ("Covenant breach probability", "covenant_breach_probability"),
        ("Revolver shortfall probability", "revolver_shortfall_probability"),
        ("Loss-of-capital probability", "loss_of_capital_probability"),
    ):
        typer.echo(f"  {label + ':':<32}{search_cv.get(key, float('nan')):>7.1%} -> {cv[key]:.1%}")
    typer.echo("")

    if not confirmation.feasible:
        typer.echo("WARNING: the confirmation run (larger sample) found this structure infeasible:")
        for v in confirmation.violations:
            typer.echo(f"  - {v.message}")
        typer.echo("")

    if diagnostics.binding:
        typer.echo("Binding at the optimum:")
        for item in diagnostics.binding:
            typer.echo(
                f"  [{item.kind:<10}] {item.name}: {item.actual:.3f} vs {item.direction} "
                f"limit {item.limit:.3f} (gap {item.gap:.2f})"
            )
    else:
        typer.echo("Binding at the optimum: none within tolerance")
    typer.echo("")

    if diagnostics.limiting_constraints:
        typer.echo("What's limiting adjacent (untried-better) candidates:")
        for summary in diagnostics.limiting_constraints:
            typer.echo(f"  {summary.name}: blocked {summary.count} neighboring candidate(s)")
            typer.echo(f"    e.g. {summary.example_message}")
    else:
        typer.echo("What's limiting adjacent candidates: no infeasible neighbors found")
    typer.echo("")

    if diagnostics.relaxation:
        typer.echo("Relaxation sensitivity (loosen one item, re-search, same random numbers):")
        for r in diagnostics.relaxation:
            delta_str = (
                f"{r.delta_objective:+.2%}" if r.delta_objective == r.delta_objective else "n/a"
            )
            leverage_str = f"{r.leverage_before:.2f}x -> {r.leverage_after:.2f}x"
            typer.echo(
                f"  {r.item_name} [{r.kind}]: {r.original_limit:.2f} -> {r.relaxed_limit:.2f}   "
                f"objective {delta_str}   leverage {leverage_str}"
            )
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

    export_optimization_to_excel(
        str(output), timeline, result, diagnostics, recommended_credit_metrics
    )
    typer.echo(f"Exported grid/frontier/recommended structure to {output}")

    render_objective_heatmap(result, str(heatmap))
    typer.echo(f"Exported objective heatmap to {heatmap}")

    render_leverage_frontier(result, str(leverage_frontier))
    typer.echo(f"Exported leverage frontier to {leverage_frontier}")


@app.command()
def simulate(
    config: Path = typer.Option(
        ..., "--config", exists=True, dir_okay=False, help="Path to a YAML assumptions file."
    ),
    structure: str = typer.Option(
        "both", "--structure", help="Which structure(s) to run: input, optimized, or both."
    ),
    scenarios: int = typer.Option(5000, "--scenarios", help="Number of Monte Carlo scenarios."),
    output: Path = typer.Option(
        Path("simulation.xlsx"), "--output", help="Where to export the Excel workbook."
    ),
    charts_dir: Path = typer.Option(
        Path("."), "--charts-dir", help="Directory to write the six chart PNGs into."
    ),
) -> None:
    """Stress-tests a deal structure (typically the optimizer's
    recommendation) with the Sponsor LBO Monte Carlo engine: richer
    scenario generation, named stress scenarios, downside/distress
    analytics, returns attribution, driver importance and structure
    comparison. Config must have an `optimizer:` section (the structure
    builder is reused from there) -- `simulate:` is optional and adds
    stress scenarios and an IRR hurdle."""
    if structure not in ("input", "optimized", "both"):
        typer.echo(
            f"Error: --structure must be one of input/optimized/both (got {structure!r}).",
            err=True,
        )
        raise typer.Exit(code=1)

    root_config = load_config(config)
    if root_config.optimizer is None:
        typer.echo(f"Error: {config} has no 'optimizer:' section.", err=True)
        raise typer.Exit(code=1)

    timeline = Timeline.annual(
        n_periods=root_config.timeline.n_periods,
        n_historical=root_config.timeline.n_historical,
        start_year=root_config.timeline.start_year,
    )
    irr_hurdle = root_config.simulate.irr_hurdle if root_config.simulate is not None else 0.15

    named_decision_values: dict[str, dict[str, float]] = {}
    if structure in ("input", "both"):
        named_decision_values["Input"] = input_config_decision_values(root_config)
    if structure in ("optimized", "both"):
        try:
            optimization_result = run_optimization(root_config, timeline)
        except (NoFeasibleStructureError, NoConfirmedStructureError) as exc:
            typer.echo(f"Error: {exc}", err=True)
            raise typer.Exit(code=1) from exc
        recommended = optimization_result.recommended.candidate.decision_values
        named_decision_values["Optimized"] = recommended
        first_dv_name = root_config.optimizer.decision_variables[0].tranche_name
        named_decision_values[f"Optimized +1.0x {first_dv_name}"] = bump_decision_values(
            recommended, first_dv_name, 1.0
        )
    primary_name = "Optimized" if "Optimized" in named_decision_values else "Input"

    typer.echo(f"corefin simulate -- {config.name}")
    typer.echo(f"  Scenarios: {scenarios}   Structures: {', '.join(named_decision_values)}")
    typer.echo("")

    entries = compare_structures(
        root_config,
        timeline,
        named_decision_values,
        n_scenarios=scenarios,
        seed=root_config.scenario.random_seed,
        irr_hurdle=irr_hurdle,
    )
    entries_by_name = {e.name: e for e in entries}
    primary = entries_by_name[primary_name]

    typer.echo("Structure comparison (common random numbers):")
    header = "".join(f"{name:>24}" for name in entries_by_name)
    typer.echo(f"  {'':<30}{header}")

    def _row(label: str, fmt, getter) -> None:
        typer.echo(f"  {label:<30}" + "".join(f"{fmt(getter(e)):>24}" for e in entries))

    _row("Total leverage", lambda v: f"{v:.2f}x", lambda e: e.candidate.leverage.total_leverage)
    _row("Mean IRR", lambda v: f"{v:.1%}", lambda e: e.downside.returns.mean_irr)
    _row("Median IRR", lambda v: f"{v:.1%}", lambda e: e.downside.returns.median_irr)
    _row(
        "Expected shortfall (worst 10%)",
        lambda v: f"{v:.1%}",
        lambda e: e.downside.returns.expected_shortfall_irr_10pct,
    )
    _row("P(MOIC < 1.0x)", lambda v: f"{v:.1%}", lambda e: e.downside.returns.prob_moic_below_1)
    _row(
        f"P(IRR < {irr_hurdle:.0%} hurdle)",
        lambda v: f"{v:.1%}",
        lambda e: e.downside.returns.prob_irr_below_hurdle,
    )
    _row(
        "Covenant breach probability",
        lambda v: f"{v:.1%}",
        lambda e: e.downside.covenant_breaches.overall_breach_probability,
    )
    _row(
        "Distress probability",
        lambda v: f"{v:.1%}",
        lambda e: e.downside.distress.overall_probability,
    )
    typer.echo("")

    r = primary.downside.returns
    typer.echo(f"Detail for {primary_name}:")
    typer.echo(
        "  IRR percentiles: " + "  ".join(f"p{p}={v:.1%}" for p, v in r.percentiles_irr.items())
    )
    typer.echo(
        "  MOIC percentiles: " + "  ".join(f"p{p}={v:.2f}x" for p, v in r.percentiles_moic.items())
    )
    convergence = primary.downside.convergence
    trace = " / ".join(f"{v:.2%}" for v in convergence.mean_irr_trace)
    typer.echo(
        f"  Convergence: mean IRR @ {convergence.scenario_counts} scenarios = {trace}  "
        f"(stable: {convergence.is_stable})"
    )
    typer.echo("")

    if primary.stress_results:
        typer.echo(f"Named stress scenarios ({primary_name}):")
        for s in primary.stress_results:
            distress_label = "YES" if s.distress_overall else "no"
            typer.echo(
                f"  {s.name}: IRR={s.irr:.1%}  MOIC={s.moic:.2f}x  "
                f"MinLiquidity=${s.min_liquidity_mm:.1f}mm  "
                f"MaxRevolverDraw={s.max_revolver_draw_pct:.0%}  Distress={distress_label}"
            )
        typer.echo("")

    bridge = compute_value_creation_bridge(root_config, primary.simulation)
    bridge_summaries = summarize_value_creation_bridge(bridge, primary.simulation.exit_result.irr)
    typer.echo(f"Value creation bridge ({primary_name}):")
    for s in bridge_summaries:
        typer.echo(
            f"  {s.label:<8} EBITDA={s.ebitda_growth:>8.1f}  Multiple={s.multiple_change:>8.1f}  "
            f"Debt/Cash={s.debt_paydown_and_cash:>8.1f}  Fees={s.fees_and_leakage:>7.1f}  "
            f"Total={s.total_value_creation:>8.1f}"
        )
    typer.echo("")

    driver_importance = compute_driver_importance(primary.simulation)
    typer.echo(f"Driver importance ({primary_name}):")
    for imp in driver_importance:
        typer.echo(
            f"  {imp.name:<20} beta={imp.standardized_coefficient:+.4f}  "
            f"rank_corr={imp.rank_correlation:+.3f}"
        )
    typer.echo("")

    tornado = compute_tornado_chart(root_config, primary.candidate, timeline, primary.simulation)
    typer.echo(f"Tornado (p10 -> base -> p90), {primary_name}:")
    for bar in sorted(tornado, key=lambda b: -b.irr_range):
        typer.echo(
            f"  {bar.name:<20} {bar.irr_at_p10:.1%} -> {bar.base_irr:.1%} -> {bar.irr_at_p90:.1%}  "
            f"(range {bar.irr_range:.1%})"
        )
    typer.echo("")

    charts_dir.mkdir(parents=True, exist_ok=True)
    render_irr_histogram(
        primary.downside, primary.simulation.exit_result.irr, str(charts_dir / "irr_histogram.png")
    )
    render_leverage_fan_chart(primary.downside, timeline, str(charts_dir / "leverage_fan.png"))
    render_breach_distress_chart(
        primary.downside, timeline, str(charts_dir / "breach_distress.png")
    )
    render_value_bridge_waterfall(bridge_summaries[0], str(charts_dir / "value_bridge.png"))
    render_tornado_chart(tornado, str(charts_dir / "tornado.png"))
    render_structure_comparison_chart(entries, str(charts_dir / "structure_comparison.png"))

    export_simulation_to_excel(
        str(output), timeline, entries, primary_name, bridge_summaries, driver_importance, tornado
    )

    typer.echo(f"Exported workbook to {output}")
    typer.echo(
        f"Exported charts to {charts_dir}/ (irr_histogram.png, leverage_fan.png, "
        "breach_distress.png, value_bridge.png, tornado.png, structure_comparison.png)"
    )


if __name__ == "__main__":
    app()
