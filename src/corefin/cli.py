"""CLI: `corefin run --config <path.yaml> [--output out.xlsx] [--scenario N]`."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import typer

from corefin.assumptions.loader import expand_series, load_config
from corefin.io.excel_export import export_scenario_to_excel
from corefin.metrics.covenants import evaluate_covenants
from corefin.metrics.credit_metrics import compute_credit_metrics
from corefin.scenarios.generator import generate_stochastic_drivers
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
    covenant_results = evaluate_covenants(root_config.covenants, credit_metrics, timeline)

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
        typer.echo("Covenant breach probability (any tested period):")
        for cov in covenant_results:
            breach_prob = np.mean(np.any(cov.breach, axis=1))
            typer.echo(f"  {cov.name}: {breach_prob:.1%}")
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


if __name__ == "__main__":
    app()
