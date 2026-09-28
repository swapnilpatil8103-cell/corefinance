"""`corefin credit fetch` / `corefin credit build` -- the Credit-Loss
Forecasting Engine's data pipeline commands. Requires `corefin[fig]`.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import typer

from corefin.credit import charts, macro
from corefin.credit.panel import build_industry_nco_rate_report, build_panel, flag_chargeoff_gaps
from corefin.credit.schema import LoanCategory
from corefin.credit.sources import fed_scenarios as fed_scenarios_source
from corefin.credit.sources import ffiec, ffiec_parse, fred

app = typer.Typer(add_completion=False, help="Credit-Loss Forecasting Engine data pipeline.")

DEFAULT_RAW_DIR = Path("data/raw/ffiec")
DEFAULT_PANEL_PATH = Path("data/processed/credit_panel.parquet")
DEFAULT_COVERAGE_REPORT_PATH = Path("data/processed/coverage_report.csv")
DEFAULT_MACRO_RAW_DIR = Path("data/raw/fred")
DEFAULT_MACRO_HISTORY_PATH = Path("data/processed/macro_history.parquet")
DEFAULT_SCENARIO_DIR = Path("data/processed/scenarios")
DEFAULT_MODELING_DATASET_PATH = Path("data/processed/modeling_dataset.parquet")
DEFAULT_CHARTS_DIR = Path("data/processed/charts")


def _quarter_range(start: str, end: str) -> list[pd.Period]:
    return list(pd.period_range(pd.Period(start, freq="Q"), pd.Period(end, freq="Q"), freq="Q"))


def _quarter_to_ffiec_date(quarter: pd.Period) -> str:
    """Quarter-end date in the "MM/DD/YYYY" format the FFIEC site's own
    dropdown uses -- confirmed live, months ARE zero-padded there
    ("03/31/2001", not "3/31/2001"); day is always 30 or 31 so padding
    doesn't matter for it, but getting the month wrong here silently
    misses every real dropdown entry for Q1/Q2/Q3 (Q4's December never
    needed padding, which is why this bug only showed up on 3 out of 4
    quarters a year)."""
    end_timestamp = quarter.end_time
    return f"{end_timestamp.month:02d}/{end_timestamp.day:02d}/{end_timestamp.year}"


def _cached_zip_path(raw_dir: Path, quarter: pd.Period) -> Path:
    date_str = _quarter_to_ffiec_date(quarter).replace("/", "-")
    return raw_dir / f"{date_str}.zip"


@app.command()
def fetch(
    start: str = typer.Option(..., "--start", help='First quarter, e.g. "2001Q1".'),
    end: str = typer.Option(..., "--end", help='Last quarter (inclusive), e.g. "2026Q2".'),
    raw_dir: Path = typer.Option(
        DEFAULT_RAW_DIR, "--raw-dir", help="Gitignored directory to cache raw bulk ZIPs in."
    ),
) -> None:
    """Download (or reuse cached) FFIEC bulk Call Report ZIPs for every
    quarter in [start, end]. No account or API key needed."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    quarters = _quarter_range(start, end)
    for quarter in quarters:
        path = _cached_zip_path(raw_dir, quarter)
        if path.exists():
            typer.echo(f"{quarter}: cached ({path.stat().st_size:,} bytes)")
            continue
        typer.echo(f"{quarter}: downloading...")
        try:
            content = ffiec.fetch_bulk_call_report_zip(_quarter_to_ffiec_date(quarter))
        except ffiec.FfiecPeriodNotFoundError:
            typer.echo(f"{quarter}: not available from FFIEC -- skipping", err=True)
            continue
        path.write_bytes(content)
        typer.echo(f"{quarter}: cached ({len(content):,} bytes)")


@app.command()
def build(
    start: str = typer.Option(..., "--start", help='First quarter, e.g. "2001Q1".'),
    end: str = typer.Option(..., "--end", help='Last quarter (inclusive), e.g. "2026Q2".'),
    raw_dir: Path = typer.Option(
        DEFAULT_RAW_DIR, "--raw-dir", help="Directory `fetch` cached raw bulk ZIPs into."
    ),
    output: Path = typer.Option(
        DEFAULT_PANEL_PATH, "--output", help="Where to write the panel as parquet."
    ),
    min_balance: float = typer.Option(
        1_000.0,
        "--min-balance",
        help="Minimum balance (thousands of dollars) to keep a bank-quarter row.",
    ),
    include_combined_consumer: bool = typer.Option(
        True,
        "--include-combined-consumer/--no-include-combined-consumer",
        help="Include AUTO_AND_OTHER_CONSUMER_COMBINED (overlaps with AUTO + OTHER_CONSUMER).",
    ),
) -> None:
    """Parse cached bulk ZIPs (run `fetch` first) for every quarter in
    [start, end] and build the bank-category-quarter panel, written as
    parquet. Missing quarters are skipped with a warning, not fatal.
    Prints a summary of skipped malformed rows per (schedule, quarter) at
    the end, and warns if any single one loses more than
    ffiec_parse.BAD_ROW_WARNING_THRESHOLD rows."""
    quarters = _quarter_range(start, end)
    categories = list(LoanCategory) if include_combined_consumer else [
        c for c in LoanCategory if c != LoanCategory.AUTO_AND_OTHER_CONSUMER_COMBINED
    ]

    item_frames = []
    bad_row_counts: dict[tuple[pd.Period, str], int] = {}
    for quarter in quarters:
        path = _cached_zip_path(raw_dir, quarter)
        if not path.exists():
            typer.echo(
                f"{quarter}: no cached ZIP (run `corefin credit fetch` first) -- skipping", err=True
            )
            continue
        typer.echo(f"{quarter}: parsing...")
        item_frame, schedule_bad_rows = ffiec_parse.parse_bulk_zip(path.read_bytes(), quarter)
        item_frames.append(item_frame)
        for schedule, count in schedule_bad_rows.items():
            if count > 0:
                bad_row_counts[(quarter, schedule)] = count

    if not item_frames:
        typer.echo("no quarters parsed -- nothing to build", err=True)
        raise typer.Exit(code=1)

    typer.echo("building panel...")
    combined_item_frame = pd.concat(item_frames, ignore_index=True)
    result = build_panel(combined_item_frame, categories=categories, min_balance=min_balance)

    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(output, index=False)
    typer.echo(f"wrote {len(result):,} rows to {output}")

    typer.echo("\nMalformed rows skipped during parsing (schedule, quarter -> row count):")
    if not bad_row_counts:
        typer.echo("  none")
    for (quarter, schedule), count in sorted(bad_row_counts.items(), key=lambda kv: kv[0]):
        typer.echo(f"  {schedule} {quarter}: {count}")
        if count > ffiec_parse.BAD_ROW_WARNING_THRESHOLD:
            typer.echo(
                f"  WARNING: {schedule} {quarter} lost {count} rows (> "
                f"{ffiec_parse.BAD_ROW_WARNING_THRESHOLD}) -- investigate before trusting this "
                "quarter's data for that schedule",
                err=True,
            )


@app.command()
def coverage(
    panel_path: Path = typer.Option(
        DEFAULT_PANEL_PATH, "--panel", help="Parquet panel written by `build`."
    ),
    nco_start: str = typer.Option(
        "2006Q1", "--nco-start", help="First quarter of the NCO-rate report."
    ),
    nco_end: str = typer.Option("2012Q4", "--nco-end", help="Last quarter of the NCO-rate report."),
    gap_start: str = typer.Option(
        "2008Q1", "--gap-start", help="First quarter of the gap-check window."
    ),
    gap_end: str = typer.Option(
        "2010Q4", "--gap-end", help="Last quarter of the gap-check window."
    ),
    min_balance: float = typer.Option(
        1_000.0,
        "--min-balance",
        help="Exclude bank-quarters below this average balance (thousands of "
        "dollars) from industry NCO rates -- a near-zero denominator produces "
        "an extreme, noise-dominated rate.",
    ),
    exclude_merger_flagged: bool = typer.Option(
        True,
        "--exclude-merger-flagged/--include-merger-flagged",
        help="Exclude merger-flagged bank-quarters from industry NCO rates (their "
        "quarterly flow is contaminated by the acquired bank's prior activity).",
    ),
    output: Path = typer.Option(
        DEFAULT_COVERAGE_REPORT_PATH, "--output", help="Where to write the coverage report."
    ),
) -> None:
    """Coverage (bank count / coverage share / aggregate balance) and
    industry NCO-rate report from an already-built panel, plus a
    charge-off-gap flag per category over [gap_start, gap_end]."""
    panel = pd.read_parquet(panel_path)
    panel["quarter"] = pd.PeriodIndex(panel["quarter"].astype(str), freq="Q")

    grouped_balance = panel.groupby(["category", "quarter"], observed=True)["balance"]
    coverage_report = grouped_balance.agg(
        n_banks="size",
        coverage_share=lambda s: s.notna().mean(),
        aggregate_balance=lambda s: s.sum(skipna=True),
    ).reset_index()

    nco_report = build_industry_nco_rate_report(
        panel, exclude_merger_flagged=exclude_merger_flagged, min_balance=min_balance
    )
    nco_window = nco_report[
        (nco_report["quarter"] >= pd.Period(nco_start, freq="Q"))
        & (nco_report["quarter"] <= pd.Period(nco_end, freq="Q"))
    ]

    gaps = flag_chargeoff_gaps(nco_report, start=gap_start, end=gap_end)
    typer.echo("Charge-off gaps in the crisis window:")
    for category, has_gap in gaps.items():
        typer.echo(f"  {category}: {'GAP' if has_gap else 'ok'}")

    merged = coverage_report.merge(
        nco_window[["category", "quarter", "industry_nco_rate", "has_chargeoff_data"]],
        on=["category", "quarter"],
        how="left",
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(output, index=False)
    typer.echo(f"wrote coverage+NCO report to {output}")


@app.command("fetch-macro")
def fetch_macro(
    start: str = typer.Option(
        "2001-01-01", "--start", help="First date to pull FRED history from."
    ),
    end: str | None = typer.Option(
        None, "--end", help="Last date to pull FRED history through (defaults to latest available)."
    ),
    raw_dir: Path = typer.Option(
        DEFAULT_MACRO_RAW_DIR,
        "--raw-dir",
        help="Gitignored directory to cache each series' raw FRED pull in.",
    ),
    output: Path = typer.Option(
        DEFAULT_MACRO_HISTORY_PATH,
        "--output",
        help="Where to write the quarterly macro history (wide, one column per variable).",
    ),
) -> None:
    """Fetch every FRED series in fred.FED_SCENARIO_VARIABLE_TO_FRED (or
    reuse a cached raw pull), aggregate to quarterly and apply each
    variable's own transform (see credit/macro.py's module docstring for
    the verified aggregation/transform rules), and write the resulting
    wide history to `output`. Requires FRED_API_KEY."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    raw_by_variable: dict[str, pd.DataFrame] = {}
    for variable, mapping in fred.FED_SCENARIO_VARIABLE_TO_FRED.items():
        cache_path = raw_dir / f"{mapping.fred_series_id}.csv"
        if cache_path.exists():
            typer.echo(f"{variable} ({mapping.fred_series_id}): cached")
            observations = pd.read_csv(cache_path, parse_dates=["date"])
        else:
            typer.echo(f"{variable} ({mapping.fred_series_id}): fetching...")
            observations = fred.fetch_series(mapping.fred_series_id, start_date=start, end_date=end)
            observations.to_csv(cache_path, index=False)
        raw_by_variable[variable] = observations

    history = macro.build_fred_history_from_raw(raw_by_variable)
    output.parent.mkdir(parents=True, exist_ok=True)
    history.reset_index().assign(quarter=lambda d: d["quarter"].astype(str)).to_parquet(
        output, index=False
    )
    typer.echo(f"wrote {len(history):,} quarters x {len(history.columns)} variables to {output}")


@app.command("fetch-scenarios")
def fetch_scenarios(
    vintage: int = typer.Option(
        fed_scenarios_source.CURRENT_SCENARIO_VINTAGE,
        "--vintage",
        help="Fed scenario vintage year.",
    ),
    macro_history_path: Path = typer.Option(
        DEFAULT_MACRO_HISTORY_PATH, "--macro-history", help="Output of `fetch-macro`."
    ),
    output_dir: Path = typer.Option(
        DEFAULT_SCENARIO_DIR, "--output-dir", help="Where to write each normalized scenario."
    ),
) -> None:
    """Fetch the Fed's baseline and severely-adverse scenarios for
    `vintage`, normalize them to the same quarter-indexed/variable-name
    shape `fetch-macro` produces, verify each starts exactly one quarter
    after the macro history's last actual quarter (no gap, no overlap --
    raises if not), and write both to `output_dir`. No key needed."""
    history = pd.read_parquet(macro_history_path)
    history["quarter"] = pd.PeriodIndex(history["quarter"].astype(str), freq="Q")
    history = history.set_index("quarter")

    output_dir.mkdir(parents=True, exist_ok=True)
    for scenario_name in ("baseline", "severely_adverse"):
        typer.echo(f"{scenario_name}: fetching...")
        raw = fed_scenarios_source.fetch_scenario(scenario_name, vintage=vintage)
        normalized = macro.normalize_scenario(raw)
        macro.assert_scenario_continues_from_history(history, normalized)
        path = output_dir / f"{scenario_name}.parquet"
        normalized.reset_index().assign(quarter=lambda d: d["quarter"].astype(str)).to_parquet(
            path, index=False
        )
        typer.echo(f"{scenario_name}: continuity OK, wrote {len(normalized)} quarters to {path}")


@app.command("build-modeling-dataset")
def build_modeling_dataset_cmd(
    panel_path: Path = typer.Option(
        DEFAULT_PANEL_PATH, "--panel", help="Parquet panel from `build`."
    ),
    macro_history_path: Path = typer.Option(
        DEFAULT_MACRO_HISTORY_PATH, "--macro-history", help="Output of `fetch-macro`."
    ),
    output: Path = typer.Option(
        DEFAULT_MODELING_DATASET_PATH, "--output", help="Where to write the modeling dataset."
    ),
    min_balance: float = typer.Option(
        1_000.0, "--min-balance", help="Exclude bank-quarters below this average balance."
    ),
    lags: str = typer.Option(
        "0,1,2,4", "--lags", help="Comma-separated quarter lags of each macro driver to include."
    ),
) -> None:
    """Joins the panel with macro drivers and their lags (no look-ahead),
    using winsorized NCO rates, excluding merger-flagged and
    below-min-balance bank-quarters -- see credit/macro.py's
    build_modeling_dataset for the exact rules."""
    panel = pd.read_parquet(panel_path)
    panel["quarter"] = pd.PeriodIndex(panel["quarter"].astype(str), freq="Q")

    history = pd.read_parquet(macro_history_path)
    history["quarter"] = pd.PeriodIndex(history["quarter"].astype(str), freq="Q")
    history = history.set_index("quarter")

    lag_values = tuple(int(x) for x in lags.split(","))
    dataset = macro.build_modeling_dataset(panel, history, lags=lag_values, min_balance=min_balance)
    output.parent.mkdir(parents=True, exist_ok=True)
    dataset.assign(quarter=lambda d: d["quarter"].astype(str)).to_parquet(output, index=False)
    typer.echo(f"wrote {len(dataset):,} rows to {output}")


@app.command("chart-macro")
def chart_macro(
    macro_history_path: Path = typer.Option(
        DEFAULT_MACRO_HISTORY_PATH, "--macro-history", help="Output of `fetch-macro`."
    ),
    scenario_dir: Path = typer.Option(
        DEFAULT_SCENARIO_DIR, "--scenario-dir", help="Output directory of `fetch-scenarios`."
    ),
    output_dir: Path = typer.Option(
        DEFAULT_CHARTS_DIR, "--output-dir", help="Where to write each variable's chart PNG."
    ),
) -> None:
    """One PNG per macro variable: its full FRED history, with the Fed's
    baseline and severely-adverse scenario paths appended, continuing
    from the last actual quarter."""
    history = pd.read_parquet(macro_history_path)
    history["quarter"] = pd.PeriodIndex(history["quarter"].astype(str), freq="Q")
    history = history.set_index("quarter")

    scenarios = {}
    for scenario_name in ("baseline", "severely_adverse"):
        path = scenario_dir / f"{scenario_name}.parquet"
        if not path.exists():
            typer.echo(
                f"{scenario_name}: no cached scenario at {path} -- run fetch-scenarios first",
                err=True,
            )
            continue
        scenario = pd.read_parquet(path)
        scenario["quarter"] = pd.PeriodIndex(scenario["quarter"].astype(str), freq="Q")
        scenarios[scenario_name] = scenario.set_index("quarter")

    output_dir.mkdir(parents=True, exist_ok=True)
    for variable in history.columns:
        safe_name = variable.replace(" ", "_").replace("/", "_")
        chart_path = output_dir / f"{safe_name}.png"
        charts.render_macro_variable_chart(history, scenarios, variable, str(chart_path))
        typer.echo(f"wrote {chart_path}")
