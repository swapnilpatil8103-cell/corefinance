"""`corefin credit fetch` / `corefin credit build` -- the Credit-Loss
Forecasting Engine's data pipeline commands. Requires `corefin[fig]`.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import typer

from corefin.credit import backtest, charts, industry_history, macro, models, projection
from corefin.credit.panel import build_industry_nco_rate_report, build_panel, flag_chargeoff_gaps
from corefin.credit.schema import LoanCategory
from corefin.credit.sources import fed_scenarios as fed_scenarios_source
from corefin.credit.sources import ffiec, ffiec_parse, fred

app = typer.Typer(add_completion=False, help="Credit-Loss Forecasting Engine data pipeline.")

DEFAULT_RAW_DIR = Path("data/raw/ffiec")
DEFAULT_PANEL_PATH = Path("data/processed/credit_panel.parquet")
DEFAULT_COVERAGE_REPORT_PATH = Path("data/processed/coverage_report.csv")
DEFAULT_FED_RAW_DIR = Path("data/raw/fed_scenarios")
DEFAULT_MACRO_RAW_DIR = Path("data/raw/fred")
DEFAULT_MACRO_HISTORY_PATH = Path("data/processed/macro_history.parquet")
DEFAULT_MACRO_HISTORY_FRED_PATH = Path("data/processed/macro_history_fred_extension.parquet")
DEFAULT_SCENARIO_DIR = Path("data/processed/scenarios")
DEFAULT_MODELING_DATASET_PATH = Path("data/processed/modeling_dataset.parquet")
DEFAULT_MODELING_DATASET_HOLDOUT_PATH = Path("data/processed/modeling_dataset_holdout.parquet")
DEFAULT_CHARTS_DIR = Path("data/processed/charts")
DEFAULT_TRAINING_JUMP_OFF_QUARTER = "2025Q4"
DEFAULT_MODELS_DIR = Path("data/processed/models")
DEFAULT_INDUSTRY_HISTORY_RAW_DIR = Path("data/raw/fred_industry")
DEFAULT_INDUSTRY_HISTORY_PATH = Path("data/processed/industry_history_fred.parquet")
DEFAULT_PROJECTIONS_DIR = Path("data/processed/projections")
DEFAULT_ALLOWANCE_ROLLFORWARD_PATH = Path("data/processed/allowance_rollforward.parquet")
FALLBACK_MODEL_FAMILY = projection.FALLBACK_MODEL_FAMILY


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


@app.command("fetch-fed-history")
def fetch_fed_history(
    vintage: int = typer.Option(
        fed_scenarios_source.CURRENT_SCENARIO_VINTAGE,
        "--vintage",
        help="Fed scenario vintage year.",
    ),
    raw_dir: Path = typer.Option(
        DEFAULT_FED_RAW_DIR, "--raw-dir", help="Gitignored directory to cache the raw pull in."
    ),
    output: Path = typer.Option(
        DEFAULT_MACRO_HISTORY_PATH,
        "--output",
        help="Where to write the quarterly macro history (wide, one column per variable).",
    ),
) -> None:
    """PRIMARY macro history source (see credit/macro.py's module
    docstring for why): fetches the Fed's own historic domestic actuals
    table for `vintage` (or reuses a cached raw pull), in the Fed's own
    definitions and units for all 16 variables, adds QoQ/YoY percent-
    change columns for the non-stationary level variables (see
    macro.LEVEL_VARIABLES_FOR_CHANGE_FEATURES), and writes the result to
    `output`. No key needed."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    cache_path = raw_dir / f"{vintage}_Final_Historic_Domestic.csv"
    if cache_path.exists():
        typer.echo(f"historic domestic ({vintage}): cached")
        raw = pd.read_csv(cache_path)
    else:
        typer.echo(f"historic domestic ({vintage}): fetching...")
        raw = fed_scenarios_source.fetch_historic_domestic(vintage=vintage)
        raw.to_csv(cache_path, index=False)

    history = macro.normalize_fed_historic(raw)
    history = macro.add_pct_change_features(history)
    output.parent.mkdir(parents=True, exist_ok=True)
    history.reset_index().assign(quarter=lambda d: d["quarter"].astype(str)).to_parquet(
        output, index=False
    )
    typer.echo(f"wrote {len(history):,} quarters x {len(history.columns)} columns to {output}")


@app.command("fetch-macro-fred")
def fetch_macro_fred(
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
        DEFAULT_MACRO_HISTORY_FRED_PATH,
        "--output",
        help="Where to write the quarterly FRED-derived history.",
    ),
) -> None:
    """OPTIONAL, secondary cross-reference only -- NOT used by
    `build-modeling-dataset`/`chart-macro` by default (see
    `fetch-fed-history`, the primary source, and credit/macro.py's module
    docstring for why: 4 of the 16 FRED proxies are on a different scale
    from the Fed's own definitions). Useful for sanity-checking the
    definition-exact variables (unemployment, GDP/income growth, CPI
    inflation, Treasury rates, mortgage/prime rate, VIX -- see
    fred.FED_SCENARIO_VARIABLE_TO_FRED's `is_exact_match` flags) against
    an independent source, or for extending beyond the Fed's own cached
    vintage before a newer one is published. Fetch every FRED series (or
    reuse a cached raw pull), aggregate to quarterly and apply each
    variable's own transform, and write the resulting wide history to
    `output`. Requires FRED_API_KEY."""
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


@app.command("fetch-industry-history")
def fetch_industry_history(
    start: str = typer.Option(
        "1985-01-01", "--start", help="First date to pull FRED industry series from."
    ),
    raw_dir: Path = typer.Option(
        DEFAULT_INDUSTRY_HISTORY_RAW_DIR,
        "--raw-dir",
        help="Gitignored directory to cache each raw FRED pull in.",
    ),
    output: Path = typer.Option(
        DEFAULT_INDUSTRY_HISTORY_PATH,
        "--output",
        help="Where to write the long-format industry charge-off/delinquency table.",
    ),
) -> None:
    """Fetches FRED's industry-wide charge-off and delinquency rate
    series (fred.INDUSTRY_CHARGEOFF_DELINQUENCY_SERIES -- one per loan
    category, verified live; home_equity deliberately has none) for every
    mapped category, or reuses a cached raw pull, and writes a long-format
    table (columns: category, series_type ["chargeoff"/"delinquency"],
    date, value_percent) to `output`. Values are left in the Fed's own
    PERCENT units here (not yet converted to this project's decimal-
    fraction scale) -- `fit-models` converts via industry_history.
    build_long_industry_rate when it loads this file. Requires
    FRED_API_KEY."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    seen_series_ids: set[str] = set()
    for category, mapping in fred.INDUSTRY_CHARGEOFF_DELINQUENCY_SERIES.items():
        for series_type, series_id in (
            ("chargeoff", mapping.chargeoff_series_id),
            ("delinquency", mapping.delinquency_series_id),
        ):
            if series_id is None:
                continue
            cache_path = raw_dir / f"{series_id}.csv"
            if cache_path.exists():
                if series_id not in seen_series_ids:
                    typer.echo(f"{series_id}: cached")
                observations = pd.read_csv(cache_path, parse_dates=["date"])
            else:
                typer.echo(f"{series_id}: fetching...")
                observations = fred.fetch_series(series_id, start_date=start)
                observations.to_csv(cache_path, index=False)
            seen_series_ids.add(series_id)
            for _, row in observations.iterrows():
                rows.append(
                    {
                        "category": category,
                        "series_type": series_type,
                        "date": row["date"],
                        "value_percent": row["value"],
                    }
                )

    table = pd.DataFrame(rows)
    output.parent.mkdir(parents=True, exist_ok=True)
    table.to_parquet(output, index=False)
    typer.echo(
        f"wrote {len(table):,} rows ({len(seen_series_ids)} distinct FRED series) to {output}"
    )


@app.command("fetch-scenarios")
def fetch_scenarios(
    vintage: int = typer.Option(
        fed_scenarios_source.CURRENT_SCENARIO_VINTAGE,
        "--vintage",
        help="Fed scenario vintage year.",
    ),
    raw_dir: Path = typer.Option(
        DEFAULT_FED_RAW_DIR, "--raw-dir", help="Gitignored directory to cache each raw pull in."
    ),
    macro_history_path: Path = typer.Option(
        DEFAULT_MACRO_HISTORY_PATH, "--macro-history", help="Output of `fetch-fed-history`."
    ),
    output_dir: Path = typer.Option(
        DEFAULT_SCENARIO_DIR, "--output-dir", help="Where to write each normalized scenario."
    ),
) -> None:
    """Fetch the Fed's baseline and severely-adverse scenarios for
    `vintage` (or reuse a cached raw pull), normalize them to the same
    quarter-indexed/variable-name shape `fetch-fed-history` produces, and
    verify each starts exactly one quarter after the macro history's last
    actual quarter -- the Fed's convention: history through the jump-off
    quarter, then every scenario quarter from the next one on, with no
    truncation (this holds by construction now that both come from the
    same Fed file family; raises if it somehow doesn't). Adds QoQ/YoY
    percent-change columns for the non-stationary level variables,
    computed against the REAL levels leading into the jump-off quarter
    (via macro.build_full_macro_path), not the scenario's own 13 rows in
    isolation. Writes both to `output_dir`."""
    history = pd.read_parquet(macro_history_path)
    history["quarter"] = pd.PeriodIndex(history["quarter"].astype(str), freq="Q")
    history = history.set_index("quarter")
    # levels only (drop this run's own QoQ/YoY columns) so build_full_macro_path's
    # continuity check compares the same 16 raw variables the scenario has.
    history_levels = history[[c for c in history.columns if "% change" not in c]]

    raw_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    for scenario_name in ("baseline", "severely_adverse"):
        cache_path = raw_dir / f"{vintage}_Final_{scenario_name}.csv"
        if cache_path.exists():
            typer.echo(f"{scenario_name} ({vintage}): cached")
            raw = pd.read_csv(cache_path)
        else:
            typer.echo(f"{scenario_name} ({vintage}): fetching...")
            raw = fed_scenarios_source.fetch_scenario(scenario_name, vintage=vintage)
            raw.to_csv(cache_path, index=False)

        normalized = macro.normalize_scenario(raw)
        full_path = macro.build_full_macro_path(history_levels, normalized)
        full_path_with_changes = macro.add_pct_change_features(full_path)
        scenario_with_changes = full_path_with_changes.loc[normalized.index]

        path = output_dir / f"{scenario_name}.parquet"
        scenario_with_changes.reset_index().assign(
            quarter=lambda d: d["quarter"].astype(str)
        ).to_parquet(path, index=False)
        typer.echo(
            f"{scenario_name}: continuity OK, wrote {len(scenario_with_changes)} quarters to {path}"
        )


@app.command("build-modeling-dataset")
def build_modeling_dataset_cmd(
    panel_path: Path = typer.Option(
        DEFAULT_PANEL_PATH, "--panel", help="Parquet panel from `build`."
    ),
    macro_history_path: Path = typer.Option(
        DEFAULT_MACRO_HISTORY_PATH, "--macro-history", help="Output of `fetch-fed-history`."
    ),
    output: Path = typer.Option(
        DEFAULT_MODELING_DATASET_PATH,
        "--output",
        help="Where to write the TRAINING modeling dataset.",
    ),
    holdout_output: Path = typer.Option(
        DEFAULT_MODELING_DATASET_HOLDOUT_PATH,
        "--holdout-output",
        help="Where to write the held-out (post-jump-off) modeling dataset.",
    ),
    max_training_quarter: str = typer.Option(
        DEFAULT_TRAINING_JUMP_OFF_QUARTER,
        "--max-training-quarter",
        help="Last quarter included in training; later panel quarters are held out as an "
        "out-of-sample check, not trained on.",
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
    build_modeling_dataset for the exact rules. Panel quarters after
    `max_training_quarter` (the Fed historic table's own jump-off
    quarter, 2025Q4 by default) are written separately to `holdout_output`
    rather than included in `output` -- the Fed's own macro history has no
    actuals past that quarter anyway, so any bank-quarter lag beyond it
    would otherwise silently pull in NaN or (once scenario paths are used
    for projection) a hypothetical value instead of a real one."""
    panel = pd.read_parquet(panel_path)
    panel["quarter"] = pd.PeriodIndex(panel["quarter"].astype(str), freq="Q")

    history = pd.read_parquet(macro_history_path)
    history["quarter"] = pd.PeriodIndex(history["quarter"].astype(str), freq="Q")
    history = history.set_index("quarter")

    lag_values = tuple(int(x) for x in lags.split(","))
    dataset = macro.build_modeling_dataset(panel, history, lags=lag_values, min_balance=min_balance)

    cutoff = pd.Period(max_training_quarter, freq="Q")
    train, holdout = macro.split_dataset_by_quarter(dataset, cutoff)

    output.parent.mkdir(parents=True, exist_ok=True)
    train.assign(quarter=lambda d: d["quarter"].astype(str)).to_parquet(output, index=False)
    holdout_output.parent.mkdir(parents=True, exist_ok=True)
    holdout.assign(quarter=lambda d: d["quarter"].astype(str)).to_parquet(
        holdout_output, index=False
    )
    typer.echo(f"wrote {len(train):,} training rows (<= {cutoff}) to {output}")
    typer.echo(f"wrote {len(holdout):,} held-out rows (> {cutoff}) to {holdout_output}")


@app.command("chart-macro")
def chart_macro(
    macro_history_path: Path = typer.Option(
        DEFAULT_MACRO_HISTORY_PATH, "--macro-history", help="Output of `fetch-fed-history`."
    ),
    scenario_dir: Path = typer.Option(
        DEFAULT_SCENARIO_DIR, "--scenario-dir", help="Output directory of `fetch-scenarios`."
    ),
    output_dir: Path = typer.Option(
        DEFAULT_CHARTS_DIR, "--output-dir", help="Where to write each variable's chart PNG."
    ),
) -> None:
    """One PNG per macro variable (including the QoQ/YoY change columns
    `fetch-fed-history`/`fetch-scenarios` add): its full history from the
    Fed's own historic domestic actuals table, with the Fed's baseline and
    severely-adverse scenario paths appended, continuing from the jump-off
    quarter."""
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
        safe_name = variable.replace(" ", "_").replace("/", "_").replace("%", "pct")
        chart_path = output_dir / f"{safe_name}.png"
        charts.render_macro_variable_chart(history, scenarios, variable, str(chart_path))
        typer.echo(f"wrote {chart_path}")


def _record_family_result(
    category: str,
    dependent: str,
    covid_spec: str,
    window: str,
    family: str,
    result: dict,
    backtest_rows: list[dict],
    coefficient_rows: list[dict],
    importance_rows: list[dict],
    forecast_store: dict,
) -> None:
    backtest_rows.append(
        {
            "category": category,
            "dependent": dependent,
            "covid_spec": covid_spec,
            "window": window,
            "model_family": family,
            **result["metrics"],
        }
    )
    if "coefficients" in result:
        t_values = result.get("t_values", {})
        classification = models.classify_coefficient_significance(result["coefficients"], t_values)
        for feature, coefficient in result["coefficients"].items():
            coefficient_rows.append(
                {
                    "category": category,
                    "dependent": dependent,
                    "covid_spec": covid_spec,
                    "window": window,
                    "model_family": family,
                    "feature": feature,
                    "coefficient": coefficient,
                    "t_value": t_values.get(feature),
                    "sign_classification": classification.get(feature),
                }
            )
    if "feature_importances" in result:
        for feature, importance in result["feature_importances"].items():
            importance_rows.append(
                {
                    "category": category,
                    "dependent": dependent,
                    "covid_spec": covid_spec,
                    "window": window,
                    "feature": feature,
                    "importance": importance,
                }
            )
    forecast_store[(category, dependent, covid_spec, window)] = forecast_store.get(
        (category, dependent, covid_spec, window), {}
    )
    forecast_store[(category, dependent, covid_spec, window)][family] = result["forecast"]


_DEPENDENT_TO_INDUSTRY_SERIES_TYPE = {"nco_rate": "chargeoff", "npl_ratio": "delinquency"}


def _load_long_history_frames(
    industry_history_path: Path, macro_history: pd.DataFrame
) -> dict[tuple[str, str], pd.DataFrame]:
    """Returns {(category, dependent_label) -> long-history frame} for
    every category/series_type fred.INDUSTRY_CHARGEOFF_DELINQUENCY_SERIES
    covers -- built once so `fit_models` doesn't reparse the raw table
    per category/dependent/covid_spec/window."""
    if not industry_history_path.exists():
        return {}
    raw_table = pd.read_parquet(industry_history_path)
    raw_table["date"] = pd.to_datetime(raw_table["date"])
    frames: dict[tuple[str, str], pd.DataFrame] = {}
    for (category, series_type), group in raw_table.groupby(["category", "series_type"]):
        dep_label = next(
            (k for k, v in _DEPENDENT_TO_INDUSTRY_SERIES_TYPE.items() if v == series_type), None
        )
        if dep_label is None:
            continue
        observations = group.rename(columns={"value_percent": "value"})[["date", "value"]]
        long_rate = industry_history.build_long_industry_rate(observations)
        frames[(category, dep_label)] = industry_history.build_long_industry_frame(
            long_rate, macro_history
        )
    return frames


@app.command("fit-models")
def fit_models(
    panel_path: Path = typer.Option(
        DEFAULT_MODELING_DATASET_PATH,
        "--panel",
        help="Training modeling dataset (quarter <= 2025Q4).",
    ),
    holdout_path: Path = typer.Option(
        DEFAULT_MODELING_DATASET_HOLDOUT_PATH,
        "--holdout",
        help="Held-out modeling dataset (2026Q1+).",
    ),
    macro_history_path: Path = typer.Option(
        DEFAULT_MACRO_HISTORY_PATH, "--macro-history", help="Output of `fetch-fed-history`."
    ),
    industry_history_path: Path = typer.Option(
        DEFAULT_INDUSTRY_HISTORY_PATH,
        "--industry-history",
        help="Output of `fetch-industry-history` (optional -- if absent, only the "
        "Call-Report-based aggregate_ar/panel_fe/gbm families run, same as before).",
    ),
    output_dir: Path = typer.Option(
        DEFAULT_MODELS_DIR, "--output-dir", help="Where to write backtest/coefficient/chart output."
    ),
    categories: str | None = typer.Option(
        None,
        "--categories",
        help="Comma-separated LoanCategory values to run (default: every category "
        "backtest.MODELING_CATEGORIES lists).",
    ),
) -> None:
    """Stage 4: for every loan category, dependent variable (NCO rate,
    NPL ratio), and COVID specification (main, robustness), fits every
    model family (credit/models.py) over both required out-of-time
    validation windows (2007-2010, 2020-2021) and the 2026Q1-Q2 holdout.
    When `industry_history_path` exists, ALSO fits aggregate_long (the
    same aggregate model trained on FRED's much longer industry history,
    back to 1991Q1 for most categories) and anchored_to_aggregate (bank-
    level projections anchored to that long-history forecast), and writes
    industry_history_comparison.csv checking the FRED-derived and Call-
    Report-derived industry rates agree over their 2001-2025 overlap.
    Writes backtest_table.csv, coefficient_table.csv (t-values and each
    core macro feature's 3-way sign classification), gbm_feature_
    importances.csv, and one actual-vs-predicted PNG per category/
    dependent for the main-spec 2007-2010 window to `output_dir`."""
    training = pd.read_parquet(panel_path)
    training["quarter"] = pd.PeriodIndex(training["quarter"].astype(str), freq="Q")
    training = backtest.add_npl_ratio_columns(training)

    holdout = pd.read_parquet(holdout_path)
    holdout["quarter"] = pd.PeriodIndex(holdout["quarter"].astype(str), freq="Q")
    holdout = backtest.add_npl_ratio_columns(holdout)

    macro_history = pd.read_parquet(macro_history_path)
    macro_history["quarter"] = pd.PeriodIndex(macro_history["quarter"].astype(str), freq="Q")
    macro_history = macro_history.set_index("quarter")
    long_history_frames = _load_long_history_frames(industry_history_path, macro_history)
    if long_history_frames:
        typer.echo(
            f"loaded long-history industry frames for {len(long_history_frames)} "
            "category/dependent combinations"
        )
    else:
        typer.echo(
            f"no industry history at {industry_history_path} -- run fetch-industry-history "
            "first for the aggregate_long/anchored_to_aggregate families",
            err=True,
        )

    selected_categories = (
        [LoanCategory(c.strip()) for c in categories.split(",")]
        if categories
        else list(backtest.MODELING_CATEGORIES)
    )

    backtest_rows: list[dict] = []
    coefficient_rows: list[dict] = []
    importance_rows: list[dict] = []
    comparison_rows: list[dict] = []
    forecast_store: dict = {}

    for category in selected_categories:
        category_train = training[training["category"] == category]
        category_holdout = holdout[holdout["category"] == category]
        if category_train.empty:
            typer.echo(f"{category}: no training rows -- skipping", err=True)
            continue
        typer.echo(f"=== {category} ({len(category_train):,} training rows) ===")

        for dep_label, dep_column in backtest.DEPENDENT_VARIABLES.items():
            long_history_frame = long_history_frames.get((category, dep_label))
            if long_history_frame is not None:
                call_report_frame = models.build_industry_series(category_train, dep_column).frame
                comparison = industry_history.compare_with_call_report_aggregate(
                    long_history_frame, call_report_frame
                )
                for _, row in comparison.iterrows():
                    comparison_rows.append(
                        {
                            "category": category,
                            "dependent": dep_label,
                            "quarter": str(row["quarter"]),
                            "fred_rate": row["fred_rate"],
                            "call_report_rate": row["call_report_rate"],
                            "difference": row["difference"],
                        }
                    )

            for covid_spec in backtest.COVID_SPECS:
                typer.echo(f"  {dep_label} / {covid_spec}: backtesting...")
                window_results = backtest.run_category_backtests(
                    category_train,
                    category,
                    dep_column,
                    covid_spec,
                    long_history_frame=long_history_frame,
                )
                for window_label, family_results in window_results.items():
                    for family, result in family_results.items():
                        _record_family_result(
                            category,
                            dep_label,
                            covid_spec,
                            window_label,
                            family,
                            result,
                            backtest_rows,
                            coefficient_rows,
                            importance_rows,
                            forecast_store,
                        )

                if not category_holdout.empty:
                    typer.echo(f"  {dep_label} / {covid_spec}: scoring 2026 holdout...")
                    holdout_results = backtest.run_holdout_backtest(
                        category_train,
                        category_holdout,
                        category,
                        dep_column,
                        covid_spec,
                        long_history_frame=long_history_frame,
                    )
                    if holdout_results is None:
                        typer.echo(
                            f"  {dep_label} / {covid_spec}: not enough training data -- "
                            "skipping 2026 holdout",
                            err=True,
                        )
                        continue
                    for family, result in holdout_results.items():
                        _record_family_result(
                            category,
                            dep_label,
                            covid_spec,
                            "2026_holdout",
                            family,
                            result,
                            backtest_rows,
                            coefficient_rows,
                            importance_rows,
                            forecast_store,
                        )

    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(backtest_rows).to_csv(output_dir / "backtest_table.csv", index=False)
    pd.DataFrame(coefficient_rows).to_csv(output_dir / "coefficient_table.csv", index=False)
    pd.DataFrame(importance_rows).to_csv(output_dir / "gbm_feature_importances.csv", index=False)
    typer.echo(
        f"wrote backtest_table.csv / coefficient_table.csv / "
        f"gbm_feature_importances.csv to {output_dir}"
    )
    if comparison_rows:
        comparison_path = output_dir / "industry_history_comparison.csv"
        pd.DataFrame(comparison_rows).to_csv(comparison_path, index=False)
        typer.echo(f"wrote {comparison_path}")

    backtest_table = pd.DataFrame(backtest_rows)
    summary = backtest.build_backtest_summary(backtest_table)
    summary_path = output_dir / "backtest_summary.csv"
    summary.to_csv(summary_path, index=False)
    typer.echo(f"wrote {summary_path}")

    charts_dir = output_dir / "charts"
    charts_dir.mkdir(parents=True, exist_ok=True)
    grid_entries: list[tuple[str, pd.Series, dict]] = []
    for (category, dep_label, covid_spec, window_label), family_forecasts in forecast_store.items():
        if window_label != "2007-2010" or covid_spec != "main":
            continue
        category_train = training[training["category"] == category]
        dep_column = backtest.DEPENDENT_VARIABLES[dep_label]
        actual_industry = models.build_industry_series(category_train, dep_column).frame
        actual_series = actual_industry.set_index("quarter")["industry_rate"]
        title = f"{category} -- {dep_label} (2007-2010 backtest)"
        chart_path = charts_dir / f"{category}_{dep_label}_2007-2010.png"
        charts.render_backtest_chart(
            actual_series,
            family_forecasts,
            title=title,
            y_label=dep_label,
            path=str(chart_path),
        )
        typer.echo(f"wrote {chart_path}")
        grid_entries.append((title, actual_series, family_forecasts))

    if grid_entries:
        grid_path = output_dir / "backtest_chart_grid_2007-2010.png"
        charts.render_backtest_chart_grid(grid_entries, str(grid_path))
        typer.echo(f"wrote {grid_path}")


def _format_coefficients_for_report(candidate: projection.CandidateModel) -> str:
    """A compact "feature=value (classification)" string per core macro
    feature for the model-selection report -- "no coefficients (family)"
    for gbm, which has none."""
    if candidate.coefficients is None:
        return f"no coefficients ({candidate.family})"
    classification = candidate.sign_classification or {}
    parts = []
    for feature in projection.CORE_SIGN_CHECK_FEATURES:
        column = models.feature_column(feature)
        if column not in candidate.coefficients:
            continue
        parts.append(
            f"{feature}={candidate.coefficients[column]:.5f} "
            f"({classification.get(column, 'n/a')})"
        )
    return "; ".join(parts)


@app.command("project")
def project_cmd(
    panel_path: Path = typer.Option(
        DEFAULT_MODELING_DATASET_PATH,
        "--panel",
        help="Training modeling dataset (quarter <= 2025Q4) -- the FULL sample Stage 5 fits on.",
    ),
    macro_history_path: Path = typer.Option(
        DEFAULT_MACRO_HISTORY_PATH, "--macro-history", help="Output of `fetch-fed-history`."
    ),
    industry_history_path: Path = typer.Option(
        DEFAULT_INDUSTRY_HISTORY_PATH,
        "--industry-history",
        help="Output of `fetch-industry-history` (optional -- categories whose best "
        "model needs it are skipped if absent).",
    ),
    scenario_dir: Path = typer.Option(
        DEFAULT_SCENARIO_DIR, "--scenario-dir", help="Output directory of `fetch-scenarios`."
    ),
    models_dir: Path = typer.Option(
        DEFAULT_MODELS_DIR,
        "--models-dir",
        help="Output directory of `fit-models` (backtest_table.csv).",
    ),
    credit_panel_path: Path = typer.Option(
        DEFAULT_PANEL_PATH,
        "--credit-panel",
        help="Parquet panel from `build` -- used only for the real (Call-Report-wide) "
        "total loan balance in the jump-off allowance calibration check.",
    ),
    allowance_path: Path = typer.Option(
        DEFAULT_ALLOWANCE_ROLLFORWARD_PATH,
        "--allowance-rollforward",
        help="Output of panel.compute_bank_allowance_rollforward (bank-TOTAL, not "
        "per-category) -- used for the real total allowance in the calibration check.",
    ),
    output_dir: Path = typer.Option(
        DEFAULT_PROJECTIONS_DIR, "--output-dir", help="Where to write the projection tables/charts."
    ),
    categories: str | None = typer.Option(
        None,
        "--categories",
        help="Comma-separated LoanCategory values to run (default: every category "
        "backtest.MODELING_CATEGORIES lists).",
    ),
) -> None:
    """Stage 5: for every loan category, selects the projection model
    (`projection.select_projection_model` -- best-backtest RMSE among the
    full-sample-clean candidates, excluding raw aggregate_long for
    `projection.PROXY_CATEGORIES_REQUIRING_ANCHOR`), fits it on the FULL
    sample through 2025Q4, projects the NCO rate forward under both Fed
    scenarios (baseline, severely_adverse), and builds the simplified
    CECL allowance/provision roll-forward (credit/projection.py) from
    each projected path. Writes projection_table.csv (one row per
    category/scenario/quarter), model_selection.csv (the selected family
    and every candidate's core-coefficient sign classification per
    category), calibration_report.csv (the jump-off allowance ratio vs.
    the real industry ratio), and one NCO-rate chart per category."""
    training = pd.read_parquet(panel_path)
    training["quarter"] = pd.PeriodIndex(training["quarter"].astype(str), freq="Q")

    macro_history = pd.read_parquet(macro_history_path)
    macro_history["quarter"] = pd.PeriodIndex(macro_history["quarter"].astype(str), freq="Q")
    macro_history = macro_history.set_index("quarter")
    jump_off_quarter = macro_history.index.max()

    backtest_table_path = models_dir / "backtest_table.csv"
    if not backtest_table_path.exists():
        typer.echo(f"no backtest table at {backtest_table_path} -- run fit-models first", err=True)
        raise typer.Exit(code=1)
    backtest_table = pd.read_csv(backtest_table_path)

    long_history_frames = _load_long_history_frames(industry_history_path, macro_history)

    scenarios: dict[str, pd.DataFrame] = {}
    for scenario_name in ("baseline", "severely_adverse"):
        path = scenario_dir / f"{scenario_name}.parquet"
        if not path.exists():
            typer.echo(f"no cached scenario at {path} -- run fetch-scenarios first", err=True)
            raise typer.Exit(code=1)
        scenario = pd.read_parquet(path)
        scenario["quarter"] = pd.PeriodIndex(scenario["quarter"].astype(str), freq="Q")
        scenarios[scenario_name] = scenario.set_index("quarter")

    selected_categories = (
        [LoanCategory(c.strip()) for c in categories.split(",")]
        if categories
        else list(backtest.MODELING_CATEGORIES)
    )

    projection_rows: list[dict] = []
    selection_rows: list[dict] = []
    jump_off_allowance_by_category: dict[str, float] = {}
    starting_balance_by_category: dict[str, float] = {}
    chart_dir = output_dir / "charts"
    chart_dir.mkdir(parents=True, exist_ok=True)

    for category in selected_categories:
        category_train = training[training["category"] == category]
        if category_train.empty:
            typer.echo(f"{category}: no training rows -- skipping", err=True)
            continue

        long_history_frame = long_history_frames.get((category, "nco_rate"))
        selection = projection.select_projection_model(
            category, category_train, long_history_frame, backtest_table
        )
        best_model = selection.selected_family
        if best_model in projection.FAMILIES_REQUIRING_LONG_HISTORY and long_history_frame is None:
            typer.echo(
                f"{category}: selected model {best_model} needs long-history data that "
                "isn't cached -- falling back to aggregate_ar",
                err=True,
            )
            best_model = FALLBACK_MODEL_FAMILY
        clean_note = "" if selection.is_clean else " (NO CLEAN CANDIDATE -- best RMSE regardless)"
        typer.echo(f"=== {category}: projecting with {best_model}{clean_note} ===")
        for family, candidate in selection.candidates.items():
            selection_rows.append(
                {
                    "category": category,
                    "selected": family == selection.selected_family,
                    "model_family": family,
                    "rmse": candidate.rmse,
                    "clean": candidate.clean,
                    "coefficients": _format_coefficients_for_report(candidate),
                }
            )
            typer.echo(
                f"    {'*' if family == selection.selected_family else ' '} {family}: "
                f"rmse={candidate.rmse} clean={candidate.clean} -- "
                f"{_format_coefficients_for_report(candidate)}"
            )

        realized_series = models.build_industry_series(category_train, "winsorized_nco_rate").frame
        realized_rate = realized_series.set_index("quarter")["industry_rate"]
        last_quarter = category_train["quarter"].max()
        starting_balance = category_train[category_train["quarter"] == last_quarter][
            "average_balance"
        ].sum()
        starting_balance_by_category[str(category)] = starting_balance

        scenario_forecasts: dict[str, pd.Series] = {}
        for scenario_name, scenario in scenarios.items():
            forecast = projection.project_category_nco_rate(
                category, best_model, category_train, macro_history, scenario, long_history_frame
            )
            scenario_forecasts[scenario_name] = forecast
            cecl = projection.build_cecl_projection(
                realized_rate, forecast, category, starting_balance
            )
            jump_off_allowance_by_category[str(category)] = cecl.iloc[0]["allowance_required"]
            nine_quarter_loss = projection.cumulative_loss_rate(
                forecast, projection.FED_COMPARISON_QUARTERS
            )
            for quarter, row in cecl.iterrows():
                projection_rows.append(
                    {
                        "category": category,
                        "scenario": scenario_name,
                        "best_model": best_model,
                        "quarter": str(quarter),
                        "projected_nco_rate": forecast.get(quarter),
                        "lifetime_expected_loss_rate": row["lifetime_expected_loss_rate"],
                        "allowance_required": row["allowance_required"],
                        "net_charge_off": row["net_charge_off"],
                        "provision_expense": row["provision_expense"],
                        "cumulative_9q_loss_rate": nine_quarter_loss,
                    }
                )

        chart_path = chart_dir / f"{category}_projection.png"
        charts.render_backtest_chart(
            realized_rate,
            scenario_forecasts,
            title=f"{category} -- NCO rate projection ({best_model})",
            y_label="nco_rate",
            path=str(chart_path),
        )
        typer.echo(f"wrote {chart_path}")

    output_dir.mkdir(parents=True, exist_ok=True)
    table_path = output_dir / "projection_table.csv"
    pd.DataFrame(projection_rows).to_csv(table_path, index=False)
    typer.echo(f"wrote {table_path}")

    selection_path = output_dir / "model_selection.csv"
    pd.DataFrame(selection_rows).to_csv(selection_path, index=False)
    typer.echo(f"wrote {selection_path}")

    if credit_panel_path.exists() and allowance_path.exists():
        credit_panel = pd.read_parquet(credit_panel_path)
        credit_panel["quarter"] = pd.PeriodIndex(credit_panel["quarter"].astype(str), freq="Q")
        real_loans_quarter = credit_panel[
            (credit_panel["quarter"] == jump_off_quarter)
            & (credit_panel["category"].isin(starting_balance_by_category.keys()))
        ]
        real_total_loans = real_loans_quarter["average_balance"].sum()

        allowance_rollforward = pd.read_parquet(allowance_path)
        real_total_allowance = allowance_rollforward[
            allowance_rollforward["quarter"] == jump_off_quarter
        ]["allowance_balance"].sum()

        calibration = projection.compute_calibration_gap(
            jump_off_allowance_by_category,
            starting_balance_by_category,
            real_total_allowance,
            real_total_loans,
        )
        typer.echo(
            f"\ncalibration check at {jump_off_quarter} (jump-off allowance as % of loans):\n"
            f"  this engine: {calibration.model_total_allowance:,.0f} / "
            f"{calibration.model_total_loans:,.0f} = {calibration.model_allowance_ratio:.4%}\n"
            f"  real industry (Call Report panel): {calibration.real_total_allowance:,.0f} / "
            f"{calibration.real_total_loans:,.0f} = {calibration.real_allowance_ratio:.4%}\n"
            f"  gap: {calibration.gap:+.4%} (note: the real allowance figure is each bank's "
            "TOTAL allowance across ALL loan types, not just these categories -- see "
            "CalibrationCheck's docstring)"
        )
        calibration_path = output_dir / "calibration_report.csv"
        pd.DataFrame(
            [
                {
                    "jump_off_quarter": str(jump_off_quarter),
                    "model_total_allowance": calibration.model_total_allowance,
                    "model_total_loans": calibration.model_total_loans,
                    "model_allowance_ratio": calibration.model_allowance_ratio,
                    "real_total_allowance": calibration.real_total_allowance,
                    "real_total_loans": calibration.real_total_loans,
                    "real_allowance_ratio": calibration.real_allowance_ratio,
                    "gap": calibration.gap,
                }
            ]
        ).to_csv(calibration_path, index=False)
        typer.echo(f"wrote {calibration_path}")
    else:
        typer.echo(
            f"skipping calibration check -- {credit_panel_path} or {allowance_path} not found",
            err=True,
        )
