"""`corefin credit fetch` / `corefin credit build` -- the Credit-Loss
Forecasting Engine's data pipeline commands. Requires `corefin[fig]`.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import typer

from corefin.credit.panel import build_industry_nco_rate_report, build_panel, flag_chargeoff_gaps
from corefin.credit.schema import LoanCategory
from corefin.credit.sources import ffiec, ffiec_parse

app = typer.Typer(add_completion=False, help="Credit-Loss Forecasting Engine data pipeline.")

DEFAULT_RAW_DIR = Path("data/raw/ffiec")
DEFAULT_PANEL_PATH = Path("data/processed/credit_panel.parquet")
DEFAULT_COVERAGE_REPORT_PATH = Path("data/processed/coverage_report.csv")


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
