"""Pure transformations for macro/scenario data: FRED history aggregation
to quarterly, Fed scenario alignment with that history, and the joined
credit-panel + macro-driver modeling dataset with lagged macro drivers.
No function here touches the network -- fetching is sources/fred.py and
sources/fed_scenarios.py's job, orchestration is the CLI's.

Quarter-averaging and transform rules (both verified against real data,
not assumed -- see the empirical checks below):

- Every sub-quarterly (D/W/M) FRED series is aggregated to quarterly via
  the plain MEAN of its observations within each calendar quarter. An
  already-quarterly series (e.g. GDPC1) passes through this step
  unchanged (each quarter has exactly one observation, so its "mean" is
  itself). Verified against UNRATE: the mean of Jul/Aug/Sep 2025
  (4.3/4.3/4.4) is 4.33, matching the Fed's own published 2025Q3
  "Unemployment rate" actual (4.3) in 2026_Final_Historic_Domestic.csv.

- `transform="level"`: the quarterly mean IS the value, in the Fed's own
  units -- no further transform (used for rates: unemployment, Treasury/
  mortgage/prime/BBB yields; and for index levels: House Price Index,
  Commercial Real Estate Price Index, the Dow Jones Total Stock Market
  Index proxy, Market Volatility Index).

- `transform="qoq_annualized_pct_change"`: annualized quarter-over-quarter
  growth of the quarterly-mean level, COMPOUNDED (not simple x4):
  ((level[t] / level[t-1]) ** 4 - 1) * 100. This is a DIFFERENT
  convention from panel.annualized_nco_rate's simple x4 (that's the
  separate, correct convention for bank charge-off rates, not
  macroeconomic growth rates -- the two domains don't share a formula).
  Verified against CPIAUCSL and DSPI, cross-checked quarter by quarter
  against the Fed's own published "CPI inflation rate" and "Nominal
  disposable income growth" actuals in 2026_Final_Historic_Domestic.csv:
  e.g. 2023Q1 nominal disposable income growth computed via the compound
  formula from real DSPI data is 16.56%, matching the Fed's published
  16.6 almost exactly, while the simple x4 formula gives 15.56% -- a full
  point off. This pattern held consistently across 7+ independent
  quarters (2023Q1-2025Q3) for both series; the compound formula matches
  to within ~0.1 point every time, simple x4 is consistently ~1 point off
  for income and CPI alike. (Real GDP growth showed a smaller, noisier
  gap against today's GDPC1 vintage -- expected, since GDP/income data is
  revised repeatedly by BEA after first release, unlike CPI, so an exact
  match years later isn't expected regardless of formula; the CPI checks,
  which barely revise, are the more reliable signal and point the same
  way GDP does, just more cleanly.)
"""

from __future__ import annotations

import pandas as pd

from corefin.credit.sources.fred import FED_SCENARIO_VARIABLE_TO_FRED, FredSeriesMapping


def aggregate_fred_series_to_quarterly(observations: pd.DataFrame) -> pd.Series:
    """observations: columns ["date", "value"] as returned by
    fred.fetch_series (any native frequency -- D/W/M/Q). Returns a Series
    of quarterly MEANS, indexed by a pandas PeriodIndex (freq="Q"). An
    already-quarterly series passes through unchanged (one observation
    per quarter -> its mean is itself)."""
    quarters = pd.PeriodIndex(observations["date"], freq="Q")
    return observations.groupby(quarters)["value"].mean()


def apply_fred_transform(quarterly_level: pd.Series, transform: str) -> pd.Series:
    """quarterly_level: a quarterly-mean level Series (e.g. from
    `aggregate_fred_series_to_quarterly`), PeriodIndex(freq="Q"), sorted
    ascending. "level" passes through unchanged; "qoq_annualized_pct_change"
    returns the compounded annualized quarter-over-quarter growth rate in
    percent -- see the module docstring for why compounding (not simple
    x4) is the verified-correct convention here."""
    if transform == "level":
        return quarterly_level
    if transform == "qoq_annualized_pct_change":
        ratio = quarterly_level / quarterly_level.shift(1)
        return (ratio**4 - 1.0) * 100.0
    raise ValueError(f"unknown transform {transform!r}")


def build_fred_history_from_raw(
    raw_observations_by_variable: dict[str, pd.DataFrame],
    mappings: dict[str, FredSeriesMapping] | None = None,
) -> pd.DataFrame:
    """raw_observations_by_variable: {Fed scenario variable name ->
    fred.fetch_series(...) output} for every variable the caller fetched.
    `mappings` defaults to fred.FED_SCENARIO_VARIABLE_TO_FRED. Returns a
    wide DataFrame indexed by quarter (PeriodIndex, freq="Q", sorted
    ascending) with one column per variable, in the Fed's own units (same
    units the scenario CSVs use -- see `normalize_scenario`), ready to
    align directly against ingested scenario paths."""
    mappings = mappings if mappings is not None else FED_SCENARIO_VARIABLE_TO_FRED
    columns = {}
    for variable, observations in raw_observations_by_variable.items():
        mapping = mappings[variable]
        quarterly_level = aggregate_fred_series_to_quarterly(observations)
        columns[variable] = apply_fred_transform(quarterly_level, mapping.transform)
    history = pd.DataFrame(columns).sort_index()
    history.index.name = "quarter"
    return history


def normalize_scenario(raw_scenario: pd.DataFrame) -> pd.DataFrame:
    """raw_scenario: fed_scenarios.fetch_scenario(...) output, with the
    Fed's own "Date" strings ("2026 Q1") and "Scenario Name"/variable
    columns exactly as published. Returns a wide DataFrame indexed by
    quarter (PeriodIndex, freq="Q"), variable columns only (Fed's own
    units, unchanged -- these ARE the Fed's units already, no FRED
    unit conversion needed since the Fed defines these variables itself)."""
    quarters = pd.PeriodIndex(
        raw_scenario["Date"].str.replace(" ", "", regex=False), freq="Q"
    )
    variable_columns = [c for c in raw_scenario.columns if c not in ("Scenario Name", "Date")]
    normalized = raw_scenario[variable_columns].copy()
    normalized.index = quarters
    normalized.index.name = "quarter"
    return normalized.sort_index()


def assert_scenario_continues_from_history(history: pd.DataFrame, scenario: pd.DataFrame) -> None:
    """Raises ValueError unless `scenario`'s first quarter is exactly one
    quarter after `history`'s last quarter -- i.e. the scenario picks up
    immediately where the actual data ends, with no gap and no overlap.
    Both must be non-empty, quarter-indexed frames (as returned by
    `build_fred_history_from_raw` / `normalize_scenario`)."""
    if history.empty or scenario.empty:
        raise ValueError("both history and scenario must be non-empty to check continuity")
    last_actual = history.index.max()
    first_scenario = scenario.index.min()
    expected = last_actual + 1
    if first_scenario != expected:
        raise ValueError(
            f"scenario starts at {first_scenario}, expected {expected} "
            f"(one quarter after the last actual, {last_actual}) -- gap or overlap"
        )


def build_macro_chart_series(
    history: pd.DataFrame, scenarios: dict[str, pd.DataFrame], variable: str
) -> pd.DataFrame:
    """Long-format frame for charting one macro `variable`'s full history
    with each named scenario (e.g. {"baseline": ..., "severely_adverse":
    ...}) appended, continuing from the last actual quarter. Columns:
    "quarter" (Period), "series" ("actual" or a scenario name), "value".
    Does not itself validate continuity -- call
    `assert_scenario_continues_from_history` first."""
    frames = [
        pd.DataFrame(
            {"quarter": history.index, "series": "actual", "value": history[variable].to_numpy()}
        )
    ]
    for name, scenario in scenarios.items():
        frames.append(
            pd.DataFrame(
                {"quarter": scenario.index, "series": name, "value": scenario[variable].to_numpy()}
            )
        )
    return pd.concat(frames, ignore_index=True)


def build_modeling_dataset(
    panel: pd.DataFrame,
    macro_history: pd.DataFrame,
    lags: tuple[int, ...] = (0, 1, 2, 4),
    min_balance: float | None = 1_000.0,
    exclude_merger_flagged: bool = True,
) -> pd.DataFrame:
    """Joins `panel` (long-format, as produced by panel.build_panel) with
    `macro_history` (wide, quarter-indexed, as produced by
    `build_fred_history_from_raw`) for modeling.

    - Excludes merger-flagged bank-quarters (if `exclude_merger_flagged`)
      and bank-quarters below `min_balance` average balance (if given) --
      the same two exclusions `build_industry_nco_rate_report` applies,
      for the same reasons (a merger-contaminated quarterly flow, or a
      near-zero denominator dominated by noise).
    - Adds `winsorized_nco_rate` via panel.winsorize_nco_rates on the
      SCOPED (post-exclusion) rows, per-category, so winsorization
      bounds aren't distorted by the excluded rows.
    - For every macro `variable` and every `lag` in `lags`, adds a column
      `f"{variable}_lag{lag}"` = that variable's value `lag` quarters
      before the panel row's own quarter (lag=0 is the row's own
      quarter's contemporaneous value, known by the time that quarter's
      Call Report is filed -- NOT a look-ahead; only lag < 0, which this
      function never constructs, would be). This is a plain per-quarter
      lookup (macro_history has one row per quarter), so there is no
      possibility of a future quarter's macro value leaking into an
      earlier panel row -- see test_credit_macro.py's explicit
      no-look-ahead regression test.
    """
    from corefin.credit.panel import winsorize_nco_rates

    scoped = panel
    if exclude_merger_flagged and "merger_flag" in scoped.columns:
        scoped = scoped[~scoped["merger_flag"].fillna(False)]
    if min_balance is not None:
        scoped = scoped[scoped["average_balance"] >= min_balance]
    scoped = scoped.copy()

    scoped["winsorized_nco_rate"] = winsorize_nco_rates(scoped, min_balance=min_balance)

    for variable in macro_history.columns:
        lookup = macro_history[variable]
        for lag in lags:
            lagged_quarter = scoped["quarter"] - lag
            scoped[f"{variable}_lag{lag}"] = lagged_quarter.map(lookup)

    return scoped
