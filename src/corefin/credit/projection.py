"""Stage 5: the projection engine -- forecasts each loan category's NCO
rate forward under the Fed's baseline and severely-adverse scenarios,
using the model Stage 4's own backtest found best for that category, fit
on the FULL sample through 2025Q4 (the training window used everywhere
else in this project is 2001Q1-2025Q4; Stage 5 is the one place that
full-sample fit is actually used for something -- a genuine, unknown-
outcome projection, not a backtest against an already-known history).
Then computes a SIMPLIFIED CECL allowance and quarterly provision
roll-forward from that projected NCO path. No function here touches the
network or reads/writes files -- see cli.py for orchestration.

SIMPLIFIED CECL, AND WHAT "SIMPLIFIED" MEANS HERE: real CECL (ASC 326)
estimates the full LIFETIME expected credit loss for each loan pool,
typically using cohort/vintage loss curves built from each bank's own
loan-level maturity, prepayment and amortization data discounted to
present value, revisited every period. This project has none of that
(no loan-level data at all -- only bank-quarter aggregates). The
simplification: approximate "lifetime expected loss rate" as the average
projected ANNUALIZED NCO rate over a fixed, documented WEIGHTED-AVERAGE-
LIFE window per category (`CATEGORY_WEIGHTED_AVERAGE_LIFE_QUARTERS`) --
industry rule-of-thumb figures (not derived from this project's own
data, since there's no loan-level maturity data to derive them from),
clearly flagged as an ASSUMPTION a real implementation would replace with
a bank's own prepayment-adjusted duration estimates. The BALANCE used
throughout is held STATIC at each category's last actual (2025Q4)
level -- the same "static balance sheet" convention real regulatory
stress tests (DFAST/CCAR) use for exactly this reason: it isolates the
loss-rate projection from a separate, unmodeled balance-growth forecast.
"""

from __future__ import annotations

import pandas as pd

from corefin.credit import macro, models

# Model families that need the FRED-derived long-history aggregate frame
# (industry_history.build_long_industry_frame's output) rather than the
# Call-Report-only one: aggregate_long fits directly on it; anchored_to_
# aggregate anchors bank-level projections to ITS forecast.
FAMILIES_REQUIRING_LONG_HISTORY = ("aggregate_long", "anchored_to_aggregate")

# Weighted-average-life, in QUARTERS, per category -- industry rule-of-
# thumb figures for the "lifetime" horizon the simplified CECL allowance
# averages projected losses over (see module docstring: NOT derived from
# this project's own data). Revolving/short-duration books (credit card,
# auto, other consumer, CRE construction) get short horizons; long-
# duration term/mortgage books (CRE multifamily/nonfarm-nonresidential,
# residential mortgage, home equity) get longer ones reflecting typical
# prepayment-adjusted effective life, not contractual maturity (e.g. a
# 30-year mortgage's WAL after prepayment/refinancing is conventionally
# much shorter than 30 years).
CATEGORY_WEIGHTED_AVERAGE_LIFE_QUARTERS: dict[str, int] = {
    "commercial_and_industrial": 14,  # ~3.5 years: revolvers/term loans
    "cre_construction": 10,  # ~2.5 years: short-term, converts or pays off
    "cre_multifamily": 32,  # ~8 years: term mortgage-style loans
    "cre_nonfarm_nonresidential": 32,  # ~8 years: term mortgage-style loans
    "residential_mortgage": 30,  # ~7.5 years: post-prepayment effective life
    "home_equity": 24,  # ~6 years: revolving/term mix
    "credit_card": 8,  # ~2 years: revolving, short expected life
    "auto": 11,  # ~2.75 years: fixed term, but prepays
    "other_consumer": 10,  # ~2.5 years: similar to auto/personal loans
}


def lifetime_expected_loss_rate(
    annualized_nco_rate_path: pd.Series, quarter: pd.Period, wal_quarters: int
) -> float:
    """The simplified "lifetime expected loss rate" as of `quarter`: the
    average of `annualized_nco_rate_path` over the `wal_quarters`
    quarters starting at `quarter` (inclusive). If the path doesn't reach
    that far forward (either because the projection horizon ends first,
    or because `quarter` is the historical jump-off itself and only past
    data exists), the LAST available quarter's rate is held constant to
    fill the remainder -- a simple, documented terminal-value convention,
    not a real amortization schedule. Raises if `quarter` itself isn't in
    `annualized_nco_rate_path` at all."""
    if quarter not in annualized_nco_rate_path.index:
        raise ValueError(f"{quarter} is not in the provided rate path")
    window_quarters = [quarter + i for i in range(wal_quarters)]
    available = annualized_nco_rate_path.reindex(window_quarters)
    if available.notna().any():
        available = available.ffill()
    return float(available.mean())


def backward_looking_lifetime_expected_loss_rate(
    realized_annualized_nco_rate: pd.Series, as_of_quarter: pd.Period, wal_quarters: int
) -> float:
    """The simplified "lifetime expected loss rate" as of a REALIZED
    (non-projected) quarter, e.g. the 2025Q4 jump-off -- the average of
    `realized_annualized_nco_rate` over the `wal_quarters` quarters
    ENDING at `as_of_quarter` (backward-looking, since there is no
    forward projection to average yet at the jump-off itself). Used only
    to establish a baseline allowance level to roll the projection's
    provision forward from -- not a claim about what the real bank
    actually held as its allowance that quarter (this project has no
    per-category real allowance data -- see panel.py's allowance
    functions, which are bank-TOTAL, not split by category)."""
    window_quarters = [as_of_quarter - i for i in range(wal_quarters)]
    available = realized_annualized_nco_rate.reindex(window_quarters).dropna()
    if available.empty:
        raise ValueError(f"no realized data available on or before {as_of_quarter}")
    return float(available.mean())


def build_cecl_projection(
    realized_annualized_nco_rate: pd.Series,
    projected_annualized_nco_rate: pd.Series,
    category: str,
    starting_balance: float,
) -> pd.DataFrame:
    """The simplified CECL allowance/provision roll-forward for one
    category/scenario. `realized_annualized_nco_rate`: the real, actual
    annualized NCO rate history (quarter-indexed), used only to establish
    the jump-off quarter's baseline allowance level (via
    `backward_looking_lifetime_expected_loss_rate`). `projected_
    annualized_nco_rate`: the projected path (quarter-indexed) for the
    scenario/model being evaluated, covering the projection horizon
    ONLY (e.g. 2026Q1-2029Q1) -- must NOT include the jump-off quarter
    itself. `starting_balance`: the category's static (held-constant)
    balance, e.g. its last actual (jump-off) average balance.

    Returns a frame indexed by quarter (the jump-off quarter plus every
    projected quarter) with columns: "lifetime_expected_loss_rate",
    "allowance_required" (= starting_balance x that rate),
    "net_charge_off" (quarterly, not annualized: projected/realized rate
    / 4 x starting_balance -- NaN for the jump-off row, since that's a
    baseline, not a quarter being rolled forward from a prior one), and
    "provision_expense" (allowance_required's own quarter-over-quarter
    change plus that quarter's net_charge_off, via the standard
    allowance roll-forward identity: ending = beginning + provision -
    net_charge_offs => provision = (ending - beginning) + net_charge_offs)."""
    wal_quarters = CATEGORY_WEIGHTED_AVERAGE_LIFE_QUARTERS[category]
    jump_off_quarter = realized_annualized_nco_rate.index.max()
    if jump_off_quarter in projected_annualized_nco_rate.index:
        raise ValueError(
            "projected_annualized_nco_rate must not include the jump-off quarter "
            f"({jump_off_quarter}) -- it's supplied separately via realized_annualized_nco_rate"
        )

    jump_off_rate = backward_looking_lifetime_expected_loss_rate(
        realized_annualized_nco_rate, jump_off_quarter, wal_quarters
    )
    rows = [
        {
            "quarter": jump_off_quarter,
            "lifetime_expected_loss_rate": jump_off_rate,
            "allowance_required": starting_balance * jump_off_rate,
            "net_charge_off": float("nan"),
            "provision_expense": float("nan"),
        }
    ]

    projected_sorted = projected_annualized_nco_rate.sort_index()
    prior_allowance = rows[0]["allowance_required"]
    for quarter in projected_sorted.index:
        lifetime_rate = lifetime_expected_loss_rate(projected_sorted, quarter, wal_quarters)
        allowance_required = starting_balance * lifetime_rate
        net_charge_off = (projected_sorted.loc[quarter] / 4.0) * starting_balance
        provision_expense = (allowance_required - prior_allowance) + net_charge_off
        rows.append(
            {
                "quarter": quarter,
                "lifetime_expected_loss_rate": lifetime_rate,
                "allowance_required": allowance_required,
                "net_charge_off": net_charge_off,
                "provision_expense": provision_expense,
            }
        )
        prior_allowance = allowance_required

    return pd.DataFrame(rows).set_index("quarter")


def build_projection_macro_lag_frame(
    combined_macro_path: pd.DataFrame, projection_quarters: list[pd.Period]
) -> pd.DataFrame:
    """combined_macro_path: quarter-indexed RAW (unlagged) macro columns
    covering at least one quarter before `projection_quarters[0]` -- e.g.
    macro.build_full_macro_path(macro_history, scenario)'s output, which
    concatenates the Fed's own actual history with a scenario path (Stage
    3) and validates they adjoin with no gap. Returns a frame with
    "quarter", one models.feature_column(variable) per models.
    CORE_MACRO_FEATURES at models.LAG, and "pandemic" (always 0.0 --
    every projection quarter here is 2026Q1 or later, past
    models.PANDEMIC_END; set explicitly rather than left to default-fill
    as NaN after a concat, which would otherwise poison every family that
    reads "pandemic" directly from the frame rather than recomputing it
    -- a real bug this project's own tests caught: forecast_aggregate_
    dynamic reads it as a plain feature column, unlike fit_panel_fe_model/
    fit_gbm_model, which always recompute it fresh via
    models.add_pandemic_indicator regardless of what's already there).
    The model-fitting functions select each category's own relevant
    subset of the macro columns; this builds the full set unconditionally,
    the same division of responsibility industry_history.py uses."""
    rows = []
    for quarter in projection_quarters:
        lagged_quarter = quarter - models.LAG
        row = {"quarter": quarter, "pandemic": 0.0}
        for variable in models.CORE_MACRO_FEATURES:
            row[models.feature_column(variable)] = combined_macro_path.loc[lagged_quarter, variable]
        rows.append(row)
    return pd.DataFrame(rows)


def build_synthetic_future_bank_frame(
    last_actual_bank_quarter: pd.DataFrame, projection_macro_lag_frame: pd.DataFrame
) -> pd.DataFrame:
    """last_actual_bank_quarter: one category's bank-level rows for the
    LAST ACTUAL quarter only (e.g. 2025Q4) -- every bank's own
    characteristics (average_balance, bank_id, etc.) held STATIC at this
    level for every future quarter (the same static-balance-sheet
    convention this module uses throughout -- a bank-level analogue of
    holding the aggregate balance constant). `projection_macro_lag_frame`:
    `build_projection_macro_lag_frame`'s output. Returns one row per
    (bank, projection quarter), with every macro lag column OVERWRITTEN
    by that quarter's own projection value (broadcast identically across
    banks, matching how these columns are already constant across banks
    within a real quarter) -- used by the bank-level families (panel_fe,
    gbm, and, for just its bank_id/quarter columns, anchored_to_
    aggregate) to project forward."""
    macro_columns = [c for c in projection_macro_lag_frame.columns if c != "quarter"]
    frames = []
    for _, macro_row in projection_macro_lag_frame.iterrows():
        quarter_frame = last_actual_bank_quarter.copy()
        quarter_frame["quarter"] = macro_row["quarter"]
        for column in macro_columns:
            quarter_frame[column] = macro_row[column]
        frames.append(quarter_frame)
    return pd.concat(frames, ignore_index=True)


def project_category_nco_rate(
    category: str,
    best_model_family: str,
    category_train_dataset: pd.DataFrame,
    macro_history: pd.DataFrame,
    scenario: pd.DataFrame,
    long_history_frame: pd.DataFrame | None,
) -> pd.Series:
    """Fits `best_model_family` on the FULL sample (`category_train_
    dataset`, quarter <= 2025Q4, "winsorized_nco_rate" as the dependent
    variable -- CECL allowances fund future charge-offs, so the NCO rate
    is the relevant loss-rate driver, not the NPL ratio) and projects the
    industry NCO rate forward over `scenario`'s own quarters, using
    `scenario`'s macro path (NOT real historical macro data -- this is a
    genuine forecast into the unknown future, unlike Stage 4's
    backtests). `macro_history`/`scenario`: quarter-indexed, raw
    (unlagged) macro columns, as macro.normalize_fed_historic/
    normalize_scenario produce. `long_history_frame` (industry_history.
    build_long_industry_frame's output) is required for "aggregate_long"
    and "anchored_to_aggregate" (see FAMILIES_REQUIRING_LONG_HISTORY),
    ignored otherwise. Returns a quarter-indexed Series covering
    `scenario`'s own quarters (the industry-level projected NCO rate)."""
    if best_model_family in FAMILIES_REQUIRING_LONG_HISTORY and long_history_frame is None:
        raise ValueError(f"{best_model_family} requires long_history_frame")

    full_macro_path = macro.build_full_macro_path(macro_history, scenario)
    projection_quarters = list(scenario.index)
    macro_lag_frame = build_projection_macro_lag_frame(full_macro_path, projection_quarters)

    industry_series_full = models.build_industry_series(
        category_train_dataset, "winsorized_nco_rate"
    ).frame

    if best_model_family in ("aggregate_ar", "aggregate_long"):
        source_frame = (
            long_history_frame if best_model_family == "aggregate_long" else industry_series_full
        )
        extended = pd.concat([source_frame, macro_lag_frame], ignore_index=True, sort=False)
        result = models.fit_aggregate_model(source_frame, category, include_pandemic_dummy=True)
        return models.forecast_aggregate_dynamic(
            result,
            extended,
            category,
            projection_quarters[0],
            projection_quarters[-1],
            include_pandemic_dummy=True,
        )

    last_quarter = category_train_dataset["quarter"].max()
    last_actual_bank_quarter = category_train_dataset[
        category_train_dataset["quarter"] == last_quarter
    ]
    synthetic_future_bank = build_synthetic_future_bank_frame(
        last_actual_bank_quarter, macro_lag_frame
    )

    if best_model_family == "panel_fe":
        fit = models.fit_panel_fe_model(
            category_train_dataset, category, "winsorized_nco_rate", include_pandemic_dummy=True
        )
        predicted_bank = models.predict_panel_fe(
            fit, synthetic_future_bank, category, "winsorized_nco_rate", include_pandemic_dummy=True
        )
        return models.aggregate_bank_predictions_to_industry_rate(
            synthetic_future_bank, predicted_bank
        )

    if best_model_family == "gbm":
        gbm_model = models.fit_gbm_model(
            category_train_dataset, category, "winsorized_nco_rate", include_pandemic_dummy=True
        )
        predicted_bank = models.predict_gbm(
            gbm_model, synthetic_future_bank, category, include_pandemic_dummy=True
        )
        return models.aggregate_bank_predictions_to_industry_rate(
            synthetic_future_bank, predicted_bank
        )

    if best_model_family == "anchored_to_aggregate":
        long_extended = pd.concat(
            [long_history_frame, macro_lag_frame], ignore_index=True, sort=False
        )
        long_result = models.fit_aggregate_model(
            long_history_frame, category, include_pandemic_dummy=True
        )
        long_forecast = models.forecast_aggregate_dynamic(
            long_result,
            long_extended,
            category,
            projection_quarters[0],
            projection_quarters[-1],
            include_pandemic_dummy=True,
        )
        long_train_matching_bank_period = long_history_frame[
            long_history_frame["quarter"].isin(category_train_dataset["quarter"])
        ]
        relative_levels = models.compute_bank_relative_levels(
            category_train_dataset, "winsorized_nco_rate", long_train_matching_bank_period
        )
        predicted_bank = models.forecast_anchored_to_aggregate(
            relative_levels, long_forecast, synthetic_future_bank
        )
        return models.aggregate_bank_predictions_to_industry_rate(
            synthetic_future_bank, predicted_bank
        )

    raise ValueError(f"unknown model family {best_model_family!r}")
