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
simplification: approximate "lifetime expected loss rate" as the
CUMULATIVE loss rate over a fixed, documented WEIGHTED-AVERAGE-LIFE
window per category (`CATEGORY_WEIGHTED_AVERAGE_LIFE_QUARTERS`) --
industry rule-of-thumb figures (not derived from this project's own
data, since there's no loan-level maturity data to derive them from),
clearly flagged as an ASSUMPTION a real implementation would replace with
a bank's own prepayment-adjusted duration estimates. Concretely: the
projected/realized NCO rate path is ANNUALIZED (a 1-year rate), so the
cumulative loss over `wal_quarters` quarters is the mean annualized rate
over that window TIMES `wal_quarters / 4` (4 quarters/year) -- NOT the
mean annualized rate on its own, which would understate a multi-year
lifetime loss by exactly the WAL's own length in years (a real bug this
project's own Stage 5 review caught: a WAL of 32 quarters, e.g. CRE
multifamily, needs roughly 8x the 1-year rate, not 1x it). The result is
floored at zero -- a negative "lifetime expected loss rate" would imply
a bank expects to be PAID to hold these loans, which can only happen here
because a short realized/projected window can have more recoveries than
charge-offs in a few quarters; CECL allowances are a reserve against
future losses, never a negative one. The BALANCE used throughout is held
STATIC at each category's last actual (2025Q4) level -- the same "static
balance sheet" convention real regulatory stress tests (DFAST/CCAR) use
for exactly this reason: it isolates the loss-rate projection from a
separate, unmodeled balance-growth forecast.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from corefin.credit import macro, models

# Model families that need the FRED-derived long-history aggregate frame
# (industry_history.build_long_industry_frame's output) rather than the
# Call-Report-only one: aggregate_long fits directly on it; anchored_to_
# aggregate anchors bank-level projections to ITS forecast.
FAMILIES_REQUIRING_LONG_HISTORY = ("aggregate_long", "anchored_to_aggregate")

# Categories whose long-history series (fred.INDUSTRY_CHARGEOFF_
# DELINQUENCY_SERIES) is a PROXY shared with at least one other category,
# rather than its own dedicated series -- flagged on a real Stage 5
# review: cre_construction and cre_multifamily produced IDENTICAL
# projections because both selected "aggregate_long" (the raw long-
# history aggregate forecast), which is fit on the exact SAME FRED
# "Commercial Real Estate Loans (Excluding Farmland)" series for all
# three CRE categories (FRED does not publish a construction/multifamily/
# nonfarm-nonresidential split); other_consumer shares its own proxy
# ("Other Consumer Loans") with auto. For these categories, projection
# uses "anchored_to_aggregate" INSTEAD of raw "aggregate_long" even when
# aggregate_long backtests best: anchored_to_aggregate still uses the
# same shared series' forecast SHAPE, but rescales it by each category's
# OWN bank-level relative historical loss level (compute_bank_relative_
# levels, computed separately per category from the Call Report panel),
# which differs across categories even though the underlying FRED series
# doesn't -- this is what actually differentiates construction's
# projection from multifamily's.
PROXY_CATEGORIES_REQUIRING_ANCHOR: frozenset[str] = frozenset(
    {"cre_construction", "cre_multifamily", "cre_nonfarm_nonresidential", "other_consumer"}
)

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
    mean of `annualized_nco_rate_path` over the `wal_quarters` quarters
    starting at `quarter` (inclusive), scaled from a 1-year ANNUALIZED
    rate to a CUMULATIVE rate over the full `wal_quarters`-quarter window
    by multiplying by `wal_quarters / 4` (see module docstring), then
    floored at 0.0 (a negative lifetime loss rate isn't meaningful for a
    loss reserve). If the path doesn't reach that far forward (either
    because the projection horizon ends first, or because `quarter` is
    the historical jump-off itself and only past data exists), the LAST
    available quarter's rate is held constant to fill the remainder -- a
    simple, documented terminal-value convention, not a real amortization
    schedule. Raises if `quarter` itself isn't in `annualized_nco_rate_
    path` at all."""
    if quarter not in annualized_nco_rate_path.index:
        raise ValueError(f"{quarter} is not in the provided rate path")
    window_quarters = [quarter + i for i in range(wal_quarters)]
    available = annualized_nco_rate_path.reindex(window_quarters)
    if available.notna().any():
        available = available.ffill()
    average_annualized_rate = float(available.mean())
    lifetime_rate = average_annualized_rate * (wal_quarters / 4.0)
    return max(lifetime_rate, 0.0)


def backward_looking_lifetime_expected_loss_rate(
    realized_annualized_nco_rate: pd.Series, as_of_quarter: pd.Period, wal_quarters: int
) -> float:
    """The simplified "lifetime expected loss rate" as of a REALIZED
    (non-projected) quarter, e.g. the 2025Q4 jump-off -- the mean of
    `realized_annualized_nco_rate` over the `wal_quarters` quarters
    ENDING at `as_of_quarter` (backward-looking, since there is no
    forward projection to average yet at the jump-off itself), scaled and
    floored exactly as `lifetime_expected_loss_rate` is (see that
    function and the module docstring). Used only to establish a baseline
    allowance level to roll the projection's provision forward from --
    not a claim about what the real bank actually held as its allowance
    that quarter (this project has no per-category real allowance data --
    see panel.py's allowance functions, which are bank-TOTAL, not split
    by category)."""
    window_quarters = [as_of_quarter - i for i in range(wal_quarters)]
    available = realized_annualized_nco_rate.reindex(window_quarters).dropna()
    if available.empty:
        raise ValueError(f"no realized data available on or before {as_of_quarter}")
    average_annualized_rate = float(available.mean())
    lifetime_rate = average_annualized_rate * (wal_quarters / 4.0)
    return max(lifetime_rate, 0.0)


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


@dataclass(frozen=True)
class CalibrationCheck:
    """Compares the simplified CECL engine's own jump-off (2025Q4)
    allowance, totaled across every projected category, against the REAL
    industry allowance ratio derived from the Call Report panel itself --
    a sanity check that the simplified WAL-window methodology produces a
    plausible overall reserve level, not just plausible per-category
    SHAPES. `model_total_loans`/`real_total_loans` are both the SAME
    category scope (whichever categories a projection run covers), so the
    comparison isolates the allowance METHODOLOGY, not a scope
    difference -- except `real_total_allowance`, which is each bank's
    TOTAL allowance balance across ALL its loan types (panel.py's
    allowance functions are bank-total, not split by category -- there is
    no real per-category allowance to compare against directly), so
    `real_allowance_ratio` is a slight OVERSTATEMENT of the true ratio for
    just these categories' loans (its numerator covers loan types outside
    `real_total_loans`' denominator too)."""

    model_total_allowance: float
    model_total_loans: float
    real_total_allowance: float
    real_total_loans: float

    @property
    def model_allowance_ratio(self) -> float:
        return self.model_total_allowance / self.model_total_loans

    @property
    def real_allowance_ratio(self) -> float:
        return self.real_total_allowance / self.real_total_loans

    @property
    def gap(self) -> float:
        """model_allowance_ratio - real_allowance_ratio -- positive means
        the simplified engine reserves MORE than the real industry ratio,
        negative means it reserves LESS."""
        return self.model_allowance_ratio - self.real_allowance_ratio


def compute_calibration_gap(
    jump_off_allowance_by_category: dict[str, float],
    starting_balance_by_category: dict[str, float],
    real_total_allowance: float,
    real_total_loans: float,
) -> CalibrationCheck:
    """`jump_off_allowance_by_category`/`starting_balance_by_category`:
    {category -> the 2025Q4 jump-off `allowance_required`/`starting_
    balance` from `build_cecl_projection`}, one entry per category a
    projection run covered. `real_total_allowance`/`real_total_loans`:
    the real, bank-level totals at 2025Q4 (e.g. summed from panel.py's
    allowance rollforward and the Call Report panel's own average
    balances) -- see `CalibrationCheck` for the scope caveat on the
    allowance side."""
    return CalibrationCheck(
        model_total_allowance=sum(jump_off_allowance_by_category.values()),
        model_total_loans=sum(starting_balance_by_category.values()),
        real_total_allowance=real_total_allowance,
        real_total_loans=real_total_loans,
    )


# The Fed's own standard DFAST reporting window is 9 quarters -- SHORTER
# than this project's own 13-quarter (3.25-year) scenario horizon. Used
# to truncate our own projection to the same window before comparing
# against the Fed's published severely-adverse cumulative loss rates
# (see cli.py's `fed-comparison` command).
FED_COMPARISON_QUARTERS = 9


def cumulative_loss_rate(annualized_nco_rate_path: pd.Series, n_quarters: int) -> float:
    """The cumulative loss rate over the FIRST `n_quarters` quarters of
    `annualized_nco_rate_path` (sorted by quarter): the sum of each
    quarter's own quarterly loss contribution (annualized rate / 4) --
    the same quarterly-flow convention `build_cecl_projection`'s own
    "net_charge_off" column uses, just summed across quarters instead of
    applied to a dollar balance. Used for the 9-quarter DFAST-style
    comparison against the Fed's own published stress test results, and
    to gauge how much a category's loss path actually moves between
    scenarios."""
    sorted_path = annualized_nco_rate_path.sort_index()
    window = sorted_path.iloc[:n_quarters]
    return float((window / 4.0).sum())


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
    ignored otherwise -- truncated here to `macro_history`'s own last
    quarter if it reaches any further (a real gap this project hit: the
    raw FRED pull behind `long_history_frame` can extend a quarter or two
    past the Fed's own historic actuals table, e.g. 2026Q1-Q2, which
    would otherwise DUPLICATE the quarters `scenario` itself covers once
    concatenated, corrupting the forecast). Returns a quarter-indexed
    Series covering `scenario`'s own quarters (the industry-level
    projected NCO rate)."""
    if best_model_family in FAMILIES_REQUIRING_LONG_HISTORY and long_history_frame is None:
        raise ValueError(f"{best_model_family} requires long_history_frame")

    jump_off_quarter = macro_history.index.max()
    if long_history_frame is not None:
        long_history_frame = long_history_frame[long_history_frame["quarter"] <= jump_off_quarter]

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


# The 3 "core" macro features model selection checks for a wrong-signed,
# statistically significant full-sample coefficient -- deliberately NOT
# the stock index: CATEGORY_MACRO_FEATURES already drops it everywhere
# except commercial_and_industrial for exactly this reason (models.py's
# module docstring), so re-checking it here would just re-flag an
# already-handled, already-documented exception rather than catch a new
# problem.
CORE_SIGN_CHECK_FEATURES: tuple[str, ...] = (
    models.UNEMPLOYMENT_FEATURE,
    models.HPI_FEATURE,
    models.CRE_PRICE_FEATURE,
)

FALLBACK_MODEL_FAMILY = "aggregate_ar"


@dataclass(frozen=True)
class CandidateModel:
    """One candidate model family considered for a category's projection.
    `coefficients`/`t_values`/`sign_classification` are None for families
    with no own linear coefficients to check (gbm; anchored_to_aggregate
    reports the UNDERLYING aggregate_long fit it anchors to instead, since
    that fit is what drives its forecast's shape -- see `select_
    projection_model`). `clean` is True when there is nothing to flag (no
    coefficients to check) or every `CORE_SIGN_CHECK_FEATURES` entry
    present classified as anything other than `models.
    WRONG_SIGN_SIGNIFICANT`. `rmse`: this family's best-available backtest
    RMSE for this category (2007-2010/main, falling back to 2020-2021/
    main) -- None if neither window has a row for it at all."""

    family: str
    rmse: float | None
    coefficients: dict[str, float] | None
    t_values: dict[str, float] | None
    sign_classification: dict[str, str] | None
    clean: bool


@dataclass(frozen=True)
class ProjectionModelSelection:
    """The outcome of `select_projection_model` for one category:
    `selected_family` is the chosen model, `is_clean` is False only when
    NO candidate was clean and the lowest-RMSE (or, lacking any RMSE,
    `FALLBACK_MODEL_FAMILY`) family had to be selected anyway, and
    `candidates` has one `CandidateModel` per family actually considered
    (for reporting every category's key coefficients, not just the
    selected one)."""

    category: str
    selected_family: str
    is_clean: bool
    candidates: dict[str, CandidateModel]


def _classify_core_signs(
    coefficients: dict[str, float], t_values: dict[str, float]
) -> dict[str, str]:
    classification = models.classify_coefficient_significance(coefficients, t_values)
    core_columns = {models.feature_column(feature) for feature in CORE_SIGN_CHECK_FEATURES}
    return {column: value for column, value in classification.items() if column in core_columns}


def _is_clean(sign_classification: dict[str, str] | None) -> bool:
    if sign_classification is None:
        return True
    return models.WRONG_SIGN_SIGNIFICANT not in sign_classification.values()


def _lookup_backtest_rmse(
    backtest_table: pd.DataFrame,
    category: str,
    family: str,
    dependent: str = "nco_rate",
    covid_spec: str = "main",
) -> float | None:
    """This category/family's backtest RMSE, trying the primary
    2007-2010/main window first and falling back to 2020-2021/main for a
    category with no data that far back (the same two windows/fallback
    order `_determine_best_nco_model` used) -- None if neither window has
    a usable (non-NaN RMSE) row for it."""
    for window in ("2007-2010", "2020-2021"):
        rows = backtest_table[
            (backtest_table["category"] == category)
            & (backtest_table["dependent"] == dependent)
            & (backtest_table["covid_spec"] == covid_spec)
            & (backtest_table["window"] == window)
            & (backtest_table["model_family"] == family)
        ].dropna(subset=["rmse"])
        if not rows.empty:
            return float(rows.iloc[0]["rmse"])
    return None


def select_projection_model(
    category: str,
    category_train_dataset: pd.DataFrame,
    long_history_frame: pd.DataFrame | None,
    backtest_table: pd.DataFrame,
) -> ProjectionModelSelection:
    """Selects the model family to PROJECT `category`'s NCO rate forward
    with: among the ELIGIBLE families (aggregate_ar, panel_fe, and gbm
    always; aggregate_long and anchored_to_aggregate when `long_history_
    frame` is available -- except for PROXY_CATEGORIES_REQUIRING_ANCHOR,
    where aggregate_long is EXCLUDED and only anchored_to_aggregate is
    eligible, since those categories' aggregate_long fit is on a series
    shared with another category, see that constant), picks the one with
    the best (lowest) backtest RMSE AMONG those whose FULL-SAMPLE fit (on
    `category_train_dataset`/`long_history_frame` through 2025Q4, NOT a
    backtest-window fit) has no WRONG_SIGN_SIGNIFICANT core macro
    coefficient (`CORE_SIGN_CHECK_FEATURES`) -- a model that gets a core
    macro driver's sign wrong, with statistical confidence, on the FULL
    sample isn't projecting a trustworthy economic relationship forward,
    even if it happened to backtest well on a specific historical crisis
    window. gbm has no coefficients to check (always eligible on that
    basis); anchored_to_aggregate has none of its OWN either, so it
    inherits aggregate_long's full-sample sign classification (and
    reports aggregate_long's coefficients, for transparency) -- its
    forecast SHAPE comes directly from that same long-history aggregate
    fit (`project_category_nco_rate`), so a sign problem there is a sign
    problem for anchored_to_aggregate too. If NO eligible family is clean,
    falls back to the lowest-RMSE eligible family regardless (`is_clean=
    False` on the result) -- the same "pick the most useful tool
    available" principle `_determine_best_nco_model` used to apply to
    RMSE alone, now applied to the sign-filtered subset first."""
    eligible_families = ["aggregate_ar", "panel_fe", "gbm"]
    if long_history_frame is not None:
        if category in PROXY_CATEGORIES_REQUIRING_ANCHOR:
            eligible_families.append("anchored_to_aggregate")
        else:
            eligible_families.extend(["aggregate_long", "anchored_to_aggregate"])

    industry_series_full = models.build_industry_series(
        category_train_dataset, "winsorized_nco_rate"
    ).frame

    long_fit_cache: dict[str, dict] = {}

    def fit_long_once() -> dict:
        if not long_fit_cache:
            long_result = models.fit_aggregate_model(
                long_history_frame, category, include_pandemic_dummy=True
            )
            coefficients = long_result.params.to_dict()
            t_values = long_result.tvalues.to_dict()
            long_fit_cache["coefficients"] = coefficients
            long_fit_cache["t_values"] = t_values
            long_fit_cache["sign_classification"] = _classify_core_signs(coefficients, t_values)
        return long_fit_cache

    candidates: dict[str, CandidateModel] = {}
    for family in eligible_families:
        rmse = _lookup_backtest_rmse(backtest_table, category, family)
        if family == "aggregate_ar":
            result = models.fit_aggregate_model(
                industry_series_full, category, include_pandemic_dummy=True
            )
            coefficients = result.params.to_dict()
            t_values = result.tvalues.to_dict()
            sign_classification = _classify_core_signs(coefficients, t_values)
        elif family == "aggregate_long":
            long_fit = fit_long_once()
            coefficients = long_fit["coefficients"]
            t_values = long_fit["t_values"]
            sign_classification = long_fit["sign_classification"]
        elif family == "panel_fe":
            fit = models.fit_panel_fe_model(
                category_train_dataset, category, "winsorized_nco_rate", include_pandemic_dummy=True
            )
            coefficients = fit.result.params.to_dict()
            t_values = fit.result.tvalues.to_dict()
            sign_classification = _classify_core_signs(coefficients, t_values)
        elif family == "gbm":
            coefficients, t_values, sign_classification = None, None, None
        elif family == "anchored_to_aggregate":
            long_fit = fit_long_once()
            coefficients = long_fit["coefficients"]
            t_values = long_fit["t_values"]
            sign_classification = long_fit["sign_classification"]
        else:
            raise ValueError(f"unknown family {family!r}")

        candidates[family] = CandidateModel(
            family=family,
            rmse=rmse,
            coefficients=coefficients,
            t_values=t_values,
            sign_classification=sign_classification,
            clean=_is_clean(sign_classification),
        )

    rated = {family: c for family, c in candidates.items() if c.rmse is not None}
    clean_rated = {family: c for family, c in rated.items() if c.clean}
    if clean_rated:
        selected = min(clean_rated, key=lambda f: clean_rated[f].rmse)
        is_clean = True
    elif rated:
        selected = min(rated, key=lambda f: rated[f].rmse)
        is_clean = False
    else:
        selected = FALLBACK_MODEL_FAMILY
        is_clean = candidates[FALLBACK_MODEL_FAMILY].clean

    return ProjectionModelSelection(
        category=category, selected_family=selected, is_clean=is_clean, candidates=candidates
    )
