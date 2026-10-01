"""Diagnostic-only decomposition of the crisis-replay vs. Fed-comparison
gap (README's Stage 5 limitations section): for the DYNAMIC, AR-term
families (aggregate_ar, aggregate_long, anchored_to_aggregate), isolates
how much of a forecast is driven by the AR term's own STARTING LEVEL
(the real rate at the seed quarter the dynamic recursion is kicked off
from) versus the MACRO PATH it's then fed, by swapping seeds between the
crisis-replay path (seeded from 2007Q3's real rate) and the severely-
adverse projection path (seeded from 2025Q4's real rate) while holding
each path's own macro inputs fixed.

Calls `models.fit_aggregate_model`/`forecast_aggregate_dynamic` etc.
EXACTLY as `projection.replay_crisis_window`/`project_category_nco_rate`
do, on the SAME data -- this module changes NOTHING about how those
functions behave or how a real model is selected or fit for an actual
projection; it only constructs an ADDITIONAL, seed-overridden COPY of
the input frame `forecast_aggregate_dynamic` reads its seed from, for
a side-by-side diagnostic comparison. The model is always FIT on the
real, unmodified data (identical coefficients either way) -- only the
value `forecast_aggregate_dynamic` reads at the seed quarter, to kick
off its own dynamic recursion, differs between the "natural" and
"swapped" variants.

panel_fe/gbm have no AR term (no lagged dependent variable as a
regressor) and so no "seed" to swap -- not supported here, raises.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from corefin.credit import macro, models, projection

SEED_SWAPPABLE_FAMILIES: tuple[str, ...] = (
    "aggregate_ar",
    "aggregate_long",
    "anchored_to_aggregate",
)

CORE_DRIVER_FEATURES: tuple[str, ...] = (
    models.UNEMPLOYMENT_FEATURE,
    models.HPI_FEATURE,
    models.CRE_PRICE_FEATURE,
)


def _override_seed_rate(
    frame: pd.DataFrame, seed_quarter: pd.Period, new_rate: float
) -> pd.DataFrame:
    """A COPY of `frame` (one row per quarter, an "industry_rate" column
    -- the shape `models.build_industry_series`'s `.frame` and
    `industry_history.build_long_industry_frame`'s output both share)
    with the row at `seed_quarter`'s "industry_rate" replaced by
    `new_rate`. Does not touch any other row -- in particular, every
    row models.fit_aggregate_model would use as a TRAINING observation
    is untouched, so fitting on this copy (if a caller did that by
    mistake) would NOT reproduce the real coefficients; this function is
    meant to be used only as the SECOND argument to `forecast_aggregate_
    dynamic` (which reads only the seed quarter and the forecast
    quarters), never as the frame a model is FIT on."""
    overridden = frame.copy()
    overridden.loc[overridden["quarter"] == seed_quarter, "industry_rate"] = new_rate
    return overridden


def ar_seed_series(
    family: str,
    category_train_dataset: pd.DataFrame,
    long_history_frame: pd.DataFrame | None,
) -> pd.Series:
    """The quarter-indexed "industry_rate" series `family`'s dynamic
    forecast draws its AR-term seed from: the Call-Report industry rate
    for aggregate_ar, the FRED long-history industry rate (industry_
    history.build_long_industry_frame's output) for aggregate_long/
    anchored_to_aggregate -- anchored_to_aggregate has no AR term of its
    OWN, but its forecast SHAPE comes directly from the long-history
    aggregate fit, which does (the same "inherits from aggregate_long"
    reasoning `projection.select_projection_model`'s sign check uses)."""
    if family in ("aggregate_long", "anchored_to_aggregate"):
        if long_history_frame is None:
            raise ValueError(f"{family} requires long_history_frame")
        return long_history_frame.set_index("quarter")["industry_rate"]
    return models.build_industry_series(
        category_train_dataset, "winsorized_nco_rate"
    ).frame.set_index("quarter")["industry_rate"]


def replay_with_seed(
    category: str,
    family: str,
    category_train_dataset: pd.DataFrame,
    long_history_frame: pd.DataFrame | None,
    seed_override: float | None = None,
    replay_start: pd.Period = projection.CRISIS_REPLAY_START,
    replay_end: pd.Period = projection.CRISIS_REPLAY_END,
) -> pd.Series:
    """The SAME fit+forecast `projection.replay_crisis_window` performs
    for `family` (identical coefficients -- fit on the real, unmodified
    data), returning the full projected Series instead of just its 9Q
    summary. If `seed_override` is given, the AR term is kicked off from
    that value instead of the real rate at `replay_start - 1` -- the fit
    itself is unaffected (see `_override_seed_rate`'s own docstring).
    Only `SEED_SWAPPABLE_FAMILIES` are supported."""
    if family not in SEED_SWAPPABLE_FAMILIES:
        raise ValueError(f"{family} has no AR term to seed-swap")
    seed_quarter = replay_start - 1

    if family in ("aggregate_ar", "aggregate_long"):
        source_frame = (
            long_history_frame
            if family == "aggregate_long"
            else models.build_industry_series(category_train_dataset, "winsorized_nco_rate").frame
        )
        result = models.fit_aggregate_model(source_frame, category, include_pandemic_dummy=True)
        forecast_input = (
            source_frame
            if seed_override is None
            else _override_seed_rate(source_frame, seed_quarter, seed_override)
        )
        return models.forecast_aggregate_dynamic(
            result, forecast_input, category, replay_start, replay_end, include_pandemic_dummy=True
        )

    # anchored_to_aggregate
    long_result = models.fit_aggregate_model(
        long_history_frame, category, include_pandemic_dummy=True
    )
    forecast_input = (
        long_history_frame
        if seed_override is None
        else _override_seed_rate(long_history_frame, seed_quarter, seed_override)
    )
    long_forecast = models.forecast_aggregate_dynamic(
        long_result, forecast_input, category, replay_start, replay_end, include_pandemic_dummy=True
    )
    long_train_matching_bank_period = long_history_frame[
        long_history_frame["quarter"].isin(category_train_dataset["quarter"])
    ]
    relative_levels = models.compute_bank_relative_levels(
        category_train_dataset, "winsorized_nco_rate", long_train_matching_bank_period
    )
    replay_bank = category_train_dataset[
        (category_train_dataset["quarter"] >= replay_start)
        & (category_train_dataset["quarter"] <= replay_end)
    ]
    predicted_bank = models.forecast_anchored_to_aggregate(
        relative_levels, long_forecast, replay_bank
    )
    return models.aggregate_bank_predictions_to_industry_rate(replay_bank, predicted_bank)


def scenario_forecast_with_seed(
    category: str,
    family: str,
    category_train_dataset: pd.DataFrame,
    macro_history: pd.DataFrame,
    scenario: pd.DataFrame,
    long_history_frame: pd.DataFrame | None,
    seed_override: float | None = None,
) -> pd.Series:
    """The SAME fit+forecast `projection.project_category_nco_rate`
    performs for `family` (identical coefficients), returning the full
    projected Series. If `seed_override` is given, the AR term is kicked
    off from that value instead of the real rate at `macro_history`'s
    own last quarter -- see `replay_with_seed`. Only
    `SEED_SWAPPABLE_FAMILIES` are supported."""
    if family not in SEED_SWAPPABLE_FAMILIES:
        raise ValueError(f"{family} has no AR term to seed-swap")

    jump_off_quarter = macro_history.index.max()
    if long_history_frame is not None:
        long_history_frame = long_history_frame[long_history_frame["quarter"] <= jump_off_quarter]

    full_macro_path = macro.build_full_macro_path(macro_history, scenario)
    projection_quarters = list(scenario.index)
    macro_lag_frame = projection.build_projection_macro_lag_frame(
        full_macro_path, projection_quarters
    )

    if family in ("aggregate_ar", "aggregate_long"):
        source_frame = (
            long_history_frame
            if family == "aggregate_long"
            else models.build_industry_series(category_train_dataset, "winsorized_nco_rate").frame
        )
        extended = pd.concat([source_frame, macro_lag_frame], ignore_index=True, sort=False)
        result = models.fit_aggregate_model(source_frame, category, include_pandemic_dummy=True)
        forecast_input = (
            extended
            if seed_override is None
            else _override_seed_rate(extended, jump_off_quarter, seed_override)
        )
        return models.forecast_aggregate_dynamic(
            result,
            forecast_input,
            category,
            projection_quarters[0],
            projection_quarters[-1],
            include_pandemic_dummy=True,
        )

    # anchored_to_aggregate
    long_extended = pd.concat(
        [long_history_frame, macro_lag_frame], ignore_index=True, sort=False
    )
    long_result = models.fit_aggregate_model(
        long_history_frame, category, include_pandemic_dummy=True
    )
    forecast_input = (
        long_extended
        if seed_override is None
        else _override_seed_rate(long_extended, jump_off_quarter, seed_override)
    )
    long_forecast = models.forecast_aggregate_dynamic(
        long_result,
        forecast_input,
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
    last_quarter = category_train_dataset["quarter"].max()
    last_actual_bank_quarter = category_train_dataset[
        category_train_dataset["quarter"] == last_quarter
    ]
    synthetic_future_bank = projection.build_synthetic_future_bank_frame(
        last_actual_bank_quarter, macro_lag_frame
    )
    predicted_bank = models.forecast_anchored_to_aggregate(
        relative_levels, long_forecast, synthetic_future_bank
    )
    return models.aggregate_bank_predictions_to_industry_rate(synthetic_future_bank, predicted_bank)


def peak_quarter_and_rate(path: pd.Series) -> tuple[pd.Period, float]:
    """The (quarter, value) of `path`'s maximum -- `path` is an
    annualized rate series, e.g. a forecast from `replay_with_seed`/
    `scenario_forecast_with_seed`."""
    peak_quarter = path.idxmax()
    return peak_quarter, float(path.loc[peak_quarter])


@dataclass(frozen=True)
class SeedDecomposition:
    """Four variants for one category/family -- `cumulative_Nq` keyed by
    quarter count (e.g. {9: ..., 13: ...}); `replay_natural`/`scenario_
    natural` reproduce `projection.replay_crisis_window`/`project_
    category_nco_rate`'s own real outputs exactly (same fit, same seed);
    `replay_swapped`/`scenario_swapped` use the OTHER path's natural seed
    value instead, holding each path's own macro inputs fixed."""

    category: str
    family: str
    replay_natural: pd.Series
    replay_swapped: pd.Series
    scenario_natural: pd.Series
    scenario_swapped: pd.Series


def decompose_seed_vs_macro(
    category: str,
    family: str,
    category_train_dataset: pd.DataFrame,
    macro_history: pd.DataFrame,
    scenario: pd.DataFrame,
    long_history_frame: pd.DataFrame | None,
) -> SeedDecomposition:
    """Builds all 4 variants described in `SeedDecomposition` for one
    category/family. `macro_history`/`scenario`: as `projection.
    project_category_nco_rate` takes them."""
    seed_series = ar_seed_series(family, category_train_dataset, long_history_frame)
    replay_seed_quarter = projection.CRISIS_REPLAY_START - 1
    scenario_seed_quarter = macro_history.index.max()
    replay_natural_seed = float(seed_series.loc[replay_seed_quarter])
    scenario_natural_seed = float(seed_series.loc[scenario_seed_quarter])

    replay_natural = replay_with_seed(category, family, category_train_dataset, long_history_frame)
    replay_swapped = replay_with_seed(
        category,
        family,
        category_train_dataset,
        long_history_frame,
        seed_override=scenario_natural_seed,
    )
    scenario_natural = scenario_forecast_with_seed(
        category, family, category_train_dataset, macro_history, scenario, long_history_frame
    )
    scenario_swapped = scenario_forecast_with_seed(
        category,
        family,
        category_train_dataset,
        macro_history,
        scenario,
        long_history_frame,
        seed_override=replay_natural_seed,
    )
    return SeedDecomposition(
        category=category,
        family=family,
        replay_natural=replay_natural,
        replay_swapped=replay_swapped,
        scenario_natural=scenario_natural,
        scenario_swapped=scenario_swapped,
    )


def macro_driver_table(
    macro_history: pd.DataFrame,
    scenario: pd.DataFrame,
    replay_start: pd.Period = projection.CRISIS_REPLAY_START,
    replay_end: pd.Period = projection.CRISIS_REPLAY_END,
    features: tuple[str, ...] = CORE_DRIVER_FEATURES,
) -> pd.DataFrame:
    """One row per (position within the 13-quarter horizon), with the
    LAGGED (t-1, `models.LAG`) value of each `features` variable the
    model actually sees -- side by side for the crisis-replay path (the
    real 2007Q4-2010Q4 window) and the severely-adverse scenario path,
    aligned by POSITION (quarter 1 of each, quarter 2 of each, ...), NOT
    by calendar quarter (they're 18 years apart). Built from the exact
    same `projection.build_projection_macro_lag_frame` the real
    projection/replay machinery uses -- no new macro computation."""
    replay_quarters = list(pd.period_range(replay_start, replay_end, freq="Q"))
    replay_lag_frame = projection.build_projection_macro_lag_frame(macro_history, replay_quarters)

    full_macro_path = macro.build_full_macro_path(macro_history, scenario)
    scenario_quarters = list(scenario.index)
    scenario_lag_frame = projection.build_projection_macro_lag_frame(
        full_macro_path, scenario_quarters
    )

    rows = []
    for position in range(len(replay_quarters)):
        row = {
            "position": position + 1,
            "replay_quarter": str(replay_lag_frame.iloc[position]["quarter"]),
            "scenario_quarter": str(scenario_lag_frame.iloc[position]["quarter"]),
        }
        for feature in features:
            column = models.feature_column(feature)
            row[f"replay_{feature}"] = replay_lag_frame.iloc[position][column]
            row[f"scenario_{feature}"] = scenario_lag_frame.iloc[position][column]
        rows.append(row)
    return pd.DataFrame(rows)
