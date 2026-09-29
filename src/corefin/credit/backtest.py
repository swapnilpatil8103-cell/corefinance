"""Stage 4 orchestration: sweeps every loan category x dependent variable
(winsorized NCO rate, winsorized NPL ratio) x COVID specification (main,
robustness) across the three model families (models.py) and the
requested validation windows (validation.py), producing tidy backtest
metrics, coefficient, and GBM feature-importance tables plus per-model
forecast series for charting. cli.py just loads/writes files around this
-- everything here is pure pandas/dict manipulation, no file IO.
"""

from __future__ import annotations

import pandas as pd

from corefin.credit import models, validation
from corefin.credit.panel import compute_npl_ratio, winsorize_rate
from corefin.credit.schema import LoanCategory

# Excludes AUTO_AND_OTHER_CONSUMER_COMBINED -- an intentionally
# overlapping, derived series (see schema.py), not a real balance-sheet
# category to model on its own, matching every other Stage 2/3 aggregate
# that excludes it (e.g. compute_bank_allowance_rollforward's default).
MODELING_CATEGORIES: tuple[LoanCategory, ...] = tuple(
    c for c in LoanCategory if c != LoanCategory.AUTO_AND_OTHER_CONSUMER_COMBINED
)

DEPENDENT_VARIABLES: dict[str, str] = {
    "nco_rate": "winsorized_nco_rate",
    "npl_ratio": "winsorized_npl_ratio",
}

COVID_SPECS: tuple[str, ...] = ("main", "robustness")

# (label, train_end, test_start, test_end) -- the two REQUIRED out-of-time
# backtest windows. Note: for both, the training window (<= train_end)
# never reaches 2020-2021, so "robustness" (which only changes TRAINING
# rows) is identical to "main" for these two -- the pandemic dummy has no
# quarter to be nonzero for in either training sample. The one window
# where main/robustness genuinely differ is the holdout fit (train
# through 2025Q4, which DOES span 2020-2021) -- see `run_holdout_backtest`.
VALIDATION_WINDOWS: tuple[tuple[str, pd.Period, pd.Period, pd.Period], ...] = (
    (
        "2007-2010",
        pd.Period("2006Q4", freq="Q"),
        pd.Period("2007Q1", freq="Q"),
        pd.Period("2010Q4", freq="Q"),
    ),
    (
        "2020-2021",
        pd.Period("2019Q4", freq="Q"),
        pd.Period("2020Q1", freq="Q"),
        pd.Period("2021Q4", freq="Q"),
    ),
)

HOLDOUT_TRAIN_END = pd.Period("2025Q4", freq="Q")

# Some categories have a real, documented coverage gap (e.g. AUTO has no
# code set before 2011Q1 -- see schema.py), so a validation window whose
# training period predates a category's own start has no data to fit on
# at all. Skip (not crash) a window when its training slice has fewer
# than this many quarters of industry-level data -- an AR term alone
# needs at least 2 just to have one non-NaN lagged observation; this is a
# generous floor above that, not a tight one, since a handful of
# quarters would fit but not mean anything.
MIN_TRAINING_QUARTERS = 8


def add_npl_ratio_columns(dataset: pd.DataFrame) -> pd.DataFrame:
    """Adds "npl_ratio" (panel.compute_npl_ratio) and its winsorized
    counterpart "winsorized_npl_ratio" (panel.winsorize_rate) -- the
    modeling dataset already has "winsorized_nco_rate" from Stage 3, but
    NPL is computed here since Stage 3's build_modeling_dataset had no
    reason to build it (Stage 3 was macro ingestion, not modeling)."""
    result = dataset.copy()
    result["npl_ratio"] = compute_npl_ratio(result)
    result["winsorized_npl_ratio"] = winsorize_rate(result, "npl_ratio")
    return result


def _run_three_families(
    industry_series_full: pd.DataFrame,
    train_industry: pd.DataFrame,
    train_bank: pd.DataFrame,
    test_bank: pd.DataFrame,
    dependent_column: str,
    covid_spec: str,
    test_start: pd.Period,
    test_end: pd.Period,
) -> dict:
    """Fits and backtests all three model families for one (category,
    dependent, covid_spec, window) combination. `industry_series_full`
    must cover at least [<= train_industry's start>, test_end] -- it's
    both the actual series backtest metrics are scored against AND the
    only source `forecast_aggregate_dynamic` reads for its (real,
    historical) macro lag features and its seed quarter's actual rate."""
    include_pandemic = covid_spec == "main"
    if covid_spec == "robustness":
        train_industry = models.exclude_pandemic_years(train_industry)
        train_bank = models.exclude_pandemic_years(train_bank)

    actual_industry = industry_series_full.set_index("quarter")["industry_rate"]

    agg_result = models.fit_aggregate_model(train_industry, include_pandemic_dummy=include_pandemic)
    agg_forecast = models.forecast_aggregate_dynamic(
        agg_result,
        industry_series_full,
        test_start,
        test_end,
        include_pandemic_dummy=include_pandemic,
    )
    agg_metrics = validation.backtest_summary(actual_industry, agg_forecast)

    fe_fit = models.fit_panel_fe_model(
        train_bank, dependent_column, include_pandemic_dummy=include_pandemic
    )
    fe_predicted_bank = models.predict_panel_fe(
        fe_fit, test_bank, dependent_column, include_pandemic_dummy=include_pandemic
    )
    fe_forecast = models.aggregate_bank_predictions_to_industry_rate(test_bank, fe_predicted_bank)
    fe_metrics = validation.backtest_summary(actual_industry, fe_forecast)

    gbm_model = models.fit_gbm_model(
        train_bank, dependent_column, include_pandemic_dummy=include_pandemic
    )
    gbm_predicted_bank = models.predict_gbm(
        gbm_model, test_bank, include_pandemic_dummy=include_pandemic
    )
    gbm_forecast = models.aggregate_bank_predictions_to_industry_rate(test_bank, gbm_predicted_bank)
    gbm_metrics = validation.backtest_summary(actual_industry, gbm_forecast)

    return {
        "aggregate_ar": {
            "metrics": agg_metrics,
            "coefficients": agg_result.params.to_dict(),
            "forecast": agg_forecast,
        },
        "panel_fe": {
            "metrics": fe_metrics,
            "coefficients": fe_fit.result.params.to_dict(),
            "forecast": fe_forecast,
        },
        "gbm": {
            "metrics": gbm_metrics,
            "feature_importances": dict(
                zip(gbm_model.feature_names_in_, gbm_model.feature_importances_, strict=True)
            ),
            "forecast": gbm_forecast,
        },
    }


def run_category_backtests(
    category_bank_dataset: pd.DataFrame, dependent_column: str, covid_spec: str
) -> dict[str, dict]:
    """Runs both required out-of-time validation windows
    (VALIDATION_WINDOWS) for one category/dependent/covid_spec, using
    ONLY the training dataset (quarter <= 2025Q4) -- the holdout window
    is separate, see `run_holdout_backtest`. Returns {window_label ->
    `_run_three_families`'s result dict}, OMITTING any window whose
    training slice has fewer than MIN_TRAINING_QUARTERS of data or whose
    test slice is entirely empty -- a real, expected situation for a
    category with a documented start-date gap (e.g. AUTO has no data
    before 2011Q1, so the 2006Q4-training/2007-2010 window has nothing to
    fit or score for it), not an error to raise on."""
    industry_series_full = models.build_industry_series(
        category_bank_dataset, dependent_column
    ).frame
    results = {}
    for label, train_end, test_start, test_end in VALIDATION_WINDOWS:
        train_industry = industry_series_full[industry_series_full["quarter"] <= train_end]
        train_bank, test_bank = validation.split_out_of_time(
            category_bank_dataset, train_end, test_start, test_end
        )
        if len(train_industry) < MIN_TRAINING_QUARTERS or test_bank.empty:
            continue
        results[label] = _run_three_families(
            industry_series_full,
            train_industry,
            train_bank,
            test_bank,
            dependent_column,
            covid_spec,
            test_start,
            test_end,
        )
    return results


def run_holdout_backtest(
    category_train_dataset: pd.DataFrame,
    category_holdout_dataset: pd.DataFrame,
    dependent_column: str,
    covid_spec: str,
) -> dict | None:
    """Fits on the FULL training history (quarter <= HOLDOUT_TRAIN_END,
    2025Q4) and scores against the real, already-known 2026Q1-Q2 holdout
    panel quarters build-modeling-dataset set aside -- this is the ONE
    window where "main" (pandemic dummy) and "robustness" (drop
    2020-2021 from training) genuinely differ, since it's the only
    training window that spans 2020-2021 at all. Returns None (same
    reasoning as `run_category_backtests`) if there isn't enough training
    or holdout data for this category to fit and score at all."""
    if category_holdout_dataset.empty:
        return None
    combined = pd.concat([category_train_dataset, category_holdout_dataset], ignore_index=True)
    industry_series_full = models.build_industry_series(combined, dependent_column).frame
    train_industry = industry_series_full[industry_series_full["quarter"] <= HOLDOUT_TRAIN_END]
    if len(train_industry) < MIN_TRAINING_QUARTERS:
        return None

    test_start = category_holdout_dataset["quarter"].min()
    test_end = category_holdout_dataset["quarter"].max()
    return _run_three_families(
        industry_series_full,
        train_industry,
        category_train_dataset,
        category_holdout_dataset,
        dependent_column,
        covid_spec,
        test_start,
        test_end,
    )
