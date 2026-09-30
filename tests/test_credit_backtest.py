"""Offline integration tests for the Stage 4 sweep orchestration
(backtest.py) -- verifies the wiring between models.py/validation.py
across categories/dependents/COVID specs on synthetic data. The deep
per-function behavior (no leakage, sign recovery, metric correctness) is
already covered by test_credit_models.py/test_credit_validation.py; these
tests check the orchestration layer itself doesn't break, and that the
documented main-vs-robustness equivalence for pre-COVID windows holds."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from corefin.credit import backtest, models


def _synthetic_category_dataset(start="2001Q1", n_quarters=84, n_banks=8, seed=1) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    quarters = pd.period_range(pd.Period(start, freq="Q"), periods=n_quarters, freq="Q")

    unemployment = 5.0 + np.cumsum(rng.normal(0, 0.2, size=n_quarters))
    hpi_yoy = 2.0 + np.cumsum(rng.normal(0, 0.3, size=n_quarters))
    cre_yoy = 2.0 + np.cumsum(rng.normal(0, 0.3, size=n_quarters))
    stock_yoy = 4.0 + np.cumsum(rng.normal(0, 0.4, size=n_quarters))

    bank_effect = rng.normal(0, 0.004, size=n_banks)

    rows = []
    for bank_idx in range(n_banks):
        base_balance = rng.uniform(5_000.0, 50_000.0)
        for q_idx, quarter in enumerate(quarters):
            true_nco = max(
                0.01
                + 0.0015 * unemployment[q_idx]
                - 0.001 * hpi_yoy[q_idx]
                + bank_effect[bank_idx]
                + rng.normal(0, 0.0004),
                -0.05,
            )
            balance = base_balance * (1.0 + rng.normal(0, 0.02))
            past_due_90 = max(balance * 0.01 + rng.normal(0, 1.0), 0.0)
            nonaccrual = max(balance * 0.005 + rng.normal(0, 0.5), 0.0)
            rows.append(
                {
                    "bank_id": bank_idx,
                    "quarter": quarter,
                    "category": "commercial_and_industrial",
                    "average_balance": balance,
                    "balance": balance,
                    "past_due_90": past_due_90,
                    "nonaccrual": nonaccrual,
                    "winsorized_nco_rate": true_nco,
                    "merger_flag": False,
                    models.feature_column("Unemployment rate"): unemployment[q_idx],
                    models.feature_column("House Price Index YoY % change"): hpi_yoy[q_idx],
                    models.feature_column(
                        "Commercial Real Estate Price Index YoY % change"
                    ): cre_yoy[q_idx],
                    models.feature_column(
                        "Dow Jones Total Stock Market Index YoY % change"
                    ): stock_yoy[q_idx],
                }
            )
    return backtest.add_npl_ratio_columns(pd.DataFrame(rows))


def _synthetic_long_history_frame(start="1991Q1", n_quarters=124, seed=3) -> pd.DataFrame:
    # Same shape as industry_history.build_long_industry_frame's output,
    # built directly here to avoid depending on a real FRED macro history
    # fixture just to exercise the wiring.
    rng = np.random.default_rng(seed)
    quarters = pd.period_range(pd.Period(start, freq="Q"), periods=n_quarters, freq="Q")
    unemployment = 5.0 + np.cumsum(rng.normal(0, 0.2, size=n_quarters))
    hpi_yoy = 2.0 + np.cumsum(rng.normal(0, 0.3, size=n_quarters))
    cre_yoy = 2.0 + np.cumsum(rng.normal(0, 0.3, size=n_quarters))
    stock_yoy = 4.0 + np.cumsum(rng.normal(0, 0.4, size=n_quarters))
    industry_rate = np.clip(
        0.01 + 0.0015 * unemployment - 0.001 * hpi_yoy + rng.normal(0, 0.0004, size=n_quarters),
        -0.05,
        None,
    )
    frame = pd.DataFrame(
        {
            "quarter": quarters,
            "industry_rate": industry_rate,
            models.feature_column("Unemployment rate"): unemployment,
            models.feature_column("House Price Index YoY % change"): hpi_yoy,
            models.feature_column("Commercial Real Estate Price Index YoY % change"): cre_yoy,
            models.feature_column("Dow Jones Total Stock Market Index YoY % change"): stock_yoy,
        }
    )
    frame["pandemic"] = models.add_pandemic_indicator(frame)
    frame["industry_rate_lag1"] = frame["industry_rate"].shift(1)
    return frame


def test_run_category_backtests_adds_long_history_families_when_given():
    dataset = _synthetic_category_dataset(n_quarters=84, n_banks=8)  # 2001Q1..2021Q4
    long_history_frame = _synthetic_long_history_frame()  # 1991Q1..2021Q4
    results = backtest.run_category_backtests(
        dataset, "winsorized_nco_rate", covid_spec="main", long_history_frame=long_history_frame
    )
    for window_result in results.values():
        assert "aggregate_long" in window_result
        assert "anchored_to_aggregate" in window_result
        assert window_result["aggregate_long"]["metrics"]["n"] > 0
        assert window_result["anchored_to_aggregate"]["metrics"]["n"] > 0


def test_run_category_backtests_without_long_history_frame_omits_those_families():
    dataset = _synthetic_category_dataset(n_quarters=84, n_banks=8)
    results = backtest.run_category_backtests(dataset, "winsorized_nco_rate", covid_spec="main")
    for window_result in results.values():
        assert "aggregate_long" not in window_result
        assert "anchored_to_aggregate" not in window_result


def test_add_npl_ratio_columns_adds_both_columns():
    dataset = _synthetic_category_dataset(n_quarters=4, n_banks=2)
    assert "npl_ratio" in dataset.columns
    assert "winsorized_npl_ratio" in dataset.columns


def test_run_category_backtests_covers_both_windows_and_all_three_families():
    dataset = _synthetic_category_dataset(n_quarters=84, n_banks=8)  # 2001Q1..2021Q4
    results = backtest.run_category_backtests(dataset, "winsorized_nco_rate", covid_spec="main")
    assert set(results) == {"2007-2010", "2020-2021"}
    for window_result in results.values():
        assert set(window_result) == {"aggregate_ar", "panel_fe", "gbm"}
        for family_result in window_result.values():
            assert family_result["metrics"]["n"] > 0


def test_run_category_backtests_skips_a_window_with_no_training_data():
    # Real situation this guards against: AUTO has no data before
    # 2011Q1 (a documented gap in schema.py), so the 2006Q4-training/
    # 2007-2010 window has nothing to fit -- it must be OMITTED from the
    # results, not crash the whole sweep with an empty-design-matrix error.
    # covers 2015Q1..2022Q4
    dataset = _synthetic_category_dataset(start="2015Q1", n_quarters=32, n_banks=8)
    results = backtest.run_category_backtests(dataset, "winsorized_nco_rate", covid_spec="main")
    assert "2007-2010" not in results
    assert "2020-2021" in results


def test_run_holdout_backtest_returns_none_with_insufficient_training_data():
    train_dataset = _synthetic_category_dataset(start="2025Q1", n_quarters=4, n_banks=8)
    holdout_dataset = _synthetic_category_dataset(start="2026Q1", n_quarters=2, n_banks=8, seed=2)
    result = backtest.run_holdout_backtest(
        train_dataset, holdout_dataset, "winsorized_nco_rate", covid_spec="main"
    )
    assert result is None


def test_run_holdout_backtest_returns_none_with_empty_holdout():
    train_dataset = _synthetic_category_dataset(start="2019Q1", n_quarters=28, n_banks=8)
    empty_holdout = train_dataset.iloc[0:0]
    result = backtest.run_holdout_backtest(
        train_dataset, empty_holdout, "winsorized_nco_rate", covid_spec="main"
    )
    assert result is None


def test_main_and_robustness_are_identical_for_pre_covid_validation_windows():
    # Documented consequence of the design: neither validation window's
    # TRAINING period reaches 2020-2021, so the robustness treatment
    # (drop 2020-2021 from training) is a no-op for both, and the
    # pandemic dummy (main) is always 0 in both training samples too --
    # the two specs must produce numerically identical backtests here.
    dataset = _synthetic_category_dataset(n_quarters=84, n_banks=8)
    main_results = backtest.run_category_backtests(
        dataset, "winsorized_nco_rate", covid_spec="main"
    )
    robustness_results = backtest.run_category_backtests(
        dataset, "winsorized_nco_rate", covid_spec="robustness"
    )
    for window_label in main_results:
        main_metrics = main_results[window_label]["aggregate_ar"]["metrics"]
        robustness_metrics = robustness_results[window_label]["aggregate_ar"]["metrics"]
        assert main_metrics["rmse"] == pytest.approx(robustness_metrics["rmse"])


def test_run_holdout_backtest_main_and_robustness_can_differ():
    # The holdout fit's training window (through 2025Q4) DOES span
    # 2020-2021 -- this is the one place main vs robustness should be
    # free to disagree (not required to, but the sweep must at least run
    # both without erroring, and it's a real possibility, not always
    # forced to coincide the way the two backtest windows are).
    # covers 2019Q1..2025Q4
    train_dataset = _synthetic_category_dataset(start="2019Q1", n_quarters=28, n_banks=8)
    holdout_dataset = _synthetic_category_dataset(start="2026Q1", n_quarters=2, n_banks=8, seed=2)
    main_result = backtest.run_holdout_backtest(
        train_dataset, holdout_dataset, "winsorized_nco_rate", covid_spec="main"
    )
    robustness_result = backtest.run_holdout_backtest(
        train_dataset, holdout_dataset, "winsorized_nco_rate", covid_spec="robustness"
    )
    assert main_result["aggregate_ar"]["metrics"]["n"] > 0
    assert robustness_result["aggregate_ar"]["metrics"]["n"] > 0
    # robustness trained on fewer rows (2020-2021 dropped) -- coefficients
    # need not be identical (they're allowed to differ; just confirm the
    # pipeline ran both specs distinctly, not that it silently reused one
    # fit for both).
    main_coefs = main_result["aggregate_ar"]["coefficients"]
    robustness_coefs = robustness_result["aggregate_ar"]["coefficients"]
    assert "pandemic" in main_coefs
    assert "pandemic" not in robustness_coefs


# ------------------------------------------------------ backtest summary ---


def test_build_backtest_summary_picks_the_lowest_rmse_family():
    table = pd.DataFrame(
        [
            {
                "category": "commercial_and_industrial",
                "dependent": "nco_rate",
                "covid_spec": "main",
                "window": "2007-2010",
                "model_family": "aggregate_ar",
                "rmse": 0.02,
                "actual_peak_rate": 0.03,
                "predicted_peak_rate": 0.05,
                "peak_rate_error": 0.02,
                "peak_timing_error_quarters": -1,
            },
            {
                "category": "commercial_and_industrial",
                "dependent": "nco_rate",
                "covid_spec": "main",
                "window": "2007-2010",
                "model_family": "panel_fe",
                "rmse": 0.006,
                "actual_peak_rate": 0.03,
                "predicted_peak_rate": 0.018,
                "peak_rate_error": -0.012,
                "peak_timing_error_quarters": -1,
            },
        ]
    )
    summary = backtest.build_backtest_summary(table)
    assert len(summary) == 1
    row = summary.iloc[0]
    assert row["best_model"] == "panel_fe"
    assert row["rmse"] == pytest.approx(0.006)


def test_build_backtest_summary_omits_a_category_with_no_rows_in_the_window():
    table = pd.DataFrame(
        [
            {
                "category": "auto",
                "dependent": "nco_rate",
                "covid_spec": "main",
                "window": "2020-2021",  # not 2007-2010 -- AUTO has no data that far back
                "model_family": "aggregate_ar",
                "rmse": 0.01,
                "actual_peak_rate": 0.02,
                "predicted_peak_rate": 0.02,
                "peak_rate_error": 0.0,
                "peak_timing_error_quarters": 0,
            }
        ]
    )
    summary = backtest.build_backtest_summary(table)
    assert summary.empty


def test_build_backtest_summary_one_row_per_category_dependent_pair():
    table = pd.DataFrame(
        [
            {
                "category": "commercial_and_industrial",
                "dependent": dep,
                "covid_spec": "main",
                "window": "2007-2010",
                "model_family": "aggregate_ar",
                "rmse": 0.01,
                "actual_peak_rate": 0.02,
                "predicted_peak_rate": 0.02,
                "peak_rate_error": 0.0,
                "peak_timing_error_quarters": 0,
            }
            for dep in ("nco_rate", "npl_ratio")
        ]
    )
    summary = backtest.build_backtest_summary(table)
    assert len(summary) == 2
    assert set(summary["dependent"]) == {"nco_rate", "npl_ratio"}
