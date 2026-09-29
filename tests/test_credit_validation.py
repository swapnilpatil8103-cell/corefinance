"""Offline tests for the out-of-time backtest harness -- synthetic data
only, no network."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from corefin.credit import validation


def _quarters(labels: list[str]) -> pd.PeriodIndex:
    return pd.PeriodIndex(labels, freq="Q")


# ------------------------------------------------------------- split ---


def test_split_out_of_time_train_is_at_or_before_cutoff():
    dataset = pd.DataFrame(
        {"quarter": _quarters(["2005Q4", "2006Q1", "2006Q2", "2006Q3", "2006Q4", "2007Q1"])}
    )
    train, _ = validation.split_out_of_time(
        dataset,
        train_end=pd.Period("2006Q4", freq="Q"),
        test_start=pd.Period("2007Q1", freq="Q"),
        test_end=pd.Period("2007Q1", freq="Q"),
    )
    assert train["quarter"].max() == pd.Period("2006Q4", freq="Q")
    assert (train["quarter"] <= pd.Period("2006Q4", freq="Q")).all()


def test_split_out_of_time_test_is_within_the_window_only():
    dataset = pd.DataFrame(
        {"quarter": _quarters(["2006Q4", "2007Q1", "2008Q2", "2010Q4", "2011Q1"])}
    )
    _, test = validation.split_out_of_time(
        dataset,
        train_end=pd.Period("2006Q4", freq="Q"),
        test_start=pd.Period("2007Q1", freq="Q"),
        test_end=pd.Period("2010Q4", freq="Q"),
    )
    assert set(test["quarter"]) == {
        pd.Period("2007Q1", freq="Q"),
        pd.Period("2008Q2", freq="Q"),
        pd.Period("2010Q4", freq="Q"),
    }


def test_split_out_of_time_no_row_appears_in_both_train_and_test():
    labels = [f"20{y:02d}Q{q}" for y in range(1, 20) for q in (1, 2, 3, 4)]
    dataset = pd.DataFrame({"quarter": _quarters(labels)})
    train, test = validation.split_out_of_time(
        dataset,
        train_end=pd.Period("2006Q4", freq="Q"),
        test_start=pd.Period("2007Q1", freq="Q"),
        test_end=pd.Period("2010Q4", freq="Q"),
    )
    assert set(train.index).isdisjoint(set(test.index))


# ---------------------------------------------------- forecast error ---


def test_forecast_error_metrics_zero_when_predictions_are_exact():
    quarters = _quarters(["2007Q1", "2007Q2", "2007Q3"])
    actual = pd.Series([0.01, 0.02, 0.03], index=quarters)
    predicted = pd.Series([0.01, 0.02, 0.03], index=quarters)
    metrics = validation.forecast_error_metrics(actual, predicted)
    assert metrics["rmse"] == pytest.approx(0.0)
    assert metrics["mae"] == pytest.approx(0.0)
    assert metrics["n"] == 3


def test_forecast_error_metrics_known_value():
    quarters = _quarters(["2007Q1", "2007Q2"])
    actual = pd.Series([0.01, 0.03], index=quarters)
    predicted = pd.Series([0.02, 0.01], index=quarters)
    metrics = validation.forecast_error_metrics(actual, predicted)
    # errors: -0.01, +0.02 -> RMSE = sqrt((0.0001 + 0.0004)/2), MAE = 0.015
    assert metrics["rmse"] == pytest.approx(np.sqrt(0.00025))
    assert metrics["mae"] == pytest.approx(0.015)


def test_forecast_error_metrics_only_uses_overlapping_quarters():
    actual = pd.Series([0.01, 0.02], index=_quarters(["2007Q1", "2007Q2"]))
    predicted = pd.Series([0.01], index=_quarters(["2007Q1"]))  # missing 2007Q2
    metrics = validation.forecast_error_metrics(actual, predicted)
    assert metrics["n"] == 1
    assert metrics["rmse"] == pytest.approx(0.0)


def test_forecast_error_metrics_empty_overlap_returns_nan():
    actual = pd.Series([0.01], index=_quarters(["2007Q1"]))
    predicted = pd.Series([0.01], index=_quarters(["2008Q1"]))
    metrics = validation.forecast_error_metrics(actual, predicted)
    assert metrics["n"] == 0
    assert pd.isna(metrics["rmse"])
    assert pd.isna(metrics["mae"])


# --------------------------------------------------------- peak metrics ---


def test_peak_metrics_identifies_the_correct_peak_quarter_and_value():
    quarters = _quarters(["2007Q1", "2008Q2", "2009Q1", "2010Q4"])
    actual = pd.Series([0.01, 0.02, 0.09, 0.03], index=quarters)  # peak: 2009Q1
    predicted = pd.Series([0.01, 0.02, 0.09, 0.03], index=quarters)
    metrics = validation.peak_metrics(actual, predicted)
    assert metrics["actual_peak_rate"] == pytest.approx(0.09)
    assert metrics["predicted_peak_rate"] == pytest.approx(0.09)
    assert metrics["peak_rate_error"] == pytest.approx(0.0)
    assert metrics["peak_timing_error_quarters"] == 0


def test_peak_metrics_detects_a_later_predicted_peak():
    quarters = _quarters(["2007Q1", "2008Q1", "2009Q1", "2010Q1"])
    actual = pd.Series([0.01, 0.02, 0.09, 0.03], index=quarters)  # actual peak: 2009Q1
    predicted = pd.Series([0.01, 0.02, 0.03, 0.09], index=quarters)  # predicted peak: 2010Q1
    metrics = validation.peak_metrics(actual, predicted)
    assert metrics["peak_timing_error_quarters"] == 4  # 1 year late
    assert metrics["peak_rate_error"] == pytest.approx(0.0)  # same peak magnitude


def test_peak_metrics_detects_an_understated_peak_magnitude():
    quarters = _quarters(["2009Q1", "2009Q2"])
    actual = pd.Series([0.10, 0.02], index=quarters)
    predicted = pd.Series([0.05, 0.02], index=quarters)
    metrics = validation.peak_metrics(actual, predicted)
    assert metrics["peak_rate_error"] == pytest.approx(-0.05)  # understated by 0.05
    assert metrics["peak_timing_error_quarters"] == 0


def test_peak_metrics_empty_overlap_returns_nan():
    actual = pd.Series([0.01], index=_quarters(["2007Q1"]))
    predicted = pd.Series([0.01], index=_quarters(["2008Q1"]))
    metrics = validation.peak_metrics(actual, predicted)
    assert pd.isna(metrics["peak_rate_error"])
    assert pd.isna(metrics["peak_timing_error_quarters"])


# -------------------------------------------------------- summary ---


def test_backtest_summary_merges_both_metric_sets():
    quarters = _quarters(["2007Q1", "2007Q2"])
    actual = pd.Series([0.01, 0.05], index=quarters)
    predicted = pd.Series([0.02, 0.04], index=quarters)
    summary = validation.backtest_summary(actual, predicted)
    expected_keys = {
        "rmse",
        "mae",
        "n",
        "actual_peak_rate",
        "predicted_peak_rate",
        "peak_rate_error",
    }
    assert expected_keys <= set(summary)
