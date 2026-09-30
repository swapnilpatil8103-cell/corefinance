"""Offline tests for the long-history (FRED-derived) industry series --
synthetic data only, no network."""

from __future__ import annotations

import pandas as pd
import pytest

from corefin.credit import industry_history, models


def _observations(dates: list[str], values: list[float]) -> pd.DataFrame:
    return pd.DataFrame({"date": pd.to_datetime(dates), "value": values})


def test_build_long_industry_rate_converts_percent_to_decimal_fraction():
    raw = _observations(["1991-01-01", "1991-04-01"], [5.80, 6.30])
    rate = industry_history.build_long_industry_rate(raw)
    assert rate.loc[pd.Period("1991Q1", freq="Q")] == pytest.approx(0.058)
    assert rate.loc[pd.Period("1991Q2", freq="Q")] == pytest.approx(0.063)


def test_build_long_industry_rate_indexes_by_quarter():
    raw = _observations(["2008-07-01"], [1.11])
    rate = industry_history.build_long_industry_rate(raw)
    assert list(rate.index) == [pd.Period("2008Q3", freq="Q")]


def _synthetic_macro_history() -> pd.DataFrame:
    quarters = pd.period_range(pd.Period("1990Q4", freq="Q"), periods=10, freq="Q")
    return pd.DataFrame(
        {
            "Unemployment rate": [5.0 + 0.1 * i for i in range(10)],
            "House Price Index YoY % change": [2.0 - 0.2 * i for i in range(10)],
            "Commercial Real Estate Price Index YoY % change": [1.0 + 0.1 * i for i in range(10)],
            "Dow Jones Total Stock Market Index YoY % change": [4.0 - 0.3 * i for i in range(10)],
        },
        index=quarters,
    )


def test_build_long_industry_frame_has_the_same_shape_as_build_industry_series():
    long_rate = pd.Series(
        [0.01, 0.012, 0.014, 0.02],
        index=pd.period_range(pd.Period("1991Q1", freq="Q"), periods=4, freq="Q"),
        name="industry_rate",
    )
    macro_history = _synthetic_macro_history()
    frame = industry_history.build_long_industry_frame(long_rate, macro_history)

    assert set(frame.columns) == {
        "quarter",
        "industry_rate",
        "industry_rate_lag1",
        "pandemic",
        *[models.feature_column(v) for v in models.CORE_MACRO_FEATURES],
    }
    assert frame["industry_rate"].tolist() == pytest.approx([0.01, 0.012, 0.014, 0.02])
    assert pd.isna(frame["industry_rate_lag1"].iloc[0])
    assert frame["industry_rate_lag1"].iloc[1] == pytest.approx(0.01)


def test_build_long_industry_frame_is_directly_usable_by_fit_aggregate_model():
    # No-leakage / integration check: the frame this module builds must
    # work unchanged with models.fit_aggregate_model and
    # models.forecast_aggregate_dynamic -- the whole point of matching
    # build_industry_series's shape exactly.
    quarters = pd.period_range(pd.Period("1991Q1", freq="Q"), periods=20, freq="Q")
    long_rate = pd.Series(
        [0.01 + 0.001 * i for i in range(20)], index=quarters, name="industry_rate"
    )
    macro_quarters = pd.period_range(pd.Period("1990Q4", freq="Q"), periods=21, freq="Q")
    macro_history = pd.DataFrame(
        {
            "Unemployment rate": [5.0 + 0.05 * i for i in range(21)],
            "House Price Index YoY % change": [2.0 - 0.05 * i for i in range(21)],
            "Commercial Real Estate Price Index YoY % change": [1.0 + 0.02 * i for i in range(21)],
            "Dow Jones Total Stock Market Index YoY % change": [4.0 - 0.03 * i for i in range(21)],
        },
        index=macro_quarters,
    )
    frame = industry_history.build_long_industry_frame(long_rate, macro_history)
    train = frame[frame["quarter"] <= pd.Period("1994Q4", freq="Q")]
    result = models.fit_aggregate_model(
        train, "commercial_and_industrial", include_pandemic_dummy=False
    )
    forecast = models.forecast_aggregate_dynamic(
        result,
        frame,
        "commercial_and_industrial",
        pd.Period("1995Q1", freq="Q"),
        pd.Period("1995Q4", freq="Q"),
        include_pandemic_dummy=False,
    )
    assert len(forecast) == 4
    assert forecast.notna().all()


def test_compare_with_call_report_aggregate_computes_the_difference_on_overlap_only():
    long_frame = pd.DataFrame(
        {
            "quarter": pd.PeriodIndex(["2001Q1", "2001Q2", "2001Q3"], freq="Q"),
            "industry_rate": [0.01, 0.02, 0.03],
        }
    )
    call_report_frame = pd.DataFrame(
        {
            "quarter": pd.PeriodIndex(["2001Q2", "2001Q3", "2001Q4"], freq="Q"),
            "industry_rate": [0.021, 0.028, 0.05],
        }
    )
    comparison = industry_history.compare_with_call_report_aggregate(long_frame, call_report_frame)
    expected_quarters = {pd.Period("2001Q2", freq="Q"), pd.Period("2001Q3", freq="Q")}
    assert set(comparison["quarter"]) == expected_quarters
    row = comparison.set_index("quarter").loc[pd.Period("2001Q2", freq="Q")]
    assert row["difference"] == pytest.approx(0.02 - 0.021)
