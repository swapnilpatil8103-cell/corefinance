"""Offline tests for the macro/scenario transformation layer -- no network,
no FRED_API_KEY needed. The transform-methodology tests use real,
hand-verified FRED values (CPIAUCSL, DSPI, UNRATE) cross-checked against
the Fed's own published "Historic Domestic" actuals as regression fixtures
(the numbers are hardcoded here, not re-fetched -- see macro.py's module
docstring for how they were obtained and verified)."""

from __future__ import annotations

import pandas as pd
import pytest

from corefin.credit import macro
from corefin.credit.sources.fred import FredSeriesMapping


def _observations(dates: list[str], values: list[float]) -> pd.DataFrame:
    return pd.DataFrame({"date": pd.to_datetime(dates), "value": values})


# ------------------------------------------------------- aggregation ---


def test_aggregate_fred_series_to_quarterly_averages_monthly_observations():
    observations = _observations(
        ["2025-07-01", "2025-08-01", "2025-09-01"], [322.169, 323.291, 324.245]
    )
    quarterly = macro.aggregate_fred_series_to_quarterly(observations)
    assert quarterly.index[0] == pd.Period("2025Q3", freq="Q")
    assert quarterly.iloc[0] == pytest.approx((322.169 + 323.291 + 324.245) / 3)


def test_aggregate_fred_series_to_quarterly_passes_through_already_quarterly_data():
    observations = _observations(["2025-01-01", "2025-04-01"], [100.0, 105.0])
    quarterly = macro.aggregate_fred_series_to_quarterly(observations)
    assert quarterly.tolist() == [100.0, 105.0]


def test_aggregate_fred_series_to_quarterly_real_unemployment_matches_fed_actual():
    # Real UNRATE for Jul/Aug/Sep 2025: 4.3, 4.3, 4.4 -- the Fed's own
    # published "Unemployment rate" actual for 2025Q3 is 4.3.
    observations = _observations(
        ["2025-07-01", "2025-08-01", "2025-09-01"], [4.3, 4.3, 4.4]
    )
    quarterly = macro.aggregate_fred_series_to_quarterly(observations)
    assert round(quarterly.iloc[0], 1) == 4.3


# ------------------------------------------------------------ transform ---


def test_apply_fred_transform_level_passes_through():
    quarterly = pd.Series([4.0, 4.5], index=pd.PeriodIndex(["2025Q1", "2025Q2"], freq="Q"))
    result = macro.apply_fred_transform(quarterly, "level")
    assert result.tolist() == [4.0, 4.5]


def test_apply_fred_transform_growth_first_observation_is_nan():
    quarterly = pd.Series([100.0, 101.0], index=pd.PeriodIndex(["2025Q1", "2025Q2"], freq="Q"))
    result = macro.apply_fred_transform(quarterly, "qoq_annualized_pct_change")
    assert pd.isna(result.iloc[0])


def test_apply_fred_transform_growth_uses_compounding_not_simple_x4():
    # A 1% quarterly increase compounds to (1.01**4 - 1) * 100 = 4.06%,
    # not the 4.00% a simple x4 would give -- this is the distinguishing
    # synthetic case for the two formulas.
    quarterly = pd.Series([100.0, 101.0], index=pd.PeriodIndex(["2025Q1", "2025Q2"], freq="Q"))
    result = macro.apply_fred_transform(quarterly, "qoq_annualized_pct_change")
    assert result.iloc[1] == pytest.approx(4.0604, abs=1e-3)


def test_apply_fred_transform_rejects_unknown_transform():
    quarterly = pd.Series([1.0], index=pd.PeriodIndex(["2025Q1"], freq="Q"))
    with pytest.raises(ValueError, match="unknown transform"):
        macro.apply_fred_transform(quarterly, "bogus")


def test_apply_fred_transform_growth_matches_real_nominal_disposable_income_2023q1():
    # Real DSPI: Oct/Nov/Dec 2022 = 19428.1/19509.7/19625.4 (Q4 2022 avg);
    # Jan/Feb/Mar 2023 = 20121.6/20289.8/20438.8 (Q1 2023 avg). The Fed's
    # own published "Nominal disposable income growth" actual for 2023Q1
    # is 16.6 -- the compounded formula gives 16.56 (matches to within
    # 0.05 points); a simple x4 formula would give 15.56 (a full point
    # off), which is why compounding is this project's verified choice.
    observations = _observations(
        [
            "2022-10-01",
            "2022-11-01",
            "2022-12-01",
            "2023-01-01",
            "2023-02-01",
            "2023-03-01",
        ],
        [19428.1, 19509.7, 19625.4, 20121.6, 20289.8, 20438.8],
    )
    quarterly = macro.aggregate_fred_series_to_quarterly(observations)
    growth = macro.apply_fred_transform(quarterly, "qoq_annualized_pct_change")
    assert growth.loc[pd.Period("2023Q1", freq="Q")] == pytest.approx(16.6, abs=0.1)


def test_apply_fred_transform_growth_matches_real_cpi_inflation_2024q1():
    # Real CPIAUCSL: Oct/Nov/Dec 2023 = 307.696/308.148/308.741 (Q4 2023
    # avg); Jan/Feb/Mar 2024 = 309.698/310.967/312.345 (Q1 2024 avg). The
    # Fed's own published "CPI inflation rate" actual for 2024Q1 is 3.7 --
    # the compounded formula gives 3.70 (matches almost exactly); a simple
    # x4 formula would give 3.65, further off.
    observations = _observations(
        [
            "2023-10-01",
            "2023-11-01",
            "2023-12-01",
            "2024-01-01",
            "2024-02-01",
            "2024-03-01",
        ],
        [307.696, 308.148, 308.741, 309.698, 310.967, 312.345],
    )
    quarterly = macro.aggregate_fred_series_to_quarterly(observations)
    growth = macro.apply_fred_transform(quarterly, "qoq_annualized_pct_change")
    assert growth.loc[pd.Period("2024Q1", freq="Q")] == pytest.approx(3.7, abs=0.1)


# ---------------------------------------------------- build_fred_history ---


def test_build_fred_history_from_raw_applies_each_variables_own_transform():
    raw = {
        "Unemployment rate": _observations(
            ["2025-01-01", "2025-02-01", "2025-03-01"], [4.0, 4.0, 4.0]
        ),
        "CPI inflation rate": _observations(
            ["2024-10-01", "2024-11-01", "2024-12-01", "2025-01-01", "2025-02-01", "2025-03-01"],
            [100.0, 100.0, 100.0, 101.0, 101.0, 101.0],
        ),
    }
    mappings = {
        "Unemployment rate": FredSeriesMapping("UNRATE", "M", "1948-01-01", "level", True),
        "CPI inflation rate": FredSeriesMapping(
            "CPIAUCSL", "M", "1947-01-01", "qoq_annualized_pct_change", True
        ),
    }
    history = macro.build_fred_history_from_raw(raw, mappings)
    assert history.loc[pd.Period("2025Q1", freq="Q"), "Unemployment rate"] == pytest.approx(4.0)
    assert history.loc[pd.Period("2025Q1", freq="Q"), "CPI inflation rate"] == pytest.approx(
        4.0604, abs=1e-3
    )
    assert pd.isna(history.loc[pd.Period("2024Q4", freq="Q"), "CPI inflation rate"])


# -------------------------------------------------------- scenario ingest ---


def test_normalize_scenario_parses_fed_date_format_and_drops_label_columns():
    raw = pd.DataFrame(
        {
            "Scenario Name": ["Supervisory Baseline", "Supervisory Baseline"],
            "Date": ["2026 Q1", "2026 Q2"],
            "Unemployment rate": [4.6, 4.6],
        }
    )
    normalized = macro.normalize_scenario(raw)
    assert list(normalized.index) == [pd.Period("2026Q1", freq="Q"), pd.Period("2026Q2", freq="Q")]
    assert "Scenario Name" not in normalized.columns
    assert "Date" not in normalized.columns
    assert normalized["Unemployment rate"].tolist() == [4.6, 4.6]


def test_normalize_scenario_strips_the_fed_level_suffix_to_match_fred_history_names():
    # Real gap this project hit: the Fed's own 2026 vintage CSV names four
    # columns with a trailing " (Level)" ("Dow Jones Total Stock Market
    # Index (Level)", "House Price Index (Level)", "Commercial Real
    # Estate Price Index (Level)", "Market Volatility Index (Level)")
    # that fred.FED_SCENARIO_VARIABLE_TO_FRED's keys don't carry --
    # without stripping it, chart-macro/build_full_macro_path can't find
    # the matching history column at all (a real KeyError this test
    # guards against).
    raw = pd.DataFrame(
        {
            "Scenario Name": ["Supervisory Baseline"],
            "Date": ["2026 Q1"],
            "Dow Jones Total Stock Market Index (Level)": [68298.9],
            "House Price Index (Level)": [324.7],
            "Unemployment rate": [4.6],
        }
    )
    normalized = macro.normalize_scenario(raw)
    assert "Dow Jones Total Stock Market Index" in normalized.columns
    assert "House Price Index" in normalized.columns
    assert "Dow Jones Total Stock Market Index (Level)" not in normalized.columns
    assert "Unemployment rate" in normalized.columns


def test_assert_scenario_continues_from_history_passes_with_no_gap():
    history = pd.DataFrame(
        {"Unemployment rate": [4.5, 4.5]},
        index=pd.PeriodIndex(["2025Q3", "2025Q4"], freq="Q"),
    )
    scenario = pd.DataFrame(
        {"Unemployment rate": [4.6]}, index=pd.PeriodIndex(["2026Q1"], freq="Q")
    )
    macro.assert_scenario_continues_from_history(history, scenario)  # must not raise


def test_assert_scenario_continues_from_history_raises_on_a_gap():
    history = pd.DataFrame(
        {"Unemployment rate": [4.5]}, index=pd.PeriodIndex(["2025Q3"], freq="Q")
    )
    scenario = pd.DataFrame(
        {"Unemployment rate": [4.6]}, index=pd.PeriodIndex(["2026Q1"], freq="Q")
    )
    with pytest.raises(ValueError, match="expected"):
        macro.assert_scenario_continues_from_history(history, scenario)


def test_assert_scenario_continues_from_history_raises_on_overlap():
    history = pd.DataFrame(
        {"Unemployment rate": [4.5]}, index=pd.PeriodIndex(["2026Q1"], freq="Q")
    )
    scenario = pd.DataFrame(
        {"Unemployment rate": [4.6]}, index=pd.PeriodIndex(["2026Q1"], freq="Q")
    )
    with pytest.raises(ValueError, match="expected"):
        macro.assert_scenario_continues_from_history(history, scenario)


def test_build_macro_chart_series_labels_actual_and_each_scenario():
    history = pd.DataFrame(
        {"Unemployment rate": [4.5]}, index=pd.PeriodIndex(["2025Q4"], freq="Q")
    )
    baseline = pd.DataFrame(
        {"Unemployment rate": [4.6]}, index=pd.PeriodIndex(["2026Q1"], freq="Q")
    )
    severely_adverse = pd.DataFrame(
        {"Unemployment rate": [5.9]}, index=pd.PeriodIndex(["2026Q1"], freq="Q")
    )
    chart_series = macro.build_macro_chart_series(
        history, {"baseline": baseline, "severely_adverse": severely_adverse}, "Unemployment rate"
    )
    assert set(chart_series["series"]) == {"actual", "baseline", "severely_adverse"}
    # 1 actual row + (1 prepended last-actual point + 1 own row) per scenario
    assert len(chart_series) == 5


def test_build_macro_chart_series_scenario_lines_start_from_the_last_actual():
    # No visual gap at the history/scenario boundary: each scenario's
    # first plotted point must be the last actual quarter's own value,
    # not the scenario's own first (later) quarter in isolation.
    history = pd.DataFrame(
        {"Unemployment rate": [4.4, 4.5]}, index=pd.PeriodIndex(["2025Q3", "2025Q4"], freq="Q")
    )
    baseline = pd.DataFrame(
        {"Unemployment rate": [4.6, 4.7]}, index=pd.PeriodIndex(["2026Q1", "2026Q2"], freq="Q")
    )
    chart_series = macro.build_macro_chart_series(
        history, {"baseline": baseline}, "Unemployment rate"
    )
    baseline_rows = chart_series[chart_series["series"] == "baseline"].reset_index(drop=True)
    assert baseline_rows.loc[0, "quarter"] == pd.Period("2025Q4", freq="Q")
    assert baseline_rows.loc[0, "value"] == pytest.approx(4.5)
    assert baseline_rows.loc[1, "quarter"] == pd.Period("2026Q1", freq="Q")
    assert baseline_rows.loc[1, "value"] == pytest.approx(4.6)


# ---------------------------------------------------- modeling dataset ---


def _synthetic_panel() -> pd.DataFrame:
    quarters = pd.PeriodIndex(["2020Q1", "2020Q2", "2020Q3"], freq="Q")
    return pd.DataFrame(
        {
            "bank_id": [1, 1, 1],
            "quarter": quarters,
            "category": ["auto", "auto", "auto"],
            "average_balance": [10_000.0, 10_000.0, 10_000.0],
            "chargeoff_quarterly": [10.0, 12.0, 14.0],
            "recovery_quarterly": [1.0, 1.0, 1.0],
            "annualized_nco_rate": [0.0036, 0.0044, 0.0052],
            "merger_flag": [False, False, False],
        }
    )


def _synthetic_macro_history() -> pd.DataFrame:
    quarters = pd.PeriodIndex(["2019Q3", "2019Q4", "2020Q1", "2020Q2", "2020Q3"], freq="Q")
    return pd.DataFrame({"Unemployment rate": [3.5, 3.6, 4.4, 13.0, 8.4]}, index=quarters)


def test_build_modeling_dataset_lag_columns_never_look_ahead():
    panel = _synthetic_panel()
    macro_history = _synthetic_macro_history()
    dataset = macro.build_modeling_dataset(panel, macro_history, lags=(0, 1, 2))

    for _, row in dataset.iterrows():
        for lag in (0, 1, 2):
            col = f"Unemployment rate_lag{lag}"
            source_quarter = row["quarter"] - lag
            if source_quarter in macro_history.index:
                expected = macro_history.loc[source_quarter, "Unemployment rate"]
                assert row[col] == pytest.approx(expected)
                assert source_quarter <= row["quarter"]
            else:
                assert pd.isna(row[col])


def test_build_modeling_dataset_lag0_is_contemporaneous_not_future():
    panel = _synthetic_panel()
    macro_history = _synthetic_macro_history()
    dataset = macro.build_modeling_dataset(panel, macro_history, lags=(0,))
    row_2020q2 = dataset[dataset["quarter"] == pd.Period("2020Q2", freq="Q")].iloc[0]
    assert row_2020q2["Unemployment rate_lag0"] == pytest.approx(13.0)


def test_build_modeling_dataset_excludes_merger_flagged_rows():
    panel = _synthetic_panel()
    panel.loc[1, "merger_flag"] = True
    macro_history = _synthetic_macro_history()
    dataset = macro.build_modeling_dataset(panel, macro_history, lags=(0,))
    assert len(dataset) == 2
    assert pd.Period("2020Q2", freq="Q") not in dataset["quarter"].values


def test_build_modeling_dataset_excludes_below_min_balance_rows():
    panel = _synthetic_panel()
    panel.loc[2, "average_balance"] = 500.0
    macro_history = _synthetic_macro_history()
    dataset = macro.build_modeling_dataset(panel, macro_history, lags=(0,), min_balance=1_000.0)
    assert len(dataset) == 2
    assert pd.Period("2020Q3", freq="Q") not in dataset["quarter"].values


def test_build_modeling_dataset_adds_winsorized_nco_rate_column():
    panel = _synthetic_panel()
    macro_history = _synthetic_macro_history()
    dataset = macro.build_modeling_dataset(panel, macro_history, lags=(0,))
    assert "winsorized_nco_rate" in dataset.columns
    assert dataset["winsorized_nco_rate"].notna().all()


# ------------------------------------------------ units consistency ---


def test_fred_history_and_scenario_share_the_same_units_for_every_variable():
    # The (now secondary/optional) FRED path must still produce column
    # names matching a normalized scenario's -- both a FRED-history frame
    # and a normalized scenario frame must use the SAME column names for
    # every shared variable, or `build_full_macro_path`/charts can't
    # compare the right series (this is now checked strictly against real
    # data with the PRIMARY Fed-historic path -- see
    # test_real_fed_historic_and_scenario_adjoin_exactly_with_matching_columns).
    raw_scenario = pd.DataFrame(
        {
            "Scenario Name": ["Supervisory Baseline"],
            "Date": ["2026 Q1"],
            "Unemployment rate": [4.6],
            "CPI inflation rate": [3.0],
        }
    )
    scenario = macro.normalize_scenario(raw_scenario)
    # Synthetic illustrative values (this test checks column/unit
    # consistency, not a real-data numeric match -- see the
    # growth-transform tests above for those).
    raw = {
        "Unemployment rate": _observations(
            ["2025-10-01", "2025-11-01", "2025-12-01"], [4.4, 4.5, 4.4]
        ),
        "CPI inflation rate": _observations(
            ["2025-07-01", "2025-08-01", "2025-09-01", "2025-10-01", "2025-11-01", "2025-12-01"],
            [322.0, 323.0, 324.0, 324.5, 325.0, 326.0],
        ),
    }
    mappings = {
        "Unemployment rate": FredSeriesMapping("UNRATE", "M", "1948-01-01", "level", True),
        "CPI inflation rate": FredSeriesMapping(
            "CPIAUCSL", "M", "1947-01-01", "qoq_annualized_pct_change", True
        ),
    }
    history = macro.build_fred_history_from_raw(raw, mappings)
    assert set(scenario.columns) <= set(history.columns)
    # both express "Unemployment rate" as a level in percentage points
    # (not e.g. history in decimal fraction and scenario in percent)
    assert 0 < history["Unemployment rate"].iloc[-1] < 20
    assert 0 < scenario["Unemployment rate"].iloc[0] < 20


def test_real_fed_historic_and_scenario_adjoin_exactly_with_matching_columns():
    # Regression guard for the fix that replaced FRED-derived history with
    # the Fed's own historic domestic actuals as the PRIMARY source (see
    # this module's docstring): a real problem with the FRED-derived
    # approach was that four variables were on a completely different
    # numeric scale from the Fed's own scenario values (House Price Index,
    # the Dow Jones Total Stock Market Index proxy, Commercial Real Estate
    # Price Index, BBB corporate yield). Since history and scenario now
    # come from the SAME Fed file family, this replaces the old "same
    # order of magnitude" heuristic with a STRICT check: they must adjoin
    # with no gap/overlap at all (not just "close enough"), and share
    # identical variable columns for every one of the 16 Fed variables.
    from pathlib import Path

    historic_path = Path("data/raw/fed_scenarios/2026_Final_Historic_Domestic.csv")
    baseline_path = Path("data/raw/fed_scenarios/2026_Final_Supervisory_Baseline_Domestic.csv")
    if not historic_path.exists() or not baseline_path.exists():
        pytest.skip("real cached Fed historic/scenario CSVs not present locally")

    history = macro.normalize_fed_historic(pd.read_csv(historic_path))
    scenario = macro.normalize_scenario(pd.read_csv(baseline_path))

    macro.assert_scenario_continues_from_history(history, scenario)  # must not raise
    assert set(history.columns) == set(scenario.columns)
    assert len(history.columns) == 16
    assert history.index.max() == pd.Period("2025Q4", freq="Q")
    assert scenario.index.min() == pd.Period("2026Q1", freq="Q")


# --------------------------------------------------- Fed historic table ---


def test_normalize_fed_historic_is_normalize_scenario_under_a_clearer_name():
    raw = pd.DataFrame(
        {
            "Scenario Name": ["Actual", "Actual"],
            "Date": ["2025 Q3", "2025 Q4"],
            "Unemployment rate": [4.3, 4.5],
        }
    )
    historic = macro.normalize_fed_historic(raw)
    assert list(historic.index) == [pd.Period("2025Q3", freq="Q"), pd.Period("2025Q4", freq="Q")]
    assert historic["Unemployment rate"].tolist() == [4.3, 4.5]


def test_real_fed_historic_table_covers_all_16_variables_through_2025q4():
    from pathlib import Path

    historic_path = Path("data/raw/fed_scenarios/2026_Final_Historic_Domestic.csv")
    if not historic_path.exists():
        pytest.skip("real cached Fed historic CSV not present locally")
    history = macro.normalize_fed_historic(pd.read_csv(historic_path))
    assert len(history.columns) == 16
    assert history.index.max() == pd.Period("2025Q4", freq="Q")
    assert history.index.min() == pd.Period("1976Q1", freq="Q")


# ------------------------------------------------------ full macro path ---


def test_build_full_macro_path_concatenates_and_validates_continuity():
    history = pd.DataFrame(
        {"Unemployment rate": [4.5, 4.4]}, index=pd.PeriodIndex(["2025Q3", "2025Q4"], freq="Q")
    )
    scenario = pd.DataFrame(
        {"Unemployment rate": [4.6, 4.7]}, index=pd.PeriodIndex(["2026Q1", "2026Q2"], freq="Q")
    )
    full_path = macro.build_full_macro_path(history, scenario)
    assert list(full_path.index) == [
        pd.Period("2025Q3", freq="Q"),
        pd.Period("2025Q4", freq="Q"),
        pd.Period("2026Q1", freq="Q"),
        pd.Period("2026Q2", freq="Q"),
    ]
    assert full_path["Unemployment rate"].tolist() == [4.5, 4.4, 4.6, 4.7]


def test_build_full_macro_path_raises_on_a_gap():
    history = pd.DataFrame(
        {"Unemployment rate": [4.5]}, index=pd.PeriodIndex(["2025Q3"], freq="Q")
    )
    scenario = pd.DataFrame(
        {"Unemployment rate": [4.6]}, index=pd.PeriodIndex(["2026Q1"], freq="Q")
    )
    with pytest.raises(ValueError, match="expected"):
        macro.build_full_macro_path(history, scenario)


# ------------------------------------------------------ pct-change features ---


def test_add_pct_change_features_computes_qoq_and_yoy():
    levels = pd.DataFrame(
        {"House Price Index": [100.0, 102.0, 104.0, 106.0, 110.0]},
        index=pd.PeriodIndex(["2024Q4", "2025Q1", "2025Q2", "2025Q3", "2025Q4"], freq="Q"),
    )
    result = macro.add_pct_change_features(levels, variables=("House Price Index",))
    assert result["House Price Index QoQ % change"].iloc[-1] == pytest.approx(
        (110.0 / 106.0 - 1.0) * 100.0
    )
    assert result["House Price Index YoY % change"].iloc[-1] == pytest.approx(
        (110.0 / 100.0 - 1.0) * 100.0
    )
    # first observation has no prior quarter to compare against
    assert pd.isna(result["House Price Index QoQ % change"].iloc[0])
    assert pd.isna(result["House Price Index YoY % change"].iloc[0])


def test_add_pct_change_features_keeps_the_raw_level_column():
    levels = pd.DataFrame(
        {"House Price Index": [100.0, 102.0]}, index=pd.PeriodIndex(["2025Q3", "2025Q4"], freq="Q")
    )
    result = macro.add_pct_change_features(levels, variables=("House Price Index",))
    assert result["House Price Index"].tolist() == [100.0, 102.0]


def test_add_pct_change_features_skips_a_variable_not_present():
    levels = pd.DataFrame(
        {"Unemployment rate": [4.5]}, index=pd.PeriodIndex(["2025Q4"], freq="Q")
    )
    result = macro.add_pct_change_features(levels, variables=("House Price Index",))
    assert "House Price Index QoQ % change" not in result.columns


def test_add_pct_change_features_on_full_path_uses_real_prior_history_for_scenario_rows():
    # A scenario's own early rows need real levels from BEFORE the jump-off
    # quarter to compute a meaningful YoY change -- computed on the
    # scenario alone, the first 4 rows would be NaN; computed on the
    # concatenated full path, they use real history instead.
    history = pd.DataFrame(
        {"House Price Index": [100.0, 101.0, 102.0, 103.0, 104.0]},
        index=pd.PeriodIndex(["2024Q4", "2025Q1", "2025Q2", "2025Q3", "2025Q4"], freq="Q"),
    )
    scenario = pd.DataFrame(
        {"House Price Index": [106.0]}, index=pd.PeriodIndex(["2026Q1"], freq="Q")
    )
    full_path = macro.build_full_macro_path(history, scenario)
    with_changes = macro.add_pct_change_features(full_path, variables=("House Price Index",))
    scenario_row = with_changes.loc[pd.Period("2026Q1", freq="Q")]
    assert scenario_row["House Price Index QoQ % change"] == pytest.approx(
        (106.0 / 104.0 - 1.0) * 100.0
    )
    assert scenario_row["House Price Index YoY % change"] == pytest.approx(
        (106.0 / 101.0 - 1.0) * 100.0
    )


# --------------------------------------------------- train/holdout split ---


def test_split_dataset_by_quarter_separates_train_and_holdout():
    dataset = pd.DataFrame(
        {
            "quarter": pd.PeriodIndex(["2025Q3", "2025Q4", "2026Q1", "2026Q2"], freq="Q"),
            "value": [1, 2, 3, 4],
        }
    )
    train, holdout = macro.split_dataset_by_quarter(dataset, pd.Period("2025Q4", freq="Q"))
    assert train["quarter"].tolist() == [
        pd.Period("2025Q3", freq="Q"),
        pd.Period("2025Q4", freq="Q"),
    ]
    assert holdout["quarter"].tolist() == [
        pd.Period("2026Q1", freq="Q"),
        pd.Period("2026Q2", freq="Q"),
    ]
