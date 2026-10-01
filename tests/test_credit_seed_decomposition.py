"""Offline tests for the seed-vs-macro decomposition diagnostic --
synthetic data only, no network."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from corefin.credit import industry_history, models, projection, seed_decomposition

_CATEGORY = "commercial_and_industrial"


def _synthetic_macro_history_and_scenario():
    history_quarters = pd.period_range(pd.Period("2001Q1", freq="Q"), periods=100, freq="Q")
    scenario_quarters = pd.period_range(pd.Period("2026Q1", freq="Q"), periods=13, freq="Q")
    n_history = len(history_quarters)
    n_scenario = len(scenario_quarters)

    def _series(n, seed_offset):
        r = np.random.default_rng(7 + seed_offset)
        return 5.0 + np.cumsum(r.normal(0, 0.2, size=n))

    history = pd.DataFrame(
        {
            "Unemployment rate": _series(n_history, 1),
            "House Price Index YoY % change": _series(n_history, 2),
            "Commercial Real Estate Price Index YoY % change": _series(n_history, 3),
            "Dow Jones Total Stock Market Index YoY % change": _series(n_history, 4),
        },
        index=history_quarters,
    )
    scenario = pd.DataFrame(
        {
            "Unemployment rate": _series(n_scenario, 5) + 3,
            "House Price Index YoY % change": _series(n_scenario, 6) - 2,
            "Commercial Real Estate Price Index YoY % change": _series(n_scenario, 7) - 2,
            "Dow Jones Total Stock Market Index YoY % change": _series(n_scenario, 8) - 5,
        },
        index=scenario_quarters,
    )
    return history, scenario


def _synthetic_category_train_dataset(macro_history: pd.DataFrame, n_banks=10, seed=9):
    rng = np.random.default_rng(seed)
    quarters = macro_history.index
    rows = []
    for bank_idx in range(n_banks):
        base_balance = rng.uniform(5_000.0, 50_000.0)
        for quarter in quarters:
            balance = base_balance * (1.0 + rng.normal(0, 0.02))
            true_nco = max(
                0.01
                + 0.0015 * macro_history.loc[quarter, "Unemployment rate"]
                - 0.001 * macro_history.loc[quarter, "House Price Index YoY % change"]
                + rng.normal(0, 0.0004),
                -0.05,
            )
            rows.append(
                {
                    "bank_id": bank_idx,
                    "quarter": quarter,
                    "category": _CATEGORY,
                    "average_balance": balance,
                    "winsorized_nco_rate": true_nco,
                    models.feature_column("Unemployment rate"): macro_history.loc[
                        quarter, "Unemployment rate"
                    ],
                    models.feature_column("House Price Index YoY % change"): macro_history.loc[
                        quarter, "House Price Index YoY % change"
                    ],
                    models.feature_column(
                        "Commercial Real Estate Price Index YoY % change"
                    ): macro_history.loc[
                        quarter, "Commercial Real Estate Price Index YoY % change"
                    ],
                    models.feature_column(
                        "Dow Jones Total Stock Market Index YoY % change"
                    ): macro_history.loc[
                        quarter, "Dow Jones Total Stock Market Index YoY % change"
                    ],
                }
            )
    return pd.DataFrame(rows)


def _long_history_frame(macro_history, category_train_dataset):
    return industry_history.build_long_industry_frame(
        models.build_industry_series(category_train_dataset, "winsorized_nco_rate").frame.set_index(
            "quarter"
        )["industry_rate"],
        macro_history,
    )


def test_replay_with_seed_matches_replay_crisis_window_with_no_override():
    macro_history, _ = _synthetic_macro_history_and_scenario()
    category_train_dataset = _synthetic_category_train_dataset(macro_history)
    forecast = seed_decomposition.replay_with_seed(
        _CATEGORY, "aggregate_ar", category_train_dataset, None
    )
    reference = projection.replay_crisis_window(
        _CATEGORY, "aggregate_ar", category_train_dataset, None
    )
    assert reference is not None
    expected_9q = projection.cumulative_loss_rate(forecast, projection.FED_COMPARISON_QUARTERS)
    assert expected_9q == pytest.approx(reference.projected_9q_cumulative_loss_rate)


def test_replay_with_seed_override_changes_only_the_starting_point():
    macro_history, _ = _synthetic_macro_history_and_scenario()
    category_train_dataset = _synthetic_category_train_dataset(macro_history)
    natural = seed_decomposition.replay_with_seed(
        _CATEGORY, "aggregate_ar", category_train_dataset, None
    )
    swapped = seed_decomposition.replay_with_seed(
        _CATEGORY, "aggregate_ar", category_train_dataset, None, seed_override=0.5
    )
    # a wildly different seed must change the forecast path
    assert not natural.equals(swapped)
    assert swapped.iloc[0] != pytest.approx(natural.iloc[0])


def test_replay_with_seed_rejects_families_without_an_ar_term():
    macro_history, _ = _synthetic_macro_history_and_scenario()
    category_train_dataset = _synthetic_category_train_dataset(macro_history)
    with pytest.raises(ValueError, match="no AR term"):
        seed_decomposition.replay_with_seed(_CATEGORY, "panel_fe", category_train_dataset, None)


def test_scenario_forecast_with_seed_matches_project_category_nco_rate_with_no_override():
    macro_history, scenario = _synthetic_macro_history_and_scenario()
    category_train_dataset = _synthetic_category_train_dataset(macro_history)
    forecast = seed_decomposition.scenario_forecast_with_seed(
        _CATEGORY, "aggregate_ar", category_train_dataset, macro_history, scenario, None
    )
    reference = projection.project_category_nco_rate(
        _CATEGORY, "aggregate_ar", category_train_dataset, macro_history, scenario, None
    )
    assert forecast.equals(reference)


def test_scenario_forecast_with_seed_override_changes_only_the_starting_point():
    macro_history, scenario = _synthetic_macro_history_and_scenario()
    category_train_dataset = _synthetic_category_train_dataset(macro_history)
    natural = seed_decomposition.scenario_forecast_with_seed(
        _CATEGORY, "aggregate_ar", category_train_dataset, macro_history, scenario, None
    )
    swapped = seed_decomposition.scenario_forecast_with_seed(
        _CATEGORY,
        "aggregate_ar",
        category_train_dataset,
        macro_history,
        scenario,
        None,
        seed_override=0.5,
    )
    assert not natural.equals(swapped)


def test_ar_seed_series_uses_long_history_for_anchored_families():
    macro_history, _ = _synthetic_macro_history_and_scenario()
    category_train_dataset = _synthetic_category_train_dataset(macro_history)
    long_history_frame = _long_history_frame(macro_history, category_train_dataset)
    seed_series = seed_decomposition.ar_seed_series(
        "anchored_to_aggregate", category_train_dataset, long_history_frame
    )
    expected = long_history_frame.set_index("quarter")["industry_rate"]
    assert seed_series.equals(expected)


def test_ar_seed_series_uses_call_report_series_for_aggregate_ar():
    macro_history, _ = _synthetic_macro_history_and_scenario()
    category_train_dataset = _synthetic_category_train_dataset(macro_history)
    seed_series = seed_decomposition.ar_seed_series("aggregate_ar", category_train_dataset, None)
    expected = models.build_industry_series(
        category_train_dataset, "winsorized_nco_rate"
    ).frame.set_index("quarter")["industry_rate"]
    assert seed_series.equals(expected)


def test_peak_quarter_and_rate_finds_the_maximum():
    quarters = pd.period_range("2026Q1", periods=4, freq="Q")
    path = pd.Series([0.01, 0.05, 0.03, 0.02], index=quarters)
    peak_quarter, peak_rate = seed_decomposition.peak_quarter_and_rate(path)
    assert peak_quarter == pd.Period("2026Q2", freq="Q")
    assert peak_rate == pytest.approx(0.05)


def test_decompose_seed_vs_macro_returns_all_four_variants():
    macro_history, scenario = _synthetic_macro_history_and_scenario()
    category_train_dataset = _synthetic_category_train_dataset(macro_history)
    result = seed_decomposition.decompose_seed_vs_macro(
        _CATEGORY, "aggregate_ar", category_train_dataset, macro_history, scenario, None
    )
    assert result.category == _CATEGORY
    assert result.family == "aggregate_ar"
    assert list(result.replay_natural.index) == list(
        pd.period_range(projection.CRISIS_REPLAY_START, projection.CRISIS_REPLAY_END, freq="Q")
    )
    assert list(result.scenario_natural.index) == list(scenario.index)
    assert not result.replay_natural.equals(result.replay_swapped)
    assert not result.scenario_natural.equals(result.scenario_swapped)


def test_macro_driver_table_aligns_both_paths_by_position():
    macro_history, scenario = _synthetic_macro_history_and_scenario()
    table = seed_decomposition.macro_driver_table(macro_history, scenario)
    assert len(table) == 13
    assert list(table["position"]) == list(range(1, 14))
    for feature in seed_decomposition.CORE_DRIVER_FEATURES:
        assert f"replay_{feature}" in table.columns
        assert f"scenario_{feature}" in table.columns
    # the replay path's first lagged quarter is the real quarter before
    # the crisis window starts (2007Q3), the scenario's is 2025Q4.
    assert table.iloc[0]["replay_quarter"] == str(projection.CRISIS_REPLAY_START)
    assert table.iloc[0]["scenario_quarter"] == str(scenario.index[0])
