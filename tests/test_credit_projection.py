"""Offline tests for Stage 5's simplified CECL allowance/provision math
-- synthetic data only, no network."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from corefin.credit import models, projection


def _quarters(start: str, n: int) -> pd.PeriodIndex:
    return pd.period_range(pd.Period(start, freq="Q"), periods=n, freq="Q")


# --------------------------------------------- lifetime expected loss rate ---


def test_lifetime_expected_loss_rate_scales_the_mean_rate_by_wal_years():
    # Lifetime loss = mean ANNUALIZED rate x (wal_quarters / 4), not the
    # mean rate on its own (which is only a 1-year rate) -- a wal of 4
    # quarters is exactly 1 year, so the scale factor is 1.0 here and the
    # scaled/unscaled results coincide; see the wal=8 test below for a
    # case where they don't.
    quarters = _quarters("2026Q1", 4)
    path = pd.Series([0.02, 0.03, 0.04, 0.05], index=quarters)
    rate = projection.lifetime_expected_loss_rate(path, pd.Period("2026Q1", freq="Q"), 4)
    assert rate == pytest.approx((0.02 + 0.03 + 0.04 + 0.05) / 4 * (4 / 4))


def test_lifetime_expected_loss_rate_scales_up_for_a_multi_year_wal():
    # wal_quarters=8 (2 years): the cumulative lifetime rate should be
    # roughly 2x the mean annualized rate, not equal to it.
    quarters = _quarters("2026Q1", 8)
    path = pd.Series([0.04] * 8, index=quarters)
    rate = projection.lifetime_expected_loss_rate(path, pd.Period("2026Q1", freq="Q"), 8)
    assert rate == pytest.approx(0.04 * (8 / 4))


def test_lifetime_expected_loss_rate_holds_last_value_beyond_the_path():
    quarters = _quarters("2026Q1", 2)
    path = pd.Series([0.02, 0.04], index=quarters)
    # wal_quarters=4 reaches 2 quarters past the path's end -- both held at 0.04
    rate = projection.lifetime_expected_loss_rate(path, pd.Period("2026Q1", freq="Q"), 4)
    assert rate == pytest.approx((0.02 + 0.04 + 0.04 + 0.04) / 4 * (4 / 4))


def test_lifetime_expected_loss_rate_is_floored_at_zero():
    quarters = _quarters("2026Q1", 4)
    path = pd.Series([-0.01, -0.02, -0.01, -0.02], index=quarters)
    rate = projection.lifetime_expected_loss_rate(path, pd.Period("2026Q1", freq="Q"), 4)
    assert rate == 0.0


def test_lifetime_expected_loss_rate_raises_if_quarter_not_in_path():
    path = pd.Series([0.02], index=_quarters("2026Q1", 1))
    with pytest.raises(ValueError, match="not in"):
        projection.lifetime_expected_loss_rate(path, pd.Period("2027Q1", freq="Q"), 4)


def test_backward_looking_lifetime_expected_loss_rate_scales_the_mean_rate_by_wal_years():
    quarters = _quarters("2025Q1", 4)
    path = pd.Series([0.01, 0.02, 0.03, 0.04], index=quarters)
    rate = projection.backward_looking_lifetime_expected_loss_rate(
        path, pd.Period("2025Q4", freq="Q"), 4
    )
    assert rate == pytest.approx((0.01 + 0.02 + 0.03 + 0.04) / 4 * (4 / 4))


def test_backward_looking_lifetime_expected_loss_rate_uses_whatever_history_exists():
    # Only 2 quarters of real history exist even though wal_quarters=4 --
    # average over what's actually there, don't invent data for the rest.
    # Scaling still uses the FULL wal_quarters (the lifetime horizon
    # doesn't shrink just because less history happens to be available).
    quarters = _quarters("2025Q3", 2)
    path = pd.Series([0.02, 0.06], index=quarters)
    rate = projection.backward_looking_lifetime_expected_loss_rate(
        path, pd.Period("2025Q4", freq="Q"), 4
    )
    assert rate == pytest.approx((0.02 + 0.06) / 2 * (4 / 4))


def test_backward_looking_lifetime_expected_loss_rate_is_floored_at_zero():
    quarters = _quarters("2025Q1", 4)
    path = pd.Series([-0.02, -0.03, -0.01, -0.02], index=quarters)
    rate = projection.backward_looking_lifetime_expected_loss_rate(
        path, pd.Period("2025Q4", freq="Q"), 4
    )
    assert rate == 0.0


def test_backward_looking_lifetime_expected_loss_rate_raises_with_no_history():
    path = pd.Series([0.02], index=_quarters("2020Q1", 1))
    with pytest.raises(ValueError, match="no realized data"):
        projection.backward_looking_lifetime_expected_loss_rate(
            path, pd.Period("2025Q4", freq="Q"), 4
        )


# --------------------------------------------------------- CECL projection ---


def test_build_cecl_projection_jump_off_row_uses_backward_looking_rate():
    realized = pd.Series([0.01, 0.02, 0.03, 0.04], index=_quarters("2025Q1", 4))
    projected = pd.Series([0.05], index=_quarters("2026Q1", 1))
    result = projection.build_cecl_projection(
        realized, projected, "credit_card", starting_balance=1_000.0
    )
    jump_off = result.loc[pd.Period("2025Q4", freq="Q")]
    wal = projection.CATEGORY_WEIGHTED_AVERAGE_LIFE_QUARTERS["credit_card"]
    expected_rate = projection.backward_looking_lifetime_expected_loss_rate(
        realized, pd.Period("2025Q4", freq="Q"), wal
    )
    assert jump_off["lifetime_expected_loss_rate"] == pytest.approx(expected_rate)
    assert jump_off["allowance_required"] == pytest.approx(1_000.0 * expected_rate)
    assert pd.isna(jump_off["net_charge_off"])
    assert pd.isna(jump_off["provision_expense"])


def test_build_cecl_projection_rejects_projected_path_including_jump_off():
    realized = pd.Series([0.02], index=_quarters("2025Q4", 1))
    projected = pd.Series([0.03], index=_quarters("2025Q4", 1))  # overlaps jump-off
    with pytest.raises(ValueError, match="jump-off"):
        projection.build_cecl_projection(realized, projected, "auto", starting_balance=1_000.0)


def test_build_cecl_projection_provision_follows_the_rollforward_identity():
    # ending = beginning + provision - net_charge_offs
    # => provision = (ending - beginning) + net_charge_offs
    realized = pd.Series([0.02, 0.02, 0.02, 0.02], index=_quarters("2025Q1", 4))
    projected = pd.Series([0.02, 0.02], index=_quarters("2026Q1", 2))  # flat rate throughout
    result = projection.build_cecl_projection(
        realized, projected, "auto", starting_balance=10_000.0
    )
    row_q1 = result.loc[pd.Period("2026Q1", freq="Q")]
    prior_allowance = result.loc[pd.Period("2025Q4", freq="Q"), "allowance_required"]
    expected_provision = (row_q1["allowance_required"] - prior_allowance) + row_q1["net_charge_off"]
    assert row_q1["provision_expense"] == pytest.approx(expected_provision)


def test_build_cecl_projection_net_charge_off_is_quarterly_not_annualized():
    realized = pd.Series([0.02], index=_quarters("2025Q4", 1))
    projected = pd.Series([0.08], index=_quarters("2026Q1", 1))  # 8% annualized
    result = projection.build_cecl_projection(
        realized, projected, "auto", starting_balance=1_000.0
    )
    row = result.loc[pd.Period("2026Q1", freq="Q")]
    assert row["net_charge_off"] == pytest.approx((0.08 / 4.0) * 1_000.0)


def test_build_cecl_projection_allowance_rises_when_projected_losses_rise():
    realized = pd.Series([0.01] * 4, index=_quarters("2025Q1", 4))
    projected_rising = pd.Series([0.02, 0.05, 0.08], index=_quarters("2026Q1", 3))
    result = projection.build_cecl_projection(
        realized, projected_rising, "auto", starting_balance=1_000.0
    )
    allowances = result["allowance_required"]
    assert allowances.iloc[-1] > allowances.iloc[0]


def test_category_weighted_average_life_covers_every_modeling_category():
    from corefin.credit.backtest import MODELING_CATEGORIES

    for category in MODELING_CATEGORIES:
        assert str(category) in projection.CATEGORY_WEIGHTED_AVERAGE_LIFE_QUARTERS
        assert projection.CATEGORY_WEIGHTED_AVERAGE_LIFE_QUARTERS[str(category)] > 0


# ----------------------------------------------------- forward projection ---

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


@pytest.mark.parametrize(
    "family", ["aggregate_ar", "aggregate_long", "panel_fe", "gbm", "anchored_to_aggregate"]
)
def test_project_category_nco_rate_covers_every_scenario_quarter(family):
    macro_history, scenario = _synthetic_macro_history_and_scenario()
    category_train_dataset = _synthetic_category_train_dataset(macro_history)
    from corefin.credit import industry_history

    long_history_frame = industry_history.build_long_industry_frame(
        models.build_industry_series(category_train_dataset, "winsorized_nco_rate").frame.set_index(
            "quarter"
        )["industry_rate"],
        macro_history,
    )
    forecast = projection.project_category_nco_rate(
        _CATEGORY, family, category_train_dataset, macro_history, scenario, long_history_frame
    )
    assert list(forecast.index) == list(scenario.index)
    assert forecast.notna().all()


def test_project_category_nco_rate_truncates_long_history_that_overlaps_the_scenario():
    # Real bug this reproduces: the raw FRED pull behind long_history_frame
    # can extend a quarter or two past the Fed's own historic actuals
    # table (macro_history's own last quarter) -- e.g. 2026Q1-Q2 real FRED
    # data existing even though the Fed's table stops at 2025Q4. Left
    # untruncated, concatenating that extra data with the scenario (which
    # ALSO starts at 2026Q1) duplicates those quarters and corrupts the
    # forecast (each duplicated quarter's "row" becomes a 2-row slice, not
    # a scalar, and result.predict(...).iloc[0] raises instead of a float).
    macro_history, scenario = _synthetic_macro_history_and_scenario()
    category_train_dataset = _synthetic_category_train_dataset(macro_history)
    from corefin.credit import industry_history

    long_rate = models.build_industry_series(
        category_train_dataset, "winsorized_nco_rate"
    ).frame.set_index("quarter")["industry_rate"]
    # extend long_rate 2 quarters PAST macro_history's own last quarter,
    # overlapping scenario's first 2 quarters -- the real situation.
    overlap_quarters = scenario.index[:2]
    extended_long_rate = pd.concat(
        [long_rate, pd.Series([0.5, 0.5], index=overlap_quarters)]
    )
    extended_macro_history_for_long = pd.concat(
        [macro_history, scenario.iloc[:2]]
    )  # so build_long_industry_frame has macro data to join against
    long_history_frame = industry_history.build_long_industry_frame(
        extended_long_rate, extended_macro_history_for_long
    )
    assert long_history_frame["quarter"].max() > macro_history.index.max()  # confirms the overlap

    forecast = projection.project_category_nco_rate(
        _CATEGORY,
        "aggregate_long",
        category_train_dataset,
        macro_history,
        scenario,
        long_history_frame,
    )
    assert list(forecast.index) == list(scenario.index)
    assert forecast.notna().all()


def test_project_category_nco_rate_raises_without_long_history_when_required():
    macro_history, scenario = _synthetic_macro_history_and_scenario()
    category_train_dataset = _synthetic_category_train_dataset(macro_history)
    with pytest.raises(ValueError, match="requires long_history_frame"):
        projection.project_category_nco_rate(
            _CATEGORY, "aggregate_long", category_train_dataset, macro_history, scenario, None
        )


# ------------------------------------------------------------ calibration ---


def test_compute_calibration_gap_computes_model_and_real_ratios():
    check = projection.compute_calibration_gap(
        jump_off_allowance_by_category={"auto": 100.0, "credit_card": 300.0},
        starting_balance_by_category={"auto": 5_000.0, "credit_card": 10_000.0},
        real_total_allowance=450.0,
        real_total_loans=15_000.0,
    )
    assert check.model_total_allowance == pytest.approx(400.0)
    assert check.model_total_loans == pytest.approx(15_000.0)
    assert check.model_allowance_ratio == pytest.approx(400.0 / 15_000.0)
    assert check.real_allowance_ratio == pytest.approx(450.0 / 15_000.0)
    assert check.gap == pytest.approx((400.0 / 15_000.0) - (450.0 / 15_000.0))


def test_cumulative_loss_rate_sums_the_quarterly_flow_over_the_window():
    quarters = _quarters("2026Q1", 4)
    path = pd.Series([0.04, 0.04, 0.08, 0.08], index=quarters)
    rate = projection.cumulative_loss_rate(path, n_quarters=4)
    assert rate == pytest.approx((0.04 / 4) * 2 + (0.08 / 4) * 2)


def test_cumulative_loss_rate_truncates_to_n_quarters():
    quarters = _quarters("2026Q1", 6)
    path = pd.Series([0.04] * 6, index=quarters)
    rate = projection.cumulative_loss_rate(path, n_quarters=3)
    assert rate == pytest.approx((0.04 / 4) * 3)


# ----------------------------------------------------- model selection -----


def _synthetic_category_train_dataset_wrong_signed_hpi(macro_history, n_banks=12, seed=11):
    # Deliberately WRONG-signed (and, given the low noise/large N here,
    # SIGNIFICANT) HPI relationship -- EXPECTED_COEFFICIENT_SIGNS says
    # HPI's coefficient should be negative (rising home prices lower
    # losses); this generator makes losses rise WITH home prices instead.
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
                + 0.01 * macro_history.loc[quarter, "House Price Index YoY % change"]
                + rng.normal(0, 0.0002),
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


def _backtest_table_for(category: str, rmse_by_family: dict[str, float]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "category": category,
                "dependent": "nco_rate",
                "covid_spec": "main",
                "window": "2007-2010",
                "model_family": family,
                "rmse": rmse,
            }
            for family, rmse in rmse_by_family.items()
        ]
    )


def test_select_projection_model_excludes_wrong_signed_significant_families():
    macro_history, _ = _synthetic_macro_history_and_scenario()
    category_train_dataset = _synthetic_category_train_dataset_wrong_signed_hpi(macro_history)
    # aggregate_ar and panel_fe both see the same wrong-signed, strongly
    # significant HPI relationship -- gbm has no coefficients to flag, so
    # it must be selected despite its worse (higher) RMSE.
    backtest_table = _backtest_table_for(
        _CATEGORY, {"aggregate_ar": 0.001, "panel_fe": 0.002, "gbm": 0.003}
    )
    selection = projection.select_projection_model(
        _CATEGORY, category_train_dataset, None, backtest_table
    )
    assert selection.candidates["aggregate_ar"].clean is False
    assert selection.candidates["panel_fe"].clean is False
    assert selection.candidates["gbm"].clean is True
    assert selection.selected_family == "gbm"
    assert selection.is_clean is True


def test_select_projection_model_picks_lowest_rmse_among_clean_candidates():
    macro_history, _ = _synthetic_macro_history_and_scenario()
    category_train_dataset = _synthetic_category_train_dataset(macro_history)
    backtest_table = _backtest_table_for(
        _CATEGORY, {"aggregate_ar": 0.002, "panel_fe": 0.001, "gbm": 0.003}
    )
    selection = projection.select_projection_model(
        _CATEGORY, category_train_dataset, None, backtest_table
    )
    assert selection.candidates["aggregate_ar"].clean is True
    assert selection.candidates["panel_fe"].clean is True
    assert selection.selected_family == "panel_fe"  # lowest RMSE among the clean ones
    assert selection.is_clean is True


def test_select_projection_model_excludes_raw_aggregate_long_for_proxy_categories():
    macro_history, _ = _synthetic_macro_history_and_scenario()
    category_train_dataset = _synthetic_category_train_dataset(macro_history)
    from corefin.credit import industry_history

    long_history_frame = industry_history.build_long_industry_frame(
        models.build_industry_series(category_train_dataset, "winsorized_nco_rate").frame.set_index(
            "quarter"
        )["industry_rate"],
        macro_history,
    )
    backtest_table = _backtest_table_for(
        "cre_construction",
        {"aggregate_ar": 0.01, "panel_fe": 0.01, "gbm": 0.01, "aggregate_long": 0.0001},
    )
    selection = projection.select_projection_model(
        "cre_construction", category_train_dataset, long_history_frame, backtest_table
    )
    assert "aggregate_long" not in selection.candidates
    assert "anchored_to_aggregate" in selection.candidates


def test_select_projection_model_includes_aggregate_long_for_non_proxy_categories():
    macro_history, _ = _synthetic_macro_history_and_scenario()
    category_train_dataset = _synthetic_category_train_dataset(macro_history)
    from corefin.credit import industry_history

    long_history_frame = industry_history.build_long_industry_frame(
        models.build_industry_series(category_train_dataset, "winsorized_nco_rate").frame.set_index(
            "quarter"
        )["industry_rate"],
        macro_history,
    )
    backtest_table = _backtest_table_for(
        "residential_mortgage",
        {"aggregate_ar": 0.01, "panel_fe": 0.01, "gbm": 0.01, "aggregate_long": 0.0001},
    )
    selection = projection.select_projection_model(
        "residential_mortgage", category_train_dataset, long_history_frame, backtest_table
    )
    assert "aggregate_long" in selection.candidates


def test_select_projection_model_falls_back_when_no_family_has_a_backtest_rmse():
    macro_history, _ = _synthetic_macro_history_and_scenario()
    category_train_dataset = _synthetic_category_train_dataset(macro_history)
    empty_backtest_table = pd.DataFrame(
        columns=["category", "dependent", "covid_spec", "window", "model_family", "rmse"]
    )
    selection = projection.select_projection_model(
        _CATEGORY, category_train_dataset, None, empty_backtest_table
    )
    assert selection.selected_family == projection.FALLBACK_MODEL_FAMILY


def test_build_cecl_projection_end_to_end_with_a_projected_path():
    macro_history, scenario = _synthetic_macro_history_and_scenario()
    category_train_dataset = _synthetic_category_train_dataset(macro_history)
    forecast = projection.project_category_nco_rate(
        _CATEGORY, "aggregate_ar", category_train_dataset, macro_history, scenario, None
    )
    realized = models.build_industry_series(category_train_dataset, "winsorized_nco_rate").frame
    realized_rate = realized.set_index("quarter")["industry_rate"]
    last_quarter = category_train_dataset["quarter"].max()
    starting_balance = category_train_dataset[category_train_dataset["quarter"] == last_quarter][
        "average_balance"
    ].sum()
    result = projection.build_cecl_projection(realized_rate, forecast, _CATEGORY, starting_balance)
    assert len(result) == len(scenario) + 1  # + the jump-off row
    assert result["allowance_required"].notna().all()


# ------------------------------------------------------------ crisis replay ---


def test_replay_crisis_window_returns_none_without_crisis_era_data():
    macro_history, _ = _synthetic_macro_history_and_scenario()
    category_train_dataset = _synthetic_category_train_dataset(macro_history)
    # Only quarters from 2011Q1 on -- no real data at all for the
    # 2007Q4-2010Q4 replay window (the real auto/other_consumer situation).
    cutoff = pd.Period("2011Q1", "Q")
    post_2011 = category_train_dataset[category_train_dataset["quarter"] >= cutoff]
    result = projection.replay_crisis_window(_CATEGORY, "aggregate_ar", post_2011, None)
    assert result is None


@pytest.mark.parametrize(
    "family", ["aggregate_ar", "aggregate_long", "panel_fe", "gbm", "anchored_to_aggregate"]
)
def test_replay_crisis_window_computes_projected_and_actual_cumulative_loss(family):
    macro_history, _ = _synthetic_macro_history_and_scenario()
    category_train_dataset = _synthetic_category_train_dataset(macro_history)
    from corefin.credit import industry_history

    long_history_frame = industry_history.build_long_industry_frame(
        models.build_industry_series(category_train_dataset, "winsorized_nco_rate").frame.set_index(
            "quarter"
        )["industry_rate"],
        macro_history,
    )
    result = projection.replay_crisis_window(
        _CATEGORY, family, category_train_dataset, long_history_frame
    )
    assert result is not None
    assert result.category == _CATEGORY
    assert result.family == family
    assert result.projected_9q_cumulative_loss_rate >= 0.0
    assert result.actual_9q_cumulative_loss_rate >= 0.0
    assert result.gap == pytest.approx(
        result.projected_9q_cumulative_loss_rate - result.actual_9q_cumulative_loss_rate
    )


def test_replay_crisis_window_actual_loss_matches_the_real_call_report_series():
    macro_history, _ = _synthetic_macro_history_and_scenario()
    category_train_dataset = _synthetic_category_train_dataset(macro_history)
    result = projection.replay_crisis_window(
        _CATEGORY, "aggregate_ar", category_train_dataset, None
    )
    assert result is not None
    actual_series = models.build_industry_series(
        category_train_dataset, "winsorized_nco_rate"
    ).frame.set_index("quarter")["industry_rate"]
    window = actual_series[
        (actual_series.index >= projection.CRISIS_REPLAY_START)
        & (actual_series.index <= projection.CRISIS_REPLAY_END)
    ]
    expected_actual = projection.cumulative_loss_rate(window, projection.FED_COMPARISON_QUARTERS)
    assert result.actual_9q_cumulative_loss_rate == pytest.approx(expected_actual)
