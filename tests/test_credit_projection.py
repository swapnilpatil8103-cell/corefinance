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


def test_lifetime_expected_loss_rate_averages_the_forward_window():
    quarters = _quarters("2026Q1", 4)
    path = pd.Series([0.02, 0.03, 0.04, 0.05], index=quarters)
    rate = projection.lifetime_expected_loss_rate(path, pd.Period("2026Q1", freq="Q"), 4)
    assert rate == pytest.approx((0.02 + 0.03 + 0.04 + 0.05) / 4)


def test_lifetime_expected_loss_rate_holds_last_value_beyond_the_path():
    quarters = _quarters("2026Q1", 2)
    path = pd.Series([0.02, 0.04], index=quarters)
    # wal_quarters=4 reaches 2 quarters past the path's end -- both held at 0.04
    rate = projection.lifetime_expected_loss_rate(path, pd.Period("2026Q1", freq="Q"), 4)
    assert rate == pytest.approx((0.02 + 0.04 + 0.04 + 0.04) / 4)


def test_lifetime_expected_loss_rate_raises_if_quarter_not_in_path():
    path = pd.Series([0.02], index=_quarters("2026Q1", 1))
    with pytest.raises(ValueError, match="not in"):
        projection.lifetime_expected_loss_rate(path, pd.Period("2027Q1", freq="Q"), 4)


def test_backward_looking_lifetime_expected_loss_rate_averages_the_trailing_window():
    quarters = _quarters("2025Q1", 4)
    path = pd.Series([0.01, 0.02, 0.03, 0.04], index=quarters)
    rate = projection.backward_looking_lifetime_expected_loss_rate(
        path, pd.Period("2025Q4", freq="Q"), 4
    )
    assert rate == pytest.approx((0.01 + 0.02 + 0.03 + 0.04) / 4)


def test_backward_looking_lifetime_expected_loss_rate_uses_whatever_history_exists():
    # Only 2 quarters of real history exist even though wal_quarters=4 --
    # average over what's actually there, don't invent data for the rest.
    quarters = _quarters("2025Q3", 2)
    path = pd.Series([0.02, 0.06], index=quarters)
    rate = projection.backward_looking_lifetime_expected_loss_rate(
        path, pd.Period("2025Q4", freq="Q"), 4
    )
    assert rate == pytest.approx((0.02 + 0.06) / 2)


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
