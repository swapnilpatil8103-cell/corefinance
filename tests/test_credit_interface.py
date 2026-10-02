"""Offline tests for Stage 7's typed CreditLossProjection interface --
synthetic data only, no network."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from corefin.credit import industry_history, interface, models

_CATEGORIES = ["commercial_and_industrial", "credit_card"]


def _synthetic_macro_history_and_scenario():
    history_quarters = pd.period_range(pd.Period("2001Q1", freq="Q"), periods=60, freq="Q")
    scenario_quarters = pd.period_range(pd.Period("2016Q1", freq="Q"), periods=13, freq="Q")
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


def _synthetic_training_dataset(macro_history: pd.DataFrame, n_banks=8, seed=9):
    rng = np.random.default_rng(seed)
    quarters = macro_history.index
    rows = []
    for category in _CATEGORIES:
        for bank_idx in range(n_banks):
            base_balance = rng.uniform(5_000.0, 50_000.0)
            for quarter in quarters:
                balance = base_balance * (1.0 + rng.normal(0, 0.02))
                nonaccrual = balance * max(rng.normal(0.02, 0.005), 0.0)
                past_due_90 = balance * max(rng.normal(0.01, 0.003), 0.0)
                true_nco = max(
                    0.01
                    + 0.0015 * macro_history.loc[quarter, "Unemployment rate"]
                    - 0.001 * macro_history.loc[quarter, "House Price Index YoY % change"]
                    + rng.normal(0, 0.0004),
                    -0.05,
                )
                rows.append(
                    {
                        "bank_id": f"bank{bank_idx}",
                        "quarter": quarter,
                        "category": category,
                        "average_balance": balance,
                        "balance": balance,
                        "nonaccrual": nonaccrual,
                        "past_due_90": past_due_90,
                        "winsorized_nco_rate": true_nco,
                        models.feature_column("Unemployment rate"): macro_history.loc[
                            quarter, "Unemployment rate"
                        ],
                        models.feature_column(
                            "House Price Index YoY % change"
                        ): macro_history.loc[quarter, "House Price Index YoY % change"],
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


def test_build_credit_loss_projection_shapes_and_timeline():
    macro_history, scenario = _synthetic_macro_history_and_scenario()
    training = _synthetic_training_dataset(macro_history)
    long_history_frames = {
        (category, "nco_rate"): _long_history_frame(
            macro_history, training[training["category"] == category]
        )
        for category in _CATEGORIES
    }
    selected_family_by_category = {
        "commercial_and_industrial": "aggregate_ar",
        "credit_card": "panel_fe",
    }

    result = interface.build_credit_loss_projection(
        _CATEGORIES,
        selected_family_by_category,
        training,
        macro_history,
        scenario,
        "severely_adverse",
        long_history_frames,
        n_monte_carlo_draws=10,
    )

    n_categories = len(_CATEGORIES)
    n_periods = len(scenario) + 1  # + jump-off
    assert result.timeline.n_periods == n_periods
    assert result.timeline.n_projection_periods == len(scenario)
    assert not result.timeline.is_projection[0]
    assert result.timeline.is_projection[1:].all()
    assert result.categories == tuple(_CATEGORIES)
    assert result.scenario_name == "severely_adverse"
    assert result.bank_identifier is None

    for array_name in (
        "balance_mm",
        "net_charge_off_mm",
        "provision_expense_mm",
        "allowance_mm",
        "npl_mm",
        "nco_rate",
        "npl_ratio",
    ):
        assert getattr(result, array_name).shape == (n_categories, n_periods)

    assert result.monte_carlo_mean.shape == (n_categories,)
    for percentile_array in result.monte_carlo_percentiles.values():
        assert percentile_array.shape == (n_categories,)


def test_build_credit_loss_projection_totals_sum_across_categories():
    macro_history, scenario = _synthetic_macro_history_and_scenario()
    training = _synthetic_training_dataset(macro_history)
    long_history_frames = {
        (category, "nco_rate"): _long_history_frame(
            macro_history, training[training["category"] == category]
        )
        for category in _CATEGORIES
    }
    selected_family_by_category = {
        "commercial_and_industrial": "aggregate_ar",
        "credit_card": "panel_fe",
    }

    result = interface.build_credit_loss_projection(
        _CATEGORIES,
        selected_family_by_category,
        training,
        macro_history,
        scenario,
        "severely_adverse",
        long_history_frames,
        n_monte_carlo_draws=10,
    )
    assert result.balance_total_mm == pytest.approx(result.balance_mm.sum(axis=0))
    assert result.allowance_total_mm == pytest.approx(result.allowance_mm.sum(axis=0))
    # balance in $mm must be a real, positive scale-down of the raw
    # (thousands-of-dollars) balance -- not accidentally left unconverted.
    assert (result.balance_mm > 0).all()
    assert (result.balance_mm < 1_000).all()  # synthetic balances are a few $mm at most


def test_build_credit_loss_projection_npl_ratio_is_held_flat_at_jump_off():
    macro_history, scenario = _synthetic_macro_history_and_scenario()
    training = _synthetic_training_dataset(macro_history)
    long_history_frames = {
        (category, "nco_rate"): _long_history_frame(
            macro_history, training[training["category"] == category]
        )
        for category in _CATEGORIES
    }
    result = interface.build_credit_loss_projection(
        _CATEGORIES,
        {"commercial_and_industrial": "aggregate_ar", "credit_card": "panel_fe"},
        training,
        macro_history,
        scenario,
        "baseline",
        long_history_frames,
        n_monte_carlo_draws=10,
    )
    for i in range(len(_CATEGORIES)):
        assert np.allclose(result.npl_ratio[i], result.npl_ratio[i, 0])


def test_build_credit_loss_projection_single_bank_mode_skips_monte_carlo():
    macro_history, scenario = _synthetic_macro_history_and_scenario()
    training = _synthetic_training_dataset(macro_history)
    long_history_frames = {
        (category, "nco_rate"): _long_history_frame(
            macro_history, training[training["category"] == category]
        )
        for category in _CATEGORIES
    }
    result = interface.build_credit_loss_projection(
        _CATEGORIES,
        {"commercial_and_industrial": "aggregate_ar", "credit_card": "panel_fe"},
        training,
        macro_history,
        scenario,
        "severely_adverse",
        long_history_frames,
        bank_id="bank0",
    )
    assert result.bank_identifier == "bank0"
    assert np.isnan(result.monte_carlo_mean).all()
    assert (result.balance_mm > 0).all()


def test_credit_loss_projection_rejects_mismatched_array_shapes():
    from corefin.timeline import Timeline

    timeline = Timeline.quarterly(n_periods=3, n_historical=1, start_year=2025, start_quarter=4)
    good_shape = np.zeros((1, 3))
    with pytest.raises(ValueError, match="expected"):
        interface.CreditLossProjection(
            timeline=timeline,
            categories=("credit_card",),
            scenario_name="baseline",
            bank_identifier=None,
            balance_mm=np.zeros((2, 3)),  # wrong n_categories
            net_charge_off_mm=good_shape,
            provision_expense_mm=good_shape,
            allowance_mm=good_shape,
            npl_mm=good_shape,
            nco_rate=good_shape,
            npl_ratio=good_shape,
            monte_carlo_mean=np.zeros(1),
            monte_carlo_percentiles={5: np.zeros(1)},
        )
