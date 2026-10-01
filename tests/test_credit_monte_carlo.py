"""Offline tests for Stage 6's Monte Carlo loss-distribution simulation
-- synthetic data only, no network."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from corefin.credit import industry_history, models, monte_carlo

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


# ------------------------------------------------------- macro shocks ---


def test_estimate_macro_shock_std_computes_real_qoq_std():
    quarters = pd.period_range("2020Q1", periods=5, freq="Q")
    history = pd.DataFrame({"Unemployment rate": [4.0, 4.5, 5.5, 5.0, 4.8]}, index=quarters)
    stds = monte_carlo.estimate_macro_shock_std(history, variables=("Unemployment rate",))
    expected = pd.Series([4.0, 4.5, 5.5, 5.0, 4.8]).diff().dropna().std()
    assert stds["Unemployment rate"] == pytest.approx(expected)


def test_estimate_macro_shock_std_skips_missing_variable():
    history = pd.DataFrame(
        {"Unemployment rate": [4.0, 4.5]}, index=pd.period_range("2020Q1", periods=2, freq="Q")
    )
    stds = monte_carlo.estimate_macro_shock_std(history, variables=("House Price Index",))
    assert stds == {}


def test_perturb_scenario_adds_noise_with_the_right_scale():
    quarters = pd.period_range("2026Q1", periods=2_000, freq="Q")
    scenario = pd.DataFrame({"Unemployment rate": [5.0] * len(quarters)}, index=quarters)
    rng = np.random.default_rng(0)
    perturbed = monte_carlo.perturb_scenario(scenario, {"Unemployment rate": 0.3}, rng)
    deviations = perturbed["Unemployment rate"] - 5.0
    assert deviations.std() == pytest.approx(0.3, rel=0.1)
    assert deviations.mean() == pytest.approx(0.0, abs=0.05)


def test_perturb_scenario_is_deterministic_given_a_seeded_rng():
    quarters = pd.period_range("2026Q1", periods=4, freq="Q")
    scenario = pd.DataFrame({"Unemployment rate": [5.0] * 4}, index=quarters)
    a = monte_carlo.perturb_scenario(scenario, {"Unemployment rate": 0.3}, np.random.default_rng(1))
    b = monte_carlo.perturb_scenario(scenario, {"Unemployment rate": 0.3}, np.random.default_rng(1))
    assert a["Unemployment rate"].tolist() == b["Unemployment rate"].tolist()


def test_perturb_scenario_leaves_zero_std_variables_unchanged():
    quarters = pd.period_range("2026Q1", periods=4, freq="Q")
    scenario = pd.DataFrame({"Unemployment rate": [5.0] * 4}, index=quarters)
    rng = np.random.default_rng(2)
    perturbed = monte_carlo.perturb_scenario(scenario, {"Unemployment rate": 0.0}, rng)
    assert perturbed["Unemployment rate"].tolist() == [5.0] * 4


# -------------------------------------------------------- model fitting ---


def test_fit_monte_carlo_model_raises_without_long_history_when_required():
    macro_history, _ = _synthetic_macro_history_and_scenario()
    category_train_dataset = _synthetic_category_train_dataset(macro_history)
    with pytest.raises(ValueError, match="requires long_history_frame"):
        monte_carlo.fit_monte_carlo_model(_CATEGORY, "aggregate_long", category_train_dataset, None)


@pytest.mark.parametrize(
    "family", ["aggregate_ar", "aggregate_long", "panel_fe", "gbm", "anchored_to_aggregate"]
)
def test_fit_monte_carlo_model_produces_a_nonnegative_residual_std(family):
    macro_history, _ = _synthetic_macro_history_and_scenario()
    category_train_dataset = _synthetic_category_train_dataset(macro_history)
    long_history_frame = _long_history_frame(macro_history, category_train_dataset)
    mc_model = monte_carlo.fit_monte_carlo_model(
        _CATEGORY, family, category_train_dataset, long_history_frame
    )
    assert mc_model.category == _CATEGORY
    assert mc_model.family == family
    assert mc_model.residual_std >= 0.0


# --------------------------------------------------------- simulation ---


def test_simulate_category_losses_returns_n_draws_and_varies():
    macro_history, scenario = _synthetic_macro_history_and_scenario()
    category_train_dataset = _synthetic_category_train_dataset(macro_history)
    mc_model = monte_carlo.fit_monte_carlo_model(
        _CATEGORY, "aggregate_ar", category_train_dataset, None
    )
    shock_std = monte_carlo.estimate_macro_shock_std(macro_history)
    losses = monte_carlo.simulate_category_losses(
        mc_model, macro_history, scenario, shock_std, n_draws=50, random_state=0
    )
    assert losses.shape == (50,)
    assert losses.std() > 0.0  # draws actually differ -- randomness is doing something


def test_simulate_category_losses_is_reproducible_with_same_random_state():
    macro_history, scenario = _synthetic_macro_history_and_scenario()
    category_train_dataset = _synthetic_category_train_dataset(macro_history)
    mc_model = monte_carlo.fit_monte_carlo_model(
        _CATEGORY, "aggregate_ar", category_train_dataset, None
    )
    shock_std = monte_carlo.estimate_macro_shock_std(macro_history)
    a = monte_carlo.simulate_category_losses(
        mc_model, macro_history, scenario, shock_std, n_draws=20, random_state=5
    )
    b = monte_carlo.simulate_category_losses(
        mc_model, macro_history, scenario, shock_std, n_draws=20, random_state=5
    )
    assert np.array_equal(a, b)


def test_simulate_category_losses_respects_n_quarters_truncation():
    macro_history, scenario = _synthetic_macro_history_and_scenario()
    category_train_dataset = _synthetic_category_train_dataset(macro_history)
    mc_model = monte_carlo.fit_monte_carlo_model(
        _CATEGORY, "aggregate_ar", category_train_dataset, None
    )
    shock_std = monte_carlo.estimate_macro_shock_std(macro_history)
    full = monte_carlo.simulate_category_losses(
        mc_model, macro_history, scenario, shock_std, n_draws=10, n_quarters=13, random_state=3
    )
    nine_q = monte_carlo.simulate_category_losses(
        mc_model, macro_history, scenario, shock_std, n_draws=10, n_quarters=9, random_state=3
    )
    # same draws (same random_state/sequence of rng calls up to the truncation point),
    # but a shorter cumulative window -> smaller magnitude on average.
    assert abs(nine_q).mean() <= abs(full).mean()


# --------------------------------------------------------- summarizing ---


def test_summarize_loss_distribution_percentiles_are_ordered():
    losses = np.array([0.01, 0.02, 0.03, 0.04, 0.05, 0.10, 0.15, 0.20, 0.5, 1.0])
    summary = monte_carlo.summarize_loss_distribution(_CATEGORY, "severely_adverse", "gbm", losses)
    assert summary.category == _CATEGORY
    assert summary.n_draws == len(losses)
    ordered = [summary.percentiles[p] for p in monte_carlo.MONTE_CARLO_PERCENTILES]
    assert ordered == sorted(ordered)
    assert summary.mean == pytest.approx(losses.mean())
    assert summary.std == pytest.approx(losses.std())
