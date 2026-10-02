"""Offline tests for the survivorship-bias anchoring sensitivity --
synthetic data only, no network."""

from __future__ import annotations

import numpy as np
import pandas as pd

from corefin.credit import anchoring_sensitivity, industry_history, models, projection

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
        # bank 0 "fails" after the crisis window -- never appears again,
        # so it's absent from today's (last actual quarter) population,
        # exercising the survivorship-drop path.
        bank_quarters = (
            quarters
            if bank_idx != 0
            else [q for q in quarters if q <= pd.Period("2010Q4", freq="Q")]
        )
        for quarter in bank_quarters:
            balance = base_balance * (1.0 + rng.normal(0, 0.02))
            true_nco = max(
                0.01
                + 0.0015 * macro_history.loc[quarter, "Unemployment rate"]
                - 0.001 * macro_history.loc[quarter, "House Price Index YoY % change"]
                + (0.02 if bank_idx == 0 else 0.0)  # bank 0 is riskier while it existed
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


def test_industry_average_variant_matches_raw_aggregate_long_forecast():
    macro_history, scenario = _synthetic_macro_history_and_scenario()
    category_train_dataset = _synthetic_category_train_dataset(macro_history)
    long_history_frame = _long_history_frame(macro_history, category_train_dataset)

    variants = anchoring_sensitivity.compute_anchoring_variants(
        _CATEGORY, category_train_dataset, macro_history, scenario, long_history_frame
    )
    expected = projection.project_category_nco_rate(
        _CATEGORY,
        "aggregate_long",
        category_train_dataset,
        macro_history,
        scenario,
        long_history_frame,
    )
    pd.testing.assert_series_equal(
        variants["industry_average"], expected, check_names=False
    )


def test_full_sample_survivors_variant_matches_production_anchored_forecast():
    macro_history, scenario = _synthetic_macro_history_and_scenario()
    category_train_dataset = _synthetic_category_train_dataset(macro_history)
    long_history_frame = _long_history_frame(macro_history, category_train_dataset)

    variants = anchoring_sensitivity.compute_anchoring_variants(
        _CATEGORY, category_train_dataset, macro_history, scenario, long_history_frame
    )
    expected = projection.project_category_nco_rate(
        _CATEGORY,
        "anchored_to_aggregate",
        category_train_dataset,
        macro_history,
        scenario,
        long_history_frame,
    )
    pd.testing.assert_series_equal(
        variants["full_sample_survivors"], expected, check_names=False
    )


def test_crisis_era_survivors_variant_differs_from_full_sample_variant():
    # bank 0 was deliberately riskier DURING the crisis than its
    # (nonexistent, since it fails after 2010) full-sample average would
    # be -- but bank 0 isn't in the "last actual quarter" population
    # either way, so what should differ here is the SURVIVING banks'
    # own crisis-era-only relative levels vs. their full-sample ones.
    macro_history, scenario = _synthetic_macro_history_and_scenario()
    category_train_dataset = _synthetic_category_train_dataset(macro_history)
    long_history_frame = _long_history_frame(macro_history, category_train_dataset)

    variants = anchoring_sensitivity.compute_anchoring_variants(
        _CATEGORY, category_train_dataset, macro_history, scenario, long_history_frame
    )
    assert not variants["crisis_era_survivors"].equals(variants["full_sample_survivors"])


def test_crisis_era_survivors_variant_drops_banks_absent_from_the_crisis_window():
    # A bank chartered only AFTER the crisis window has no crisis-era
    # relative level and must be excluded from THIS variant only.
    macro_history, scenario = _synthetic_macro_history_and_scenario()
    category_train_dataset = _synthetic_category_train_dataset(macro_history, n_banks=10)
    new_bank_rows = category_train_dataset[
        category_train_dataset["quarter"] > pd.Period("2015Q4", freq="Q")
    ].copy()
    new_bank_rows["bank_id"] = 99
    category_train_dataset = pd.concat(
        [category_train_dataset, new_bank_rows], ignore_index=True
    )
    long_history_frame = _long_history_frame(macro_history, category_train_dataset)

    long_result_forecast = anchoring_sensitivity.project_with_crisis_era_anchoring(
        _CATEGORY, category_train_dataset, macro_history, scenario, long_history_frame
    )
    assert list(long_result_forecast.index) == list(scenario.index)
    assert long_result_forecast.notna().all()


def test_anchoring_variants_cover_all_documented_variants():
    macro_history, scenario = _synthetic_macro_history_and_scenario()
    category_train_dataset = _synthetic_category_train_dataset(macro_history)
    long_history_frame = _long_history_frame(macro_history, category_train_dataset)

    variants = anchoring_sensitivity.compute_anchoring_variants(
        _CATEGORY, category_train_dataset, macro_history, scenario, long_history_frame
    )
    assert set(variants) == set(anchoring_sensitivity.ANCHORING_VARIANTS)
    for forecast in variants.values():
        assert list(forecast.index) == list(scenario.index)
        assert forecast.notna().all()
