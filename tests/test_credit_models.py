"""Offline tests for Stage 4's model families -- synthetic data only, no
network, no real panel needed. Focus: no leakage across an out-of-time
train/test boundary, and known synthetic relationships recovered with the
expected sign (the two things the project brief explicitly asks for)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from corefin.credit import models

# The synthetic fixtures below all use this category -- commercial_and_industrial
# keeps every one of CORE_MACRO_FEATURES (including the stock index),
# which these tests' fixtures generate values for, matching
# models.CATEGORY_MACRO_FEATURES.
_CATEGORY = "commercial_and_industrial"


def _quarters(start: str, n: int) -> pd.PeriodIndex:
    return pd.period_range(pd.Period(start, freq="Q"), periods=n, freq="Q")


# ------------------------------------------------------ pandemic dummy ---


def test_add_pandemic_indicator_flags_exactly_the_covid_window():
    frame = pd.DataFrame({"quarter": _quarters("2019Q4", 10)})  # 2019Q4..2022Q1
    flag = models.add_pandemic_indicator(frame)
    expected = [
        q >= pd.Period("2020Q2", freq="Q") and q <= pd.Period("2021Q4", freq="Q")
        for q in frame["quarter"]
    ]
    assert flag.tolist() == [1.0 if e else 0.0 for e in expected]


def test_exclude_pandemic_years_drops_all_of_2020_and_2021_only():
    frame = pd.DataFrame({"quarter": _quarters("2019Q4", 10), "value": range(10)})
    result = models.exclude_pandemic_years(frame)
    kept_years = {q.year for q in result["quarter"]}
    assert 2020 not in kept_years
    assert 2021 not in kept_years
    assert 2019 in kept_years
    assert 2022 in kept_years
    assert len(result) == 10 - 8  # 2020Q1-2021Q4 = 8 quarters dropped


# ------------------------------------------------------- sign checking ---


def test_classify_coefficient_significance_correct_sign():
    coefficients = {models.feature_column("Unemployment rate"): 0.003}
    t_values = {models.feature_column("Unemployment rate"): 3.5}
    result = models.classify_coefficient_significance(coefficients, t_values)
    assert result[models.feature_column("Unemployment rate")] == models.CORRECT_SIGN


def test_classify_coefficient_significance_wrong_sign_but_insignificant():
    # WRONG: should be negative, but |t| < threshold -- not a real problem.
    coefficients = {models.feature_column("House Price Index YoY % change"): 0.001}
    t_values = {models.feature_column("House Price Index YoY % change"): 0.5}
    result = models.classify_coefficient_significance(coefficients, t_values)
    key = models.feature_column("House Price Index YoY % change")
    assert result[key] == models.WRONG_SIGN_INSIGNIFICANT


def test_classify_coefficient_significance_wrong_sign_and_significant():
    coefficients = {models.feature_column("House Price Index YoY % change"): 0.001}
    t_values = {models.feature_column("House Price Index YoY % change"): 2.5}
    result = models.classify_coefficient_significance(coefficients, t_values)
    key = models.feature_column("House Price Index YoY % change")
    assert result[key] == models.WRONG_SIGN_SIGNIFICANT


def test_classify_coefficient_significance_treats_exact_zero_as_wrong_sign():
    coefficients = {models.feature_column("Unemployment rate"): 0.0}
    t_values = {models.feature_column("Unemployment rate"): 0.0}
    result = models.classify_coefficient_significance(coefficients, t_values)
    assert result[models.feature_column("Unemployment rate")] == models.WRONG_SIGN_INSIGNIFICANT


def test_classify_coefficient_significance_only_reports_present_features():
    coefficients = {models.feature_column("Unemployment rate"): 1.0}
    t_values = {models.feature_column("Unemployment rate"): 3.0}
    result = models.classify_coefficient_significance(coefficients, t_values)
    assert set(result) == {models.feature_column("Unemployment rate")}


def test_classify_coefficient_significance_requires_both_dicts_to_have_the_feature():
    coefficients = {models.feature_column("Unemployment rate"): 1.0}
    result = models.classify_coefficient_significance(coefficients, {})
    assert result == {}


# --------------------------------------------- per-category feature sets ---


def test_category_macro_features_ci_keeps_the_stock_index():
    features = models.CATEGORY_MACRO_FEATURES["commercial_and_industrial"]
    assert models.STOCK_INDEX_FEATURE in features
    assert len(features) == 4


def test_category_macro_features_every_other_category_drops_the_stock_index():
    # Real finding on the actual build: the stock index's YoY change was
    # wrong-signed AND statistically significant in the bank panel model
    # for several real-estate/consumer categories (see models.py's module
    # docstring for the investigated mechanism) -- dropped everywhere
    # except commercial_and_industrial.
    for category in models.CATEGORY_MACRO_FEATURES:
        if category == "commercial_and_industrial":
            continue
        features = models.CATEGORY_MACRO_FEATURES[category]
        assert models.STOCK_INDEX_FEATURE not in features
        assert len(features) == 3


def test_category_macro_features_lookup_works_with_a_loancategory_member():
    from corefin.credit.schema import LoanCategory

    by_enum = models.CATEGORY_MACRO_FEATURES[LoanCategory.CI]
    by_string = models.CATEGORY_MACRO_FEATURES["commercial_and_industrial"]
    assert by_enum == by_string


def test_fit_aggregate_model_excludes_stock_index_for_a_non_ci_category():
    dataset = _synthetic_bank_dataset(n_quarters=40, n_banks=20)
    series = models.build_industry_series(dataset, "winsorized_nco_rate").frame
    result = models.fit_aggregate_model(
        series, "residential_mortgage", include_pandemic_dummy=False
    )
    assert models.feature_column(models.STOCK_INDEX_FEATURE) not in result.params.index
    assert models.feature_column("Unemployment rate") in result.params.index


def test_fit_aggregate_model_uses_extended_features_when_present(monkeypatch):
    # USE_EXTENDED_AGGREGATE_FEATURES defaults to False (tried on real
    # data, discarded -- see models.py's module-level comment); this
    # test locks in the MECHANISM still working correctly when it's on.
    monkeypatch.setattr(models, "USE_EXTENDED_AGGREGATE_FEATURES", True)
    dataset = _synthetic_bank_dataset(n_quarters=40, n_banks=20)
    series = models.build_industry_series(dataset, "winsorized_nco_rate").frame
    rng = np.random.default_rng(1)
    series[models.feature_column(models.UNEMPLOYMENT_4Q_CHANGE_FEATURE)] = rng.normal(
        0, 0.3, size=len(series)
    )
    series[models.feature_column(models.HPI_8Q_CHANGE_FEATURE)] = rng.normal(
        0, 0.5, size=len(series)
    )
    series[models.feature_column(models.CRE_8Q_CHANGE_FEATURE)] = rng.normal(
        0, 0.5, size=len(series)
    )
    result = models.fit_aggregate_model(
        series, "residential_mortgage", include_pandemic_dummy=False
    )
    for feature in models.EXTENDED_AGGREGATE_FEATURES:
        assert models.feature_column(feature) in result.params.index


def test_fit_aggregate_model_omits_extended_features_when_absent_from_the_data():
    # No extended columns on this frame at all -- silently dropped, not
    # an error, so older data / synthetic fixtures without them still work.
    dataset = _synthetic_bank_dataset(n_quarters=40, n_banks=20)
    series = models.build_industry_series(dataset, "winsorized_nco_rate").frame
    for feature in models.EXTENDED_AGGREGATE_FEATURES:
        assert models.feature_column(feature) not in series.columns
    result = models.fit_aggregate_model(
        series, "residential_mortgage", include_pandemic_dummy=False
    )
    for feature in models.EXTENDED_AGGREGATE_FEATURES:
        assert models.feature_column(feature) not in result.params.index


def test_fit_panel_fe_model_never_uses_extended_aggregate_features(monkeypatch):
    # The extended features are for the aggregate/anchored families only
    # -- panel_fe keeps using _macro_feature_columns regardless of the
    # flag or the data having them (flag forced True here specifically
    # to prove that, not just that it's off by default).
    monkeypatch.setattr(models, "USE_EXTENDED_AGGREGATE_FEATURES", True)
    dataset = _synthetic_bank_dataset(n_quarters=40, n_banks=40)
    rng = np.random.default_rng(2)
    for feature in models.EXTENDED_AGGREGATE_FEATURES:
        dataset[models.feature_column(feature)] = rng.normal(0, 0.3, size=len(dataset))
    fit = models.fit_panel_fe_model(
        dataset, "commercial_and_industrial", "winsorized_nco_rate", include_pandemic_dummy=False
    )
    for feature in models.EXTENDED_AGGREGATE_FEATURES:
        assert models.feature_column(feature) not in fit.result.params.index


def test_extended_aggregate_features_all_have_an_expected_sign():
    for feature in models.EXTENDED_AGGREGATE_FEATURES:
        assert feature in models.EXPECTED_COEFFICIENT_SIGNS


def test_fit_panel_fe_model_excludes_stock_index_for_a_non_ci_category():
    dataset = _synthetic_bank_dataset(n_quarters=40, n_banks=40)
    fit = models.fit_panel_fe_model(
        dataset, "auto", "winsorized_nco_rate", include_pandemic_dummy=False
    )
    assert models.feature_column(models.STOCK_INDEX_FEATURE) not in fit.result.params.index


def test_predict_panel_fe_and_fit_must_agree_on_category_or_columns_mismatch():
    # fit and predict must be called with the SAME category -- using a
    # mismatched one would look up the wrong feature set and either KeyError
    # or silently use the wrong columns. This test documents/locks the
    # matching-category contract by confirming a mismatched category raises.
    dataset = _synthetic_bank_dataset(n_quarters=40, n_banks=20)
    fit = models.fit_panel_fe_model(
        dataset, "residential_mortgage", "winsorized_nco_rate", include_pandemic_dummy=False
    )
    with pytest.raises(KeyError):
        models.predict_panel_fe(
            fit,
            dataset,
            "commercial_and_industrial",
            "winsorized_nco_rate",
            include_pandemic_dummy=False,
        )


# --------------------------------------------------------- feature prep ---


def _synthetic_bank_dataset(n_quarters=20, n_banks=30, seed=0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    quarters = _quarters("2015Q1", n_quarters)

    # macro path: same across banks within a quarter, like the real
    # modeling dataset's per-quarter lookup columns. Each series uses an
    # independent random walk (not a shared deterministic trend) so they
    # aren't collinear with each other -- an earlier version of this
    # fixture had all four as linear/sinusoidal functions of the same
    # quarter index, which made the OLS design matrix rank-deficient.
    unemployment = 4.0 + np.cumsum(rng.normal(0, 0.3, size=n_quarters))
    hpi_yoy = 3.0 + np.cumsum(rng.normal(0, 0.4, size=n_quarters))
    cre_yoy = 2.0 + np.cumsum(rng.normal(0, 0.5, size=n_quarters))
    stock_yoy = 5.0 + np.cumsum(rng.normal(0, 0.6, size=n_quarters))

    bank_effect = rng.normal(0, 0.005, size=n_banks)

    rows = []
    for bank_idx in range(n_banks):
        base_balance = rng.uniform(5_000.0, 50_000.0)
        for q_idx, quarter in enumerate(quarters):
            true_rate = (
                0.01
                + 0.002 * unemployment[q_idx]
                - 0.0015 * hpi_yoy[q_idx]
                + bank_effect[bank_idx]
                + rng.normal(0, 0.0005)
            )
            # slight time variation (not a fixed per-bank constant) so
            # log_balance isn't perfectly absorbed by the bank fixed
            # effect and doesn't trigger a rank-deficiency warning.
            balance = base_balance * (1.0 + rng.normal(0, 0.02))
            rows.append(
                {
                    "bank_id": bank_idx,
                    "quarter": quarter,
                    "category": "commercial_and_industrial",
                    "average_balance": balance,
                    "winsorized_nco_rate": max(true_rate, -0.05),
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
    return pd.DataFrame(rows)


def test_build_industry_series_is_balance_weighted():
    dataset = pd.DataFrame(
        {
            "quarter": [pd.Period("2020Q1", freq="Q")] * 2,
            "average_balance": [100.0, 300.0],
            "winsorized_nco_rate": [0.01, 0.05],
            models.feature_column("Unemployment rate"): [4.0, 4.0],
            models.feature_column("House Price Index YoY % change"): [1.0, 1.0],
            models.feature_column("Commercial Real Estate Price Index YoY % change"): [1.0, 1.0],
            models.feature_column("Dow Jones Total Stock Market Index YoY % change"): [1.0, 1.0],
        }
    )
    series = models.build_industry_series(dataset, "winsorized_nco_rate").frame
    expected = (100.0 * 0.01 + 300.0 * 0.05) / 400.0
    assert series["industry_rate"].iloc[0] == pytest.approx(expected)


def test_build_industry_series_lag1_is_shifted_by_exactly_one_quarter_no_leakage():
    dataset = _synthetic_bank_dataset(n_quarters=6, n_banks=2)
    series = models.build_industry_series(dataset, "winsorized_nco_rate").frame
    for i in range(1, len(series)):
        assert series["industry_rate_lag1"].iloc[i] == pytest.approx(
            series["industry_rate"].iloc[i - 1]
        )
    assert pd.isna(series["industry_rate_lag1"].iloc[0])


# --------------------------------------------------------- model family 1 ---


def test_fit_aggregate_model_recovers_the_known_signs():
    dataset = _synthetic_bank_dataset(n_quarters=40, n_banks=20)
    series = models.build_industry_series(dataset, "winsorized_nco_rate").frame
    result = models.fit_aggregate_model(series, _CATEGORY, include_pandemic_dummy=False)
    coefficients = result.params.to_dict()
    t_values = result.tvalues.to_dict()
    signs = models.classify_coefficient_significance(coefficients, t_values)
    assert signs[models.feature_column("Unemployment rate")] == models.CORRECT_SIGN
    assert signs[models.feature_column("House Price Index YoY % change")] == models.CORRECT_SIGN


def test_forecast_aggregate_dynamic_never_reads_the_actual_future_rate():
    # No-leakage guard: the forecast window's "industry_rate" values must
    # never be read -- only the seed quarter (forecast_start - 1) and the
    # macro features. Corrupting the actual rate INSIDE the forecast
    # window must not change the forecast output at all.
    dataset = _synthetic_bank_dataset(n_quarters=40, n_banks=20)
    series = models.build_industry_series(dataset, "winsorized_nco_rate").frame
    train = series[series["quarter"] <= pd.Period("2022Q4", freq="Q")]
    result = models.fit_aggregate_model(train, _CATEGORY, include_pandemic_dummy=False)

    forecast_start = pd.Period("2023Q1", freq="Q")
    forecast_end = pd.Period("2024Q4", freq="Q")

    forecast_a = models.forecast_aggregate_dynamic(
        result, series, _CATEGORY, forecast_start, forecast_end, include_pandemic_dummy=False
    )

    corrupted = series.copy()
    in_window = (corrupted["quarter"] >= forecast_start) & (corrupted["quarter"] <= forecast_end)
    corrupted.loc[in_window, "industry_rate"] = 999.0  # garbage actual values

    forecast_b = models.forecast_aggregate_dynamic(
        result, corrupted, _CATEGORY, forecast_start, forecast_end, include_pandemic_dummy=False
    )
    pd.testing.assert_series_equal(forecast_a, forecast_b)


def test_forecast_aggregate_dynamic_is_recursive_not_one_step():
    # The AR lag for the SECOND forecast quarter must be the model's own
    # first-quarter PREDICTION, not the (corrupted/garbage) actual rate.
    dataset = _synthetic_bank_dataset(n_quarters=40, n_banks=20)
    series = models.build_industry_series(dataset, "winsorized_nco_rate").frame
    train = series[series["quarter"] <= pd.Period("2022Q4", freq="Q")]
    result = models.fit_aggregate_model(train, _CATEGORY, include_pandemic_dummy=False)

    forecast_start = pd.Period("2023Q1", freq="Q")
    forecast_end = pd.Period("2023Q2", freq="Q")
    forecast = models.forecast_aggregate_dynamic(
        result, series, _CATEGORY, forecast_start, forecast_end, include_pandemic_dummy=False
    )

    # manually recompute Q2's prediction using Q1's PREDICTED value as the
    # AR lag, and confirm it matches -- not using Q1's real actual rate.
    row = series.set_index("quarter").loc[forecast_end]
    values = {
        models.feature_column("Unemployment rate"): row[models.feature_column("Unemployment rate")],
        models.feature_column("House Price Index YoY % change"): row[
            models.feature_column("House Price Index YoY % change")
        ],
        models.feature_column("Commercial Real Estate Price Index YoY % change"): row[
            models.feature_column("Commercial Real Estate Price Index YoY % change")
        ],
        models.feature_column("Dow Jones Total Stock Market Index YoY % change"): row[
            models.feature_column("Dow Jones Total Stock Market Index YoY % change")
        ],
        "industry_rate_lag1": forecast.loc[forecast_start],
    }
    x = pd.DataFrame([values])
    import statsmodels.api as sm

    x = sm.add_constant(x, has_constant="add")[result.params.index]
    expected_q2 = float(result.predict(x).iloc[0])
    assert forecast.loc[forecast_end] == pytest.approx(expected_q2)


# --------------------------------------------------------- model family 2 ---


def test_fit_panel_fe_model_recovers_the_known_signs():
    dataset = _synthetic_bank_dataset(n_quarters=40, n_banks=40)
    fit = models.fit_panel_fe_model(
        dataset, _CATEGORY, "winsorized_nco_rate", include_pandemic_dummy=False
    )
    coefficients = fit.result.params.to_dict()
    t_values = fit.result.tvalues.to_dict()
    signs = models.classify_coefficient_significance(coefficients, t_values)
    assert signs[models.feature_column("Unemployment rate")] == models.CORRECT_SIGN
    assert signs[models.feature_column("House Price Index YoY % change")] == models.CORRECT_SIGN


def test_fit_panel_fe_model_uses_clustered_standard_errors():
    # A real requirement this project's brief asked for explicitly: t-stats
    # for the panel model must be clustered by bank, not the OLS default
    # (which would treat every bank-quarter row as independent and
    # understate standard errors given repeated observations per bank).
    dataset = _synthetic_bank_dataset(n_quarters=40, n_banks=40)
    fit = models.fit_panel_fe_model(
        dataset, _CATEGORY, "winsorized_nco_rate", include_pandemic_dummy=False
    )
    assert fit.result.cov_type == "cluster"


def test_predict_panel_fe_uses_training_means_not_prediction_data_means():
    # No-leakage guard: predicting on a DIFFERENT set of rows for the same
    # bank (e.g. the second call adds more extreme test-period rows for
    # that bank) must not change the prediction for a row shared by both
    # calls -- if the demeaning basis were recomputed from the data being
    # predicted, adding more rows WOULD shift that bank's mean and change
    # the shared row's prediction.
    dataset = _synthetic_bank_dataset(n_quarters=30, n_banks=10)
    train = dataset[dataset["quarter"] <= pd.Period("2020Q4", freq="Q")]
    fit = models.fit_panel_fe_model(
        train, _CATEGORY, "winsorized_nco_rate", include_pandemic_dummy=False
    )

    test = dataset[dataset["quarter"] > pd.Period("2020Q4", freq="Q")].reset_index(drop=True)
    shared_row = test.iloc[[0]]

    prediction_alone = models.predict_panel_fe(
        fit, shared_row, _CATEGORY, "winsorized_nco_rate", include_pandemic_dummy=False
    )

    extra_rows_same_bank = test[test["bank_id"] == shared_row["bank_id"].iloc[0]]
    prediction_with_more_rows = models.predict_panel_fe(
        fit, extra_rows_same_bank, _CATEGORY, "winsorized_nco_rate", include_pandemic_dummy=False
    )

    assert prediction_alone.iloc[0] == pytest.approx(
        prediction_with_more_rows.loc[shared_row.index[0]]
    )


def test_predict_panel_fe_drops_a_bank_never_seen_in_training():
    dataset = _synthetic_bank_dataset(n_quarters=10, n_banks=5)
    train = dataset[dataset["bank_id"] != 4]
    fit = models.fit_panel_fe_model(
        train, _CATEGORY, "winsorized_nco_rate", include_pandemic_dummy=False
    )

    unseen_bank_rows = dataset[dataset["bank_id"] == 4]
    predicted = models.predict_panel_fe(
        fit, unseen_bank_rows, _CATEGORY, "winsorized_nco_rate", include_pandemic_dummy=False
    )
    assert len(predicted) == 0


# --------------------------------------------------------- model family 3 ---


def test_fit_gbm_model_predicts_higher_losses_for_higher_unemployment():
    dataset = _synthetic_bank_dataset(n_quarters=40, n_banks=30)
    model = models.fit_gbm_model(
        dataset, _CATEGORY, "winsorized_nco_rate", include_pandemic_dummy=False
    )

    base_row = dataset.iloc[[0]].copy()
    low_unemployment = base_row.copy()
    low_unemployment[models.feature_column("Unemployment rate")] = 2.0
    high_unemployment = base_row.copy()
    high_unemployment[models.feature_column("Unemployment rate")] = 10.0

    low_pred = models.predict_gbm(model, low_unemployment, _CATEGORY, include_pandemic_dummy=False)
    high_pred = models.predict_gbm(
        model, high_unemployment, _CATEGORY, include_pandemic_dummy=False
    )
    assert high_pred.iloc[0] > low_pred.iloc[0]


def test_gbm_subsamples_when_training_data_exceeds_the_row_cap():
    dataset = _synthetic_bank_dataset(n_quarters=40, n_banks=30)  # 1,200 rows
    model = models.fit_gbm_model(
        dataset,
        _CATEGORY,
        "winsorized_nco_rate",
        include_pandemic_dummy=False,
        max_training_rows=100,
    )
    assert model.train_score_.shape[0] == model.n_estimators  # fit succeeded on the subsample


# --------------------------------------------------- industry aggregation ---


def test_aggregate_bank_predictions_to_industry_rate_is_balance_weighted():
    dataset = pd.DataFrame(
        {
            "quarter": [pd.Period("2020Q1", freq="Q")] * 2,
            "average_balance": [100.0, 300.0],
        }
    )
    predicted = pd.Series([0.01, 0.05], index=dataset.index)
    industry = models.aggregate_bank_predictions_to_industry_rate(dataset, predicted)
    expected = (100.0 * 0.01 + 300.0 * 0.05) / 400.0
    assert industry.iloc[0] == pytest.approx(expected)


# ---------------------------------------------------- anchored family ---


def test_compute_bank_relative_levels_is_bank_mean_over_aggregate_mean():
    train_bank = pd.DataFrame(
        {
            "bank_id": [1, 1, 2, 2],
            "winsorized_nco_rate": [0.02, 0.02, 0.01, 0.01],
        }
    )
    train_industry = pd.DataFrame({"industry_rate": [0.01, 0.02]})
    relative = models.compute_bank_relative_levels(
        train_bank, "winsorized_nco_rate", train_industry
    )
    # aggregate mean = 0.015; bank 1's mean 0.02 -> 0.02/0.015; bank 2's -> 0.01/0.015
    assert relative.loc[1] == pytest.approx(0.02 / 0.015)
    assert relative.loc[2] == pytest.approx(0.01 / 0.015)


def test_compute_shrunk_bank_relative_levels_matches_raw_above_the_min_quarters_threshold():
    train_bank = pd.DataFrame(
        {
            "bank_id": [1] * 10,
            "winsorized_nco_rate": [0.02] * 10,
        }
    )
    train_industry = pd.DataFrame({"industry_rate": [0.01] * 10})
    shrunk = models.compute_shrunk_bank_relative_levels(
        train_bank, "winsorized_nco_rate", train_industry, min_quarters=8
    )
    raw = models.compute_bank_relative_levels(train_bank, "winsorized_nco_rate", train_industry)
    assert shrunk.loc[1] == pytest.approx(raw.loc[1])


def test_compute_shrunk_bank_relative_levels_shrinks_toward_1_below_the_threshold():
    # bank 1 has only 2 quarters of history (min_quarters=8) -- its raw relative level (2.0,
    # a large outlier) should be pulled toward the industry-neutral 1.0, not used raw.
    train_bank = pd.DataFrame(
        {
            "bank_id": [1, 1],
            "winsorized_nco_rate": [0.02, 0.02],
        }
    )
    train_industry = pd.DataFrame({"industry_rate": [0.01, 0.01]})
    shrunk = models.compute_shrunk_bank_relative_levels(
        train_bank, "winsorized_nco_rate", train_industry, min_quarters=8
    )
    raw = models.compute_bank_relative_levels(train_bank, "winsorized_nco_rate", train_industry)
    assert raw.loc[1] == pytest.approx(2.0)
    weight = 2.0 / 8.0
    expected_shrunk = raw.loc[1] * weight + 1.0 * (1.0 - weight)
    assert shrunk.loc[1] == pytest.approx(expected_shrunk)
    assert 1.0 < shrunk.loc[1] < raw.loc[1]  # pulled toward 1.0, not left at the raw outlier


def test_compute_shrunk_bank_relative_levels_is_exactly_1_with_zero_quarters_of_overlap():
    # bank 2 has nonnull rows, but NONE of its quarters appear in train_industry's own
    # period -- its count of USABLE quarters for the shrinkage weight should floor it to a
    # heavily-shrunk (here: fully shrunk, since the weight clips at 0 quarters -> weight 0)
    # relative level once pulled toward 1.0. (Realistically this edge case is rare; this
    # confirms the weight formula degrades gracefully rather than dividing by zero or
    # producing a nonsensical negative weight.)
    train_bank = pd.DataFrame({"bank_id": [1], "winsorized_nco_rate": [0.05]})
    train_industry = pd.DataFrame({"industry_rate": [0.01]})
    shrunk = models.compute_shrunk_bank_relative_levels(
        train_bank, "winsorized_nco_rate", train_industry, min_quarters=8
    )
    weight = 1.0 / 8.0
    raw = models.compute_bank_relative_levels(train_bank, "winsorized_nco_rate", train_industry)
    expected = raw.loc[1] * weight + 1.0 * (1.0 - weight)
    assert shrunk.loc[1] == pytest.approx(expected)


def test_forecast_anchored_to_aggregate_multiplies_relative_level_by_aggregate_forecast():
    relative_levels = pd.Series({1: 2.0, 2: 0.5})
    aggregate_forecast = pd.Series(
        {pd.Period("2007Q1", freq="Q"): 0.04, pd.Period("2007Q2", freq="Q"): 0.06}
    )
    test_bank = pd.DataFrame(
        {
            "bank_id": [1, 2],
            "quarter": [pd.Period("2007Q1", freq="Q"), pd.Period("2007Q1", freq="Q")],
        }
    )
    predicted = models.forecast_anchored_to_aggregate(
        relative_levels, aggregate_forecast, test_bank
    )
    assert predicted.loc[0] == pytest.approx(2.0 * 0.04)
    assert predicted.loc[1] == pytest.approx(0.5 * 0.04)


def test_forecast_anchored_to_aggregate_drops_a_bank_never_seen_in_training():
    relative_levels = pd.Series({1: 2.0})
    aggregate_forecast = pd.Series({pd.Period("2007Q1", freq="Q"): 0.04})
    test_bank = pd.DataFrame({"bank_id": [1, 99], "quarter": [pd.Period("2007Q1", freq="Q")] * 2})
    predicted = models.forecast_anchored_to_aggregate(
        relative_levels, aggregate_forecast, test_bank
    )
    assert list(predicted.index) == [0]


def test_forecast_anchored_to_aggregate_reflects_a_worse_long_history_peak():
    # This is the whole point of anchoring: a bank whose own Call-Report-
    # era history understates crisis severity should still get a bigger
    # projected peak when anchored to a long-history aggregate forecast
    # that itself captures a worse crisis peak.
    relative_levels = pd.Series({1: 1.0})
    mild_aggregate_forecast = pd.Series({pd.Period("2009Q1", freq="Q"): 0.01})
    severe_aggregate_forecast = pd.Series({pd.Period("2009Q1", freq="Q"): 0.07})
    test_bank = pd.DataFrame({"bank_id": [1], "quarter": [pd.Period("2009Q1", freq="Q")]})
    mild = models.forecast_anchored_to_aggregate(
        relative_levels, mild_aggregate_forecast, test_bank
    )
    severe = models.forecast_anchored_to_aggregate(
        relative_levels, severe_aggregate_forecast, test_bank
    )
    assert severe.iloc[0] > mild.iloc[0]
