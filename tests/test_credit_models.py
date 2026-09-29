"""Offline tests for Stage 4's model families -- synthetic data only, no
network, no real panel needed. Focus: no leakage across an out-of-time
train/test boundary, and known synthetic relationships recovered with the
expected sign (the two things the project brief explicitly asks for)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from corefin.credit import models


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


def test_check_coefficient_signs_flags_wrong_and_right_signs():
    coefficients = {
        models.feature_column("Unemployment rate"): 0.003,  # correct: positive
        models.feature_column("House Price Index YoY % change"): 0.001,  # WRONG: should be negative
    }
    result = models.check_coefficient_signs(coefficients)
    assert result[models.feature_column("Unemployment rate")] is True
    assert result[models.feature_column("House Price Index YoY % change")] is False


def test_check_coefficient_signs_treats_exact_zero_as_not_matching():
    coefficients = {models.feature_column("Unemployment rate"): 0.0}
    result = models.check_coefficient_signs(coefficients)
    assert result[models.feature_column("Unemployment rate")] is False


def test_check_coefficient_signs_only_reports_present_features():
    result = models.check_coefficient_signs({models.feature_column("Unemployment rate"): 1.0})
    assert set(result) == {models.feature_column("Unemployment rate")}


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
    result = models.fit_aggregate_model(series, include_pandemic_dummy=False)
    coefficients = result.params.to_dict()
    signs = models.check_coefficient_signs(coefficients)
    assert signs[models.feature_column("Unemployment rate")] is True
    assert signs[models.feature_column("House Price Index YoY % change")] is True


def test_forecast_aggregate_dynamic_never_reads_the_actual_future_rate():
    # No-leakage guard: the forecast window's "industry_rate" values must
    # never be read -- only the seed quarter (forecast_start - 1) and the
    # macro features. Corrupting the actual rate INSIDE the forecast
    # window must not change the forecast output at all.
    dataset = _synthetic_bank_dataset(n_quarters=40, n_banks=20)
    series = models.build_industry_series(dataset, "winsorized_nco_rate").frame
    train = series[series["quarter"] <= pd.Period("2022Q4", freq="Q")]
    result = models.fit_aggregate_model(train, include_pandemic_dummy=False)

    forecast_start = pd.Period("2023Q1", freq="Q")
    forecast_end = pd.Period("2024Q4", freq="Q")

    forecast_a = models.forecast_aggregate_dynamic(
        result, series, forecast_start, forecast_end, include_pandemic_dummy=False
    )

    corrupted = series.copy()
    in_window = (corrupted["quarter"] >= forecast_start) & (corrupted["quarter"] <= forecast_end)
    corrupted.loc[in_window, "industry_rate"] = 999.0  # garbage actual values

    forecast_b = models.forecast_aggregate_dynamic(
        result, corrupted, forecast_start, forecast_end, include_pandemic_dummy=False
    )
    pd.testing.assert_series_equal(forecast_a, forecast_b)


def test_forecast_aggregate_dynamic_is_recursive_not_one_step():
    # The AR lag for the SECOND forecast quarter must be the model's own
    # first-quarter PREDICTION, not the (corrupted/garbage) actual rate.
    dataset = _synthetic_bank_dataset(n_quarters=40, n_banks=20)
    series = models.build_industry_series(dataset, "winsorized_nco_rate").frame
    train = series[series["quarter"] <= pd.Period("2022Q4", freq="Q")]
    result = models.fit_aggregate_model(train, include_pandemic_dummy=False)

    forecast_start = pd.Period("2023Q1", freq="Q")
    forecast_end = pd.Period("2023Q2", freq="Q")
    forecast = models.forecast_aggregate_dynamic(
        result, series, forecast_start, forecast_end, include_pandemic_dummy=False
    )

    # manually recompute Q2's prediction using Q1's PREDICTED value as the
    # AR lag, and confirm it matches -- not using Q1's real actual rate.
    row = series.set_index("quarter").loc[forecast_end]
    values = {
        models.feature_column("Unemployment rate"): row[models.feature_column("Unemployment rate")],
        models.feature_column("House Price Index YoY % change"): row[
            models.feature_column("House Price Index YoY % change")
        ],
        models.feature_column(
            "Commercial Real Estate Price Index YoY % change"
        ): row[models.feature_column("Commercial Real Estate Price Index YoY % change")],
        models.feature_column(
            "Dow Jones Total Stock Market Index YoY % change"
        ): row[models.feature_column("Dow Jones Total Stock Market Index YoY % change")],
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
    fit = models.fit_panel_fe_model(dataset, "winsorized_nco_rate", include_pandemic_dummy=False)
    coefficients = fit.result.params.to_dict()
    signs = models.check_coefficient_signs(coefficients)
    assert signs[models.feature_column("Unemployment rate")] is True
    assert signs[models.feature_column("House Price Index YoY % change")] is True


def test_predict_panel_fe_uses_training_means_not_prediction_data_means():
    # No-leakage guard: predicting on a DIFFERENT set of rows for the same
    # bank (e.g. the second call adds more extreme test-period rows for
    # that bank) must not change the prediction for a row shared by both
    # calls -- if the demeaning basis were recomputed from the data being
    # predicted, adding more rows WOULD shift that bank's mean and change
    # the shared row's prediction.
    dataset = _synthetic_bank_dataset(n_quarters=30, n_banks=10)
    train = dataset[dataset["quarter"] <= pd.Period("2020Q4", freq="Q")]
    fit = models.fit_panel_fe_model(train, "winsorized_nco_rate", include_pandemic_dummy=False)

    test = dataset[dataset["quarter"] > pd.Period("2020Q4", freq="Q")].reset_index(drop=True)
    shared_row = test.iloc[[0]]

    prediction_alone = models.predict_panel_fe(
        fit, shared_row, "winsorized_nco_rate", include_pandemic_dummy=False
    )

    extra_rows_same_bank = test[test["bank_id"] == shared_row["bank_id"].iloc[0]]
    prediction_with_more_rows = models.predict_panel_fe(
        fit, extra_rows_same_bank, "winsorized_nco_rate", include_pandemic_dummy=False
    )

    assert prediction_alone.iloc[0] == pytest.approx(
        prediction_with_more_rows.loc[shared_row.index[0]]
    )


def test_predict_panel_fe_drops_a_bank_never_seen_in_training():
    dataset = _synthetic_bank_dataset(n_quarters=10, n_banks=5)
    train = dataset[dataset["bank_id"] != 4]
    fit = models.fit_panel_fe_model(train, "winsorized_nco_rate", include_pandemic_dummy=False)

    unseen_bank_rows = dataset[dataset["bank_id"] == 4]
    predicted = models.predict_panel_fe(
        fit, unseen_bank_rows, "winsorized_nco_rate", include_pandemic_dummy=False
    )
    assert len(predicted) == 0


# --------------------------------------------------------- model family 3 ---


def test_fit_gbm_model_predicts_higher_losses_for_higher_unemployment():
    dataset = _synthetic_bank_dataset(n_quarters=40, n_banks=30)
    model = models.fit_gbm_model(dataset, "winsorized_nco_rate", include_pandemic_dummy=False)

    base_row = dataset.iloc[[0]].copy()
    low_unemployment = base_row.copy()
    low_unemployment[models.feature_column("Unemployment rate")] = 2.0
    high_unemployment = base_row.copy()
    high_unemployment[models.feature_column("Unemployment rate")] = 10.0

    low_pred = models.predict_gbm(model, low_unemployment, include_pandemic_dummy=False)
    high_pred = models.predict_gbm(model, high_unemployment, include_pandemic_dummy=False)
    assert high_pred.iloc[0] > low_pred.iloc[0]


def test_gbm_subsamples_when_training_data_exceeds_the_row_cap():
    dataset = _synthetic_bank_dataset(n_quarters=40, n_banks=30)  # 1,200 rows
    model = models.fit_gbm_model(
        dataset, "winsorized_nco_rate", include_pandemic_dummy=False, max_training_rows=100
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
