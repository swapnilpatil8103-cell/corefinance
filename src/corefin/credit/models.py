"""Stage 4 model families: (1) aggregate industry-rate regression on
lagged macro drivers plus the lagged dependent variable, (2) a bank-level
panel with bank fixed effects (within/demeaning estimator) plus macro
drivers and bank characteristics, and (3) gradient boosting on the same
bank-level features. Each fits either the winsorized NCO rate or the
winsorized NPL ratio as the dependent variable. No function here touches
the network or reads/writes files -- see cli.py for orchestration.

FEATURE SET AND LAG, BY A SIMPLE DOCUMENTED RULE, NOT A SEARCH: every
model uses the SAME single lag (`LAG` = 1 quarter) for every macro driver
-- a charge-off/delinquency decision typically reflects economic
conditions from about a quarter earlier, the standard convention in
credit-loss stress-testing literature, and using one lag uniformly (not a
per-category or per-variable search) is what "keep models simple" means
here. `CORE_MACRO_FEATURES` is the full universe of 4 macro drivers this
project's Stage 4 modeling dataset makes available: unemployment rate
(labor-market stress), and the YoY percent-change (not the raw level --
Stage 3 built these specifically because the levels are non-stationary)
of three price/index variables: House Price Index, Commercial Real
Estate Price Index, and the Dow Jones Total Stock Market Index proxy.

`CATEGORY_MACRO_FEATURES` selects, PER CATEGORY, which of those 4 a model
actually uses -- not a per-category search over many candidates (still
"keep models simple"), but ONE documented exception found on the real
build: the stock index's YoY change came back wrong-signed AND
statistically significant (see `classify_coefficient_significance`) in
the bank panel model for cre_multifamily, cre_nonfarm_nonresidential,
residential_mortgage, auto, and other_consumer -- a real finding, not
noise (consistent across multiple validation windows and both COVID
specs for each). The likely mechanism: 2009-2010's sharp equity-market
rebound coincided with the same quarters' peak loss rates for these
categories (losses lag the initial shock; the market recovery does not
wait for losses to peak), so at LAG=1 the stock index's YoY change is
still deeply negative (base effect: measured against the 2008-2009
trough) while losses are at their worst -- a real, contemporaneous
COINCIDENCE of the recovery and the loss peak, not a genuine causal
"rising stocks raise these categories' losses" relationship. Only
commercial_and_industrial keeps the stock index (a business-lending
category genuinely sensitive to equity/credit-market conditions, and
where the coefficient came back correctly signed); every other category
uses the remaining 3 features. This changes 3->4 features for one
category and 4->3 for the rest -- not a widened search, a single,
documented correction based on a confirmed sign problem.

A SEPARATE finding, investigated but NOT changed: the aggregate_long
family (industry_history.py's FRED-derived long-history aggregate model)
shows a significant NEGATIVE unemployment coefficient for credit_card,
auto, and other_consumer specifically. Investigated via real correlation
diagnostics (not assumed): unemployment_lag1's UNIVARIATE correlation
with the industry rate is positive for all three (0.20-0.36, the
expected direction), but the AR term (industry_rate_lag1) is extremely
dominant for these series (0.85-0.94 correlation with the industry
rate itself) and is itself moderately correlated with unemployment_lag1
(0.30-0.43) -- textbook AR-term collinearity: with so much of the
variance already absorbed by the lagged dependent variable, the
REMAINING (partial) relationship attributed to unemployment is small and
statistically unstable, and can flip sign in a finite sample without
implying a real, wrong economic relationship. This is NOT fixed by
dropping the AR term (the aggregate model is explicitly specified with
one, per the original brief) or by changing the lag (that would be
exactly the per-category lag search this project deliberately avoids) --
it's disclosed here and in the coefficient table instead. The FRED series
themselves were re-checked and are not the cause (verified live in an
earlier round: real, sensible charge-off magnitudes, correctly annualized
percent units).

EXPECTED COEFFICIENT SIGNS: higher unemployment should raise losses
(positive coefficient); rising home prices, rising CRE prices, and a
rising stock market should LOWER losses, so a NEGATIVE coefficient on
each of those YoY-change features is expected (equivalently: FALLING
prices, a negative change value, combined with a negative coefficient,
raise losses -- the sign convention the project brief asks to check).
See `EXPECTED_COEFFICIENT_SIGNS` and `classify_coefficient_significance`.

COVID TREATMENT: `PANDEMIC_START`/`PANDEMIC_END` (2020Q2-2021Q4) define a
0/1 indicator included as an extra regressor in the MAIN specification
(`add_pandemic_indicator`). The ROBUSTNESS specification instead drops
every row in [`ROBUSTNESS_EXCLUDE_START`, `ROBUSTNESS_EXCLUDE_END`]
(all of 2020-2021) from the TRAINING sample entirely
(`exclude_pandemic_years`) -- forecast/test rows are never dropped by
either treatment, only training rows. Both must be fit and reported per
the brief; note that for a training window that doesn't reach 2020-2021
at all (e.g. train-through-2006), the two specifications are identical by
construction -- there's nothing to include a dummy for or exclude.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.ensemble import GradientBoostingRegressor

from corefin.credit.schema import LoanCategory

PANDEMIC_START = pd.Period("2020Q2", freq="Q")
PANDEMIC_END = pd.Period("2021Q4", freq="Q")
ROBUSTNESS_EXCLUDE_START = pd.Period("2020Q1", freq="Q")
ROBUSTNESS_EXCLUDE_END = pd.Period("2021Q4", freq="Q")

LAG = 1

UNEMPLOYMENT_FEATURE = "Unemployment rate"
HPI_FEATURE = "House Price Index YoY % change"
CRE_PRICE_FEATURE = "Commercial Real Estate Price Index YoY % change"
STOCK_INDEX_FEATURE = "Dow Jones Total Stock Market Index YoY % change"

CORE_MACRO_FEATURES: tuple[str, ...] = (
    UNEMPLOYMENT_FEATURE,
    HPI_FEATURE,
    CRE_PRICE_FEATURE,
    STOCK_INDEX_FEATURE,
)

_FEATURES_WITHOUT_STOCK_INDEX: tuple[str, ...] = (
    UNEMPLOYMENT_FEATURE,
    HPI_FEATURE,
    CRE_PRICE_FEATURE,
)

# Per-category feature set -- see the module docstring for why C&I is the
# one category that keeps the stock index. Keyed by the LoanCategory's
# own string value (StrEnum), so both a LoanCategory member and its plain
# string value look up the same entry.
CATEGORY_MACRO_FEATURES: dict[str, tuple[str, ...]] = {
    str(category): (
        CORE_MACRO_FEATURES if category == LoanCategory.CI else _FEATURES_WITHOUT_STOCK_INDEX
    )
    for category in LoanCategory
}

# EXTENDED features for the aggregate/anchored model families ONLY
# (fit_aggregate_model/forecast_aggregate_dynamic, via `_aggregate_
# feature_columns`) -- NOT panel_fe/gbm (`_macro_feature_columns`,
# unchanged). A single structured addition investigated on real Stage 5
# review (not a broad search): the existing features are all either a
# RATE LEVEL (unemployment) or a 1-year (YoY) price change; these add a
# 1-year CHANGE in the unemployment rate itself (distinct from its
# level) and a longer, 2-year CUMULATIVE price change for the two price
# indices already in CORE_MACRO_FEATURES -- see macro.
# add_extended_change_features for the exact definitions.
UNEMPLOYMENT_4Q_CHANGE_FEATURE = "Unemployment rate 4Q change"
HPI_8Q_CHANGE_FEATURE = "House Price Index 8Q change"
CRE_8Q_CHANGE_FEATURE = "Commercial Real Estate Price Index 8Q change"

EXTENDED_AGGREGATE_FEATURES: tuple[str, ...] = (
    UNEMPLOYMENT_4Q_CHANGE_FEATURE,
    HPI_8Q_CHANGE_FEATURE,
    CRE_8Q_CHANGE_FEATURE,
)

# Whether fit_aggregate_model/forecast_aggregate_dynamic include
# EXTENDED_AGGREGATE_FEATURES -- TRIED on the real data and NOT kept.
# Tested by actually refitting every aggregate_ar/aggregate_long/
# anchored_to_aggregate backtest and crisis-replay with this flag on
# (coefficient_table.csv/backtest_table.csv/crisis_replay.csv, archived
# before/after for comparison): the in-sample crisis-replay diagnostic
# DID improve on average (mean |gap| across the 7 applicable categories:
# 1.12 points -> 0.84 points), but at a real cost that fails this
# project's own "don't keep it if it breaks backtests or signs" bar --
# out-of-time backtest RMSE got WORSE in 35 of 44 (category, window,
# family) cells, often severely (e.g. credit_card's 2020-2021 aggregate_ar
# RMSE nearly quadrupled, 0.0178 -> 0.0672; commercial_and_industrial's
# nearly sextupled, 0.0033 -> 0.0184) -- textbook overfitting: 3 extra
# regressors fit on a short backtest-training window improve the IN-
# SAMPLE crisis fit almost by construction, while hurting genuine out-of-
# time generalization, especially for 2020-2021 (COVID's shock has no
# precedent a smooth "4Q change"/"8Q cumulative change" feature can
# represent). It also introduced a NEW wrong_sign_significant
# classification not present before (cre_construction's aggregate_ar
# unemployment coefficient). Documented here and in the README's Stage 5
# limitations section, per this project's "do not add a fudge multiplier,
# document the gap plainly" rule -- kept as working, tested, available
# code (not deleted) in case a future, better-regularized version is
# worth revisiting, but OFF by default. Regardless of this flag: `fit_
# aggregate_model`/`forecast_aggregate_dynamic` filter to whatever
# feature columns are ACTUALLY present, so this is a no-op against older
# data/synthetic test fixtures that don't have them.
USE_EXTENDED_AGGREGATE_FEATURES = False

EXPECTED_COEFFICIENT_SIGNS: dict[str, int] = {
    UNEMPLOYMENT_FEATURE: 1,
    HPI_FEATURE: -1,
    CRE_PRICE_FEATURE: -1,
    STOCK_INDEX_FEATURE: -1,
    # Rising unemployment (a positive 4Q change) raises losses, same
    # direction as the level; rising home/CRE prices over 2 years (a
    # positive 8Q change) lower losses, same direction as the existing
    # 1-year (YoY) change.
    UNEMPLOYMENT_4Q_CHANGE_FEATURE: 1,
    HPI_8Q_CHANGE_FEATURE: -1,
    CRE_8Q_CHANGE_FEATURE: -1,
}

# GBM is fit on a random sample (fixed seed, for reproducibility) when a
# category's bank-quarter row count exceeds this -- keeps runtime bounded
# across 9 categories x 2 dependents x 2 COVID specs x 3 validation
# windows without materially changing what a shallow, 100-tree model can
# learn from a training set this large either way.
GBM_MAX_TRAINING_ROWS = 20_000
GBM_RANDOM_STATE = 0


def feature_column(variable: str, lag: int = LAG) -> str:
    """The modeling dataset's lagged column name for a macro `variable`
    at `lag` quarters -- matches macro.build_modeling_dataset's own
    f"{variable}_lag{lag}" naming."""
    return f"{variable}_lag{lag}"


def add_pandemic_indicator(dataset: pd.DataFrame, quarter_column: str = "quarter") -> pd.Series:
    """1.0 for rows whose quarter falls in [PANDEMIC_START, PANDEMIC_END]
    (2020Q2-2021Q4 inclusive), 0.0 otherwise."""
    in_window = (dataset[quarter_column] >= PANDEMIC_START) & (
        dataset[quarter_column] <= PANDEMIC_END
    )
    return in_window.astype(float)


def exclude_pandemic_years(dataset: pd.DataFrame, quarter_column: str = "quarter") -> pd.DataFrame:
    """Drops every row whose quarter falls in [ROBUSTNESS_EXCLUDE_START,
    ROBUSTNESS_EXCLUDE_END] (all of 2020-2021) -- the robustness COVID
    treatment. Rows outside that window are returned unchanged."""
    excluded = (dataset[quarter_column] >= ROBUSTNESS_EXCLUDE_START) & (
        dataset[quarter_column] <= ROBUSTNESS_EXCLUDE_END
    )
    return dataset[~excluded]


# Two-sided ~5% significance cutoff on |t| -- the conventional threshold,
# not tuned for this project's data.
SIGNIFICANCE_T_THRESHOLD = 1.96

CORRECT_SIGN = "correct_sign"
WRONG_SIGN_INSIGNIFICANT = "wrong_sign_insignificant"
WRONG_SIGN_SIGNIFICANT = "wrong_sign_significant"


def classify_coefficient_significance(
    coefficients: dict[str, float], t_values: dict[str, float]
) -> dict[str, str]:
    """coefficients/t_values: {feature column name -> value}, as produced
    by an OLS result's `.params`/`.tvalues` (converted to dicts) --
    `fit_panel_fe_model`'s result uses bank-clustered standard errors, so
    its t-values already account for repeated observations per bank, not
    just the demeaned regression's naive (and too-small) residual
    variance. For every core macro feature present in BOTH dicts,
    classifies it as one of three buckets (only the last is a real
    problem worth flagging, per this project's brief -- a wrong sign
    that isn't statistically distinguishable from zero isn't a confirmed
    contradiction of the expected relationship, just noise):
    - CORRECT_SIGN: matches EXPECTED_COEFFICIENT_SIGNS.
    - WRONG_SIGN_INSIGNIFICANT: wrong sign, but |t| < SIGNIFICANCE_T_THRESHOLD.
    - WRONG_SIGN_SIGNIFICANT: wrong sign AND |t| >= SIGNIFICANCE_T_THRESHOLD.
    A coefficient of exactly 0.0 is treated as wrong-signed (a real
    relationship should be nonzero, not just "not wrong"), though in
    practice t=0 there too, so it lands in WRONG_SIGN_INSIGNIFICANT."""
    result = {}
    for variable, expected_sign in EXPECTED_COEFFICIENT_SIGNS.items():
        column = feature_column(variable)
        if column not in coefficients or column not in t_values:
            continue
        correct_sign = (coefficients[column] * expected_sign) > 0
        if correct_sign:
            result[column] = CORRECT_SIGN
        elif abs(t_values[column]) >= SIGNIFICANCE_T_THRESHOLD:
            result[column] = WRONG_SIGN_SIGNIFICANT
        else:
            result[column] = WRONG_SIGN_INSIGNIFICANT
    return result


def _macro_feature_columns(category: str, include_pandemic: bool) -> list[str]:
    """The macro feature columns a model for `category` uses -- see
    CATEGORY_MACRO_FEATURES (every category except commercial_and_
    industrial drops the stock index)."""
    variables = CATEGORY_MACRO_FEATURES[category]
    columns = [feature_column(variable) for variable in variables]
    if include_pandemic:
        columns = [*columns, "pandemic"]
    return columns


def _aggregate_feature_columns(category: str, include_pandemic: bool) -> list[str]:
    """Like `_macro_feature_columns`, but for the AGGREGATE/ANCHORED
    families only (fit_aggregate_model/forecast_aggregate_dynamic): adds
    EXTENDED_AGGREGATE_FEATURES when USE_EXTENDED_AGGREGATE_FEATURES is
    set. panel_fe/gbm keep using `_macro_feature_columns` directly,
    unaffected by this flag."""
    variables = list(CATEGORY_MACRO_FEATURES[category])
    if USE_EXTENDED_AGGREGATE_FEATURES:
        variables.extend(EXTENDED_AGGREGATE_FEATURES)
    columns = [feature_column(variable) for variable in variables]
    if include_pandemic:
        columns = [*columns, "pandemic"]
    return columns


@dataclass(frozen=True)
class IndustrySeries:
    """One category's quarterly industry-level series: `frame` has columns
    "quarter", "industry_rate" (balance-weighted mean of the dependent
    variable across banks), "industry_rate_lag1", one `feature_column`
    per CORE_MACRO_FEATURES, and "pandemic"."""

    frame: pd.DataFrame


def build_industry_series(bank_dataset: pd.DataFrame, dependent_column: str) -> IndustrySeries:
    """bank_dataset: one category's bank-quarter rows from the modeling
    dataset (already filtered to a single `category`). Aggregates
    `dependent_column` (e.g. "winsorized_nco_rate" or
    "winsorized_npl_ratio") into a quarterly, balance-weighted (by
    average_balance) industry rate, attaches each quarter's own macro lag
    features (identical across banks within a quarter by construction, so
    `.first()` recovers them) and the pandemic indicator, and adds
    "industry_rate_lag1" (the AR term model family 1 uses)."""
    scoped = bank_dataset.dropna(subset=[dependent_column, "average_balance"])
    weight = scoped["average_balance"].to_numpy()
    weighted_value = scoped[dependent_column].to_numpy() * weight
    grouped = pd.DataFrame(
        {"quarter": scoped["quarter"], "_w": weight, "_wx": weighted_value}
    ).groupby("quarter", observed=True)
    sums = grouped[["_w", "_wx"]].sum()
    industry_rate = (sums["_wx"] / sums["_w"]).rename("industry_rate")

    # always the FULL set of macro columns (not the category-specific
    # subset), PLUS EXTENDED_AGGREGATE_FEATURES's lag columns when present
    # (bank_dataset may not carry them -- an older modeling dataset, or a
    # synthetic test fixture) -- this is just a data-carrying frame;
    # fit_aggregate_model/forecast_aggregate_dynamic select the category-
    # appropriate subset when they actually use it as regressors.
    macro_columns = [feature_column(variable) for variable in CORE_MACRO_FEATURES]
    extended_columns = [feature_column(variable) for variable in EXTENDED_AGGREGATE_FEATURES]
    macro_columns += [c for c in extended_columns if c in scoped.columns]
    macro_first = scoped.groupby("quarter", observed=True)[macro_columns].first()

    frame = pd.concat([industry_rate, macro_first], axis=1).reset_index()
    frame = frame.sort_values("quarter").reset_index(drop=True)
    frame["pandemic"] = add_pandemic_indicator(frame)
    frame["industry_rate_lag1"] = frame["industry_rate"].shift(1)
    return IndustrySeries(frame=frame)


def fit_aggregate_model(
    industry_series: pd.DataFrame, category: str, include_pandemic_dummy: bool = True
) -> sm.regression.linear_model.RegressionResultsWrapper:
    """Model family 1: OLS of the industry rate on its own lag
    ("industry_rate_lag1") plus `category`'s own macro features
    (CATEGORY_MACRO_FEATURES, plus EXTENDED_AGGREGATE_FEATURES if
    USE_EXTENDED_AGGREGATE_FEATURES -- see `_aggregate_feature_columns`)
    at `LAG` (plus the pandemic dummy if `include_pandemic_dummy`).
    `industry_series`: output of `build_industry_series`'s `.frame`
    (already possibly filtered to a training window / pandemic-excluded
    by the caller). A feature column not actually present in `industry_
    series` (e.g. EXTENDED_AGGREGATE_FEATURES on older data or a
    synthetic test fixture that doesn't carry them) is silently dropped,
    not an error."""
    feature_columns = [
        "industry_rate_lag1",
        *_aggregate_feature_columns(category, include_pandemic_dummy),
    ]
    feature_columns = [c for c in feature_columns if c in industry_series.columns]
    data = industry_series.dropna(subset=["industry_rate", *feature_columns])
    x = sm.add_constant(data[feature_columns], has_constant="add")
    y = data["industry_rate"]
    return sm.OLS(y, x).fit()


def forecast_aggregate_dynamic(
    result: sm.regression.linear_model.RegressionResultsWrapper,
    industry_series: pd.DataFrame,
    category: str,
    forecast_start: pd.Period,
    forecast_end: pd.Period,
    include_pandemic_dummy: bool = True,
) -> pd.Series:
    """A genuine out-of-time, multi-quarter-ahead forecast: at each
    forecast quarter, "industry_rate_lag1" is the model's OWN prior-step
    PREDICTION (dynamic/recursive simulation), never the real historical
    rate -- using the real rate at every step would let the model "see"
    the actual path it's supposed to be forecasting, defeating the point
    of an out-of-time backtest. The macro lag features ARE the real,
    historical values for each forecast quarter (legitimate: by the time
    quarter t's outcome is being forecast, quarter t-1's macro data -- the
    LAG features -- has already been observed in real time). Seeded from
    `industry_series`'s own actual rate at `forecast_start - 1`. Indexed
    by quarter (PeriodIndex, freq="Q")."""
    indexed = industry_series.set_index("quarter")
    seed_quarter = forecast_start - 1
    prior_rate = indexed.loc[seed_quarter, "industry_rate"]

    feature_columns = [
        "industry_rate_lag1",
        *_aggregate_feature_columns(category, include_pandemic_dummy),
    ]
    feature_columns = [c for c in feature_columns if c in industry_series.columns]
    predictions: dict[pd.Period, float] = {}
    quarter = forecast_start
    while quarter <= forecast_end:
        row = indexed.loc[quarter]
        values = {col: row[col] for col in feature_columns if col != "industry_rate_lag1"}
        values["industry_rate_lag1"] = prior_rate
        x = pd.DataFrame([values])
        x = sm.add_constant(x, has_constant="add")
        x = x[result.params.index]
        predicted = float(result.predict(x).iloc[0])
        predictions[quarter] = predicted
        prior_rate = predicted
        quarter = quarter + 1
    return pd.Series(predictions).rename_axis("quarter")


def _add_bank_characteristics(bank_dataset: pd.DataFrame) -> pd.DataFrame:
    """Adds "log_balance" (the one bank characteristic) and "pandemic"
    (computed here, not required of callers -- `fit_panel_fe_model`/
    `fit_gbm_model`/their predict counterparts all reference a "pandemic"
    column via `_macro_feature_columns`, so it must exist on every
    bank-level frame passed to them; only `build_industry_series` adds it
    for the industry-level frame itself, which doesn't go through this
    function)."""
    result = bank_dataset.copy()
    result["log_balance"] = np.log(result["average_balance"].clip(lower=1.0))
    result["pandemic"] = add_pandemic_indicator(result)
    return result


@dataclass(frozen=True)
class PanelFEFit:
    """`result`: the within-estimator OLS fit (see `fit_panel_fe_model`).
    `bank_training_means`: each bank's OWN mean of every feature (and the
    dependent variable) OVER THE TRAINING SAMPLE ONLY -- the demeaning
    basis `predict_panel_fe` must reuse for out-of-time prediction. Using
    a mean recomputed from whatever data is being PREDICTED (e.g. the
    forecast/test window) instead of this training-only basis would leak
    the very data the forecast is supposed to be blind to into the
    demeaning step -- a real, non-obvious leakage risk this project's
    no-leakage tests guard against."""

    result: sm.regression.linear_model.RegressionResultsWrapper
    bank_training_means: pd.DataFrame


def fit_panel_fe_model(
    bank_dataset: pd.DataFrame,
    category: str,
    dependent_column: str,
    include_pandemic_dummy: bool = True,
) -> PanelFEFit:
    """Model family 2: bank-level panel with bank FIXED EFFECTS, fit via
    the within (demeaning) estimator -- each bank's own TRAINING-SAMPLE
    mean is subtracted from every variable (dependent + features) before
    an OLS fit with no intercept, the standard way to fit large-N fixed
    effects without materializing one dummy column per bank. Standard
    errors are CLUSTERED BY BANK (`cov_type="cluster"`) -- a single
    bank's own quarters are correlated with each other (the same bank's
    unobserved shocks persist across time), so treating every bank-
    quarter row as an independent observation (the OLS default) would
    understate standard errors and overstate significance; clustering is
    the standard fix. This does NOT correct residual degrees of freedom
    for the banks' own absorbed intercepts (a separate, smaller
    simplification worth being explicit about -- this project's "keep
    models simple and explainable," not a hidden shortcut); the point
    estimates (coefficients, predictions) are unaffected either way.
    Features: `category`'s own macro features (CATEGORY_MACRO_FEATURES)
    at `LAG`, the pandemic dummy (if `include_pandemic_dummy`), and
    log(average_balance) as the one bank characteristic (a simple,
    standard bank-size control). Returns a `PanelFEFit` bundling the
    fitted result with the per-bank training means `predict_panel_fe`
    needs to demean new data consistently, without leakage."""
    data = _add_bank_characteristics(bank_dataset)
    feature_columns = [*_macro_feature_columns(category, include_pandemic_dummy), "log_balance"]
    data = data.dropna(subset=[dependent_column, *feature_columns, "bank_id"])

    bank_training_means = data.groupby("bank_id")[[dependent_column, *feature_columns]].mean()
    means_aligned_x = bank_training_means.loc[data["bank_id"], feature_columns]
    means_aligned_x = means_aligned_x.set_axis(data.index)
    means_aligned_y = bank_training_means.loc[data["bank_id"], dependent_column]
    means_aligned_y = means_aligned_y.set_axis(data.index)
    demeaned_x = data[feature_columns] - means_aligned_x
    demeaned_y = data[dependent_column] - means_aligned_y

    result = sm.OLS(demeaned_y, demeaned_x).fit(
        cov_type="cluster", cov_kwds={"groups": data["bank_id"].to_numpy()}
    )
    return PanelFEFit(result=result, bank_training_means=bank_training_means)


def predict_panel_fe(
    fit: PanelFEFit,
    bank_dataset: pd.DataFrame,
    category: str,
    dependent_column: str,
    include_pandemic_dummy: bool = True,
) -> pd.Series:
    """Predicts LEVEL values (not just within-estimator deviations) for
    `bank_dataset` using `fit` (from `fit_panel_fe_model` -- `category`
    must be the SAME one `fit_panel_fe_model` was called with, so the
    feature set matches): each bank's fixed effect is recovered via the
    standard within-estimator identity alpha_i = ybar_i - beta @ xbar_i,
    using `fit.bank_training_means` -- the TRAINING sample's own per-bank
    means, never recomputed from `bank_dataset` itself (recomputing from
    the data being predicted, e.g. a forecast window, would leak that
    data into the demeaning basis -- the same leakage risk
    `fit_panel_fe_model` avoids). A bank absent from `fit.
    bank_training_means` (never seen in training) has no estimable fixed
    effect and is dropped -- an expected limitation of a bank fixed-
    effects model, not a bug: it cannot predict for a bank it never
    trained on."""
    data = _add_bank_characteristics(bank_dataset)
    feature_columns = [*_macro_feature_columns(category, include_pandemic_dummy), "log_balance"]
    data = data.dropna(subset=[*feature_columns, "bank_id"])
    data = data[data["bank_id"].isin(fit.bank_training_means.index)]

    coefficients = fit.result.params[feature_columns]
    bank_x_means = fit.bank_training_means.loc[data["bank_id"], feature_columns]
    bank_x_means = bank_x_means.set_axis(data.index)
    bank_y_mean = fit.bank_training_means.loc[data["bank_id"], dependent_column]
    bank_y_mean = bank_y_mean.set_axis(data.index)
    bank_fixed_effect = bank_y_mean - bank_x_means.dot(coefficients)

    return data[feature_columns].dot(coefficients) + bank_fixed_effect


def fit_gbm_model(
    bank_dataset: pd.DataFrame,
    category: str,
    dependent_column: str,
    include_pandemic_dummy: bool = True,
    max_training_rows: int = GBM_MAX_TRAINING_ROWS,
    random_state: int = GBM_RANDOM_STATE,
) -> GradientBoostingRegressor:
    """Model family 3: gradient boosting (sklearn's GradientBoostingRegressor
    -- 100 shallow trees, depth 3, for a simple/fast/explainable-via-
    feature_importances_ fit, not a tuned model) on the same features as
    `fit_panel_fe_model` (`category`'s own CATEGORY_MACRO_FEATURES at
    `LAG`, the pandemic dummy, log(average_balance)) -- no bank fixed
    effects (trees don't need them; log_balance is the bank
    characteristic instead). Subsamples to `max_training_rows` (fixed
    `random_state`) if the training data has more rows, to keep runtime
    bounded across the full Stage 4 sweep."""
    data = _add_bank_characteristics(bank_dataset)
    feature_columns = [*_macro_feature_columns(category, include_pandemic_dummy), "log_balance"]
    data = data.dropna(subset=[dependent_column, *feature_columns])
    if len(data) > max_training_rows:
        data = data.sample(n=max_training_rows, random_state=random_state)

    model = GradientBoostingRegressor(
        n_estimators=100, max_depth=3, learning_rate=0.05, random_state=random_state
    )
    model.fit(data[feature_columns], data[dependent_column])
    return model


def predict_gbm(
    model: GradientBoostingRegressor,
    bank_dataset: pd.DataFrame,
    category: str,
    include_pandemic_dummy: bool = True,
) -> pd.Series:
    """`category` must be the SAME one `fit_gbm_model` was called with,
    so the feature set matches."""
    data = _add_bank_characteristics(bank_dataset)
    feature_columns = [*_macro_feature_columns(category, include_pandemic_dummy), "log_balance"]
    valid = data.dropna(subset=feature_columns)
    predicted = model.predict(valid[feature_columns])
    return pd.Series(predicted, index=valid.index)


def aggregate_bank_predictions_to_industry_rate(
    bank_dataset: pd.DataFrame, predicted: pd.Series
) -> pd.Series:
    """Balance-weighted mean of a bank-level `predicted` Series (aligned
    to `bank_dataset`'s index -- e.g. from `predict_panel_fe`/
    `predict_gbm`) into a quarterly industry rate, for comparison against
    model family 1's own industry-level forecast on the same footing."""
    scoped = bank_dataset.loc[predicted.index]
    weight = scoped["average_balance"]
    weighted = predicted * weight
    grouped_weight = weight.groupby(scoped["quarter"], observed=True).sum()
    grouped_weighted = weighted.groupby(scoped["quarter"], observed=True).sum()
    return (grouped_weighted / grouped_weight).rename_axis("quarter")


def compute_bank_relative_levels(
    train_bank: pd.DataFrame, dependent_column: str, train_industry: pd.DataFrame
) -> pd.Series:
    """Model family 5 (anchored): each bank's own mean `dependent_column`
    over the training period, divided by the aggregate's own mean
    "industry_rate" over the SAME period -- a bank-specific multiplicative
    factor capturing how much riskier/safer that bank's own history has
    been relative to the industry as a whole. `train_industry`: a
    quarter-indexed frame with an "industry_rate" column (e.g. a slice of
    industry_history.build_long_industry_frame's output) covering the
    SAME quarters `train_bank` does -- using a different period's
    aggregate mean here than what `train_bank` itself covers would bias
    every bank's relative level by however much the aggregate moved
    between the two periods."""
    bank_means = train_bank.groupby("bank_id")[dependent_column].mean()
    aggregate_mean = train_industry["industry_rate"].mean()
    return bank_means / aggregate_mean


# Below this many quarters of its own training history, a bank's raw relative-level estimate
# (compute_bank_relative_levels) is shrunk toward the industry-neutral 1.0 -- see
# compute_shrunk_bank_relative_levels. 8 quarters (2 years) is a judgment call, not derived from
# this project's own data (there's no ground truth for "how many quarters is enough" without a
# separate validation study) -- a round, defensible minimum given this project's quarterly
# cadence, documented as an assumption like CATEGORY_WEIGHTED_AVERAGE_LIFE_QUARTERS.
MIN_QUARTERS_FOR_FULL_BANK_RELATIVE_LEVEL = 8


def compute_shrunk_bank_relative_levels(
    train_bank: pd.DataFrame,
    dependent_column: str,
    train_industry: pd.DataFrame,
    min_quarters: int = MIN_QUARTERS_FOR_FULL_BANK_RELATIVE_LEVEL,
) -> pd.Series:
    """`compute_bank_relative_levels`, SHRUNK toward the industry-neutral
    level of 1.0 for any bank with fewer than `min_quarters` of its own
    training history. A bank's raw mean-ratio estimate from a handful of
    quarters is noisy -- a single unusually high/low quarter can swing it
    a long way from any true long-run relative level -- so this blends
    linearly toward 1.0 by `n_quarters / min_quarters` (clipped to
    [0, 1]): full weight on the raw estimate once a bank has
    `min_quarters` or more of its own history, linearly less below that.
    A bank with zero usable quarters gets no raw estimate at all (dropped
    by `compute_bank_relative_levels`' own groupby, which only produces
    an entry per bank_id actually present in `train_bank`'s non-null
    rows) and so is absent here too -- callers (e.g. `projection.
    project_single_bank_nco_rate`) still need to handle an unseen bank
    explicitly, the same as the unshrunk function."""
    raw_levels = compute_bank_relative_levels(train_bank, dependent_column, train_industry)
    n_quarters = (
        train_bank.dropna(subset=[dependent_column]).groupby("bank_id")[dependent_column].count()
    )
    weight = (n_quarters.reindex(raw_levels.index) / min_quarters).clip(upper=1.0)
    return raw_levels * weight + 1.0 * (1.0 - weight)


def forecast_anchored_to_aggregate(
    bank_relative_levels: pd.Series, aggregate_forecast: pd.Series, test_bank: pd.DataFrame
) -> pd.Series:
    """Model family 5 (anchored): predicts each bank-quarter row in
    `test_bank` as that bank's relative level (`compute_bank_relative_
    levels`) times the aggregate model's own forecast for that row's
    quarter (`aggregate_forecast` -- typically the LONG-HISTORY aggregate
    model's forecast, `forecast_aggregate_dynamic` run against
    industry_history's frame, per this project's brief: anchoring a
    bank's projection to a longer, more crisis-informed aggregate view
    rather than the bank's own possibly too-short training window).
    A bank absent from `bank_relative_levels` (never seen in training)
    is dropped -- the same expected limitation as `predict_panel_fe`."""
    known_banks = test_bank["bank_id"].isin(bank_relative_levels.index)
    scoped = test_bank[known_banks]
    relative = scoped["bank_id"].map(bank_relative_levels)
    aggregate_value = scoped["quarter"].map(aggregate_forecast)
    predicted = (relative * aggregate_value).dropna()
    return predicted
