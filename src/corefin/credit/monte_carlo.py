"""Stage 6: Monte Carlo loss-distribution simulation -- combines TWO
sources of uncertainty Stage 5's own projection (projection.py) is
silent about, since it only ever produces a single deterministic path
per category/scenario:

1. MACRO PATH UNCERTAINTY: the Fed's own scenario is one hypothetical
   path, not the only one consistent with "severely adverse." Each draw
   perturbs the scenario's own macro feature columns (models.
   CORE_MACRO_FEATURES) with an independent, per-quarter shock drawn from
   that variable's OWN historical quarter-over-quarter standard
   deviation (`estimate_macro_shock_std`, computed from the REAL Fed
   historic actuals table, not assumed) -- a deliberately SIMPLE first
   cut: shocks are i.i.d. across both quarters and variables. A more
   sophisticated version would draw correlated shocks (across variables,
   since unemployment/HPI/CRE moves aren't independent in reality) from
   an estimated covariance matrix, and/or an AR(1) shock process instead
   of i.i.d. -- not attempted here, consistent with this project's
   "structured, not broad" rule for model changes.
2. MODEL RESIDUAL UNCERTAINTY: even with the macro path held fixed, the
   fitted model's own prediction has a residual standard error (`fit_
   monte_carlo_model`'s `residual_std`, from the SAME full-sample fit
   Stage 5 projects with). Added as i.i.d. N(0, residual_std) noise,
   independently to EACH projected quarter -- a documented simplification
   for the dynamic (AR-term) families (aggregate_ar/aggregate_long/
   anchored_to_aggregate): a more faithful AR(1)-process simulation would
   propagate each quarter's noisy outcome into the NEXT quarter's AR
   term (compounding the uncertainty through the recursion), not just
   add independent noise to the already-computed deterministic path
   after the fact. This is a real, documented scope limitation of this
   first Stage 6 build, not an oversight.

FITTING ONCE, SIMULATING MANY TIMES: refitting a model (especially GBM)
for every one of potentially thousands of draws would be far too slow
and is also unnecessary -- the SAME full-sample fit Stage 5 uses is
reused across every draw (`fit_monte_carlo_model`); only the (cheap)
forecast/predict step reruns per draw, on that draw's own perturbed
macro path. No function here touches the network or reads/writes files
-- see cli.py for orchestration.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pandas as pd

from corefin.credit import macro, models, projection

DEFAULT_N_DRAWS = 1_000
MONTE_CARLO_PERCENTILES: tuple[int, ...] = (5, 25, 50, 75, 95)


def estimate_macro_shock_std(
    macro_history: pd.DataFrame, variables: tuple[str, ...] = models.CORE_MACRO_FEATURES
) -> dict[str, float]:
    """The historical quarter-over-quarter standard deviation of each
    `variables` column in `macro_history` (the Fed's own real historic
    actuals table, quarter-indexed) -- the shock scale `perturb_scenario`
    draws from. Computed ONCE per `macro_history` (not re-estimated per
    draw). Silently skips a variable not present in `macro_history`."""
    stds: dict[str, float] = {}
    for variable in variables:
        if variable not in macro_history.columns:
            continue
        diffs = macro_history[variable].diff().dropna()
        stds[variable] = float(diffs.std())
    return stds


def perturb_scenario(
    scenario: pd.DataFrame, shock_std: dict[str, float], rng: np.random.Generator
) -> pd.DataFrame:
    """Returns a COPY of `scenario` with an independent N(0, shock_std[
    variable]) shock added to EACH quarter of each `shock_std` variable
    present in `scenario` -- see the module docstring for why this is
    i.i.d. (not correlated across variables or quarters), a deliberate
    first-cut simplification."""
    perturbed = scenario.copy()
    for variable, std in shock_std.items():
        if std <= 0.0 or variable not in perturbed.columns:
            continue
        shock = rng.normal(0.0, std, size=len(perturbed))
        perturbed[variable] = perturbed[variable].to_numpy() + shock
    return perturbed


@dataclass(frozen=True)
class MonteCarloModel:
    """One category's full-sample fit, reusable across many Monte Carlo
    draws without refitting. `forecast_fn(macro_lag_frame, projection_
    quarters) -> pd.Series`: closes over whatever `family`-specific
    fitted artifacts it needs (mirroring `projection.
    project_category_nco_rate`'s own per-family dispatch) so callers
    never need to know which family they're holding -- only the cheap
    forecast/predict step reruns per draw, on that draw's own perturbed
    macro_lag_frame."""

    category: str
    family: str
    residual_std: float
    forecast_fn: Callable[[pd.DataFrame, list[pd.Period]], pd.Series]


def fit_monte_carlo_model(
    category: str,
    family: str,
    category_train_dataset: pd.DataFrame,
    long_history_frame: pd.DataFrame | None,
) -> MonteCarloModel:
    """The ONE-TIME full-sample fit for `family` (same data/fit calls
    `projection.project_category_nco_rate` uses), bundled with its own
    residual standard error and a reusable forecast closure. `long_
    history_frame`, if given, is truncated to `category_train_dataset`'s
    own last quarter first -- the same overlap-prevention
    `project_category_nco_rate` applies (see that function's docstring)."""
    if family in projection.FAMILIES_REQUIRING_LONG_HISTORY and long_history_frame is None:
        raise ValueError(f"{family} requires long_history_frame")

    jump_off_quarter = category_train_dataset["quarter"].max()
    if long_history_frame is not None:
        long_history_frame = long_history_frame[long_history_frame["quarter"] <= jump_off_quarter]

    industry_series_full = models.build_industry_series(
        category_train_dataset, "winsorized_nco_rate"
    ).frame

    if family in ("aggregate_ar", "aggregate_long"):
        source_frame = long_history_frame if family == "aggregate_long" else industry_series_full
        result = models.fit_aggregate_model(source_frame, category, include_pandemic_dummy=True)
        residual_std = float(result.resid.std())

        def forecast_fn(
            macro_lag_frame: pd.DataFrame, projection_quarters: list[pd.Period]
        ) -> pd.Series:
            extended = pd.concat([source_frame, macro_lag_frame], ignore_index=True, sort=False)
            return models.forecast_aggregate_dynamic(
                result,
                extended,
                category,
                projection_quarters[0],
                projection_quarters[-1],
                include_pandemic_dummy=True,
            )

        return MonteCarloModel(category, family, residual_std, forecast_fn)

    last_quarter = category_train_dataset["quarter"].max()
    last_actual_bank_quarter = category_train_dataset[
        category_train_dataset["quarter"] == last_quarter
    ]

    if family == "panel_fe":
        fit = models.fit_panel_fe_model(
            category_train_dataset, category, "winsorized_nco_rate", include_pandemic_dummy=True
        )
        residual_std = float(fit.result.resid.std())

        def forecast_fn(
            macro_lag_frame: pd.DataFrame, projection_quarters: list[pd.Period]
        ) -> pd.Series:
            synthetic_future_bank = projection.build_synthetic_future_bank_frame(
                last_actual_bank_quarter, macro_lag_frame
            )
            predicted_bank = models.predict_panel_fe(
                fit,
                synthetic_future_bank,
                category,
                "winsorized_nco_rate",
                include_pandemic_dummy=True,
            )
            return models.aggregate_bank_predictions_to_industry_rate(
                synthetic_future_bank, predicted_bank
            )

        return MonteCarloModel(category, family, residual_std, forecast_fn)

    if family == "gbm":
        gbm_model = models.fit_gbm_model(
            category_train_dataset, category, "winsorized_nco_rate", include_pandemic_dummy=True
        )
        predicted_train = models.predict_gbm(
            gbm_model, category_train_dataset, category, include_pandemic_dummy=True
        )
        actual_train = category_train_dataset.loc[predicted_train.index, "winsorized_nco_rate"]
        residual_std = float((actual_train - predicted_train).std())

        def forecast_fn(
            macro_lag_frame: pd.DataFrame, projection_quarters: list[pd.Period]
        ) -> pd.Series:
            synthetic_future_bank = projection.build_synthetic_future_bank_frame(
                last_actual_bank_quarter, macro_lag_frame
            )
            predicted_bank = models.predict_gbm(
                gbm_model, synthetic_future_bank, category, include_pandemic_dummy=True
            )
            return models.aggregate_bank_predictions_to_industry_rate(
                synthetic_future_bank, predicted_bank
            )

        return MonteCarloModel(category, family, residual_std, forecast_fn)

    if family == "anchored_to_aggregate":
        long_result = models.fit_aggregate_model(
            long_history_frame, category, include_pandemic_dummy=True
        )
        # anchored_to_aggregate has no coefficients of its own -- its
        # forecast SHAPE comes directly from long_result, so its own
        # residual uncertainty is taken from that same fit (the same
        # "inherits from aggregate_long" simplification projection.
        # select_projection_model's sign check already uses).
        residual_std = float(long_result.resid.std())
        long_train_matching_bank_period = long_history_frame[
            long_history_frame["quarter"].isin(category_train_dataset["quarter"])
        ]
        relative_levels = models.compute_bank_relative_levels(
            category_train_dataset, "winsorized_nco_rate", long_train_matching_bank_period
        )

        def forecast_fn(
            macro_lag_frame: pd.DataFrame, projection_quarters: list[pd.Period]
        ) -> pd.Series:
            long_extended = pd.concat(
                [long_history_frame, macro_lag_frame], ignore_index=True, sort=False
            )
            long_forecast = models.forecast_aggregate_dynamic(
                long_result,
                long_extended,
                category,
                projection_quarters[0],
                projection_quarters[-1],
                include_pandemic_dummy=True,
            )
            synthetic_future_bank = projection.build_synthetic_future_bank_frame(
                last_actual_bank_quarter, macro_lag_frame
            )
            predicted_bank = models.forecast_anchored_to_aggregate(
                relative_levels, long_forecast, synthetic_future_bank
            )
            return models.aggregate_bank_predictions_to_industry_rate(
                synthetic_future_bank, predicted_bank
            )

        return MonteCarloModel(category, family, residual_std, forecast_fn)

    raise ValueError(f"unknown model family {family!r}")


def simulate_category_losses(
    mc_model: MonteCarloModel,
    macro_history: pd.DataFrame,
    scenario: pd.DataFrame,
    shock_std: dict[str, float],
    n_draws: int = DEFAULT_N_DRAWS,
    n_quarters: int | None = None,
    random_state: int = 0,
) -> np.ndarray:
    """Runs `n_draws` Monte Carlo draws combining macro path uncertainty
    (`perturb_scenario`, using `shock_std` from `estimate_macro_shock_
    std`) and `mc_model`'s own residual uncertainty (see the module
    docstring for both). Returns an array of `n_draws` cumulative loss
    rates (`projection.cumulative_loss_rate`) over the FIRST `n_quarters`
    of `scenario` (defaults to `scenario`'s own full length). Draws are
    NOT floored at zero -- a legitimate, informative possibility in the
    simulated distribution's lower tail (e.g. "a 5% chance this book is
    roughly breakeven"), unlike the single POINT allowance estimate
    `projection.lifetime_expected_loss_rate` floors."""
    n_quarters = n_quarters if n_quarters is not None else len(scenario)
    rng = np.random.default_rng(random_state)
    projection_quarters = list(scenario.index)
    losses = np.empty(n_draws)
    for draw in range(n_draws):
        perturbed_scenario = perturb_scenario(scenario, shock_std, rng)
        full_macro_path = macro.build_full_macro_path(macro_history, perturbed_scenario)
        macro_lag_frame = projection.build_projection_macro_lag_frame(
            full_macro_path, projection_quarters
        )
        deterministic_path = mc_model.forecast_fn(macro_lag_frame, projection_quarters)
        residual_noise = rng.normal(0.0, mc_model.residual_std, size=len(deterministic_path))
        simulated_path = deterministic_path + pd.Series(
            residual_noise, index=deterministic_path.index
        )
        losses[draw] = projection.cumulative_loss_rate(simulated_path, n_quarters)
    return losses


@dataclass(frozen=True)
class LossDistributionSummary:
    """`percentiles`: {percentile (int, e.g. 5) -> cumulative loss rate
    at that percentile}, from MONTE_CARLO_PERCENTILES."""

    category: str
    scenario: str
    family: str
    n_draws: int
    mean: float
    std: float
    percentiles: dict[int, float]


def summarize_loss_distribution(
    category: str, scenario_name: str, family: str, losses: np.ndarray
) -> LossDistributionSummary:
    percentiles = {p: float(np.percentile(losses, p)) for p in MONTE_CARLO_PERCENTILES}
    return LossDistributionSummary(
        category=category,
        scenario=scenario_name,
        family=family,
        n_draws=len(losses),
        mean=float(losses.mean()),
        std=float(losses.std()),
        percentiles=percentiles,
    )
