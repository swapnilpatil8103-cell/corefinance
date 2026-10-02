"""Stage 7: `CreditLossProjection` -- the Credit-Loss Forecasting Engine's
(Project #7, `credit/__init__.py`) typed output, aligned to the shared
`timeline.Timeline` used throughout the rest of corefin, so a downstream
consumer (Project #2, the Bank M&A CET1 & Accretion Simulator, or
anything else) gets per-category AND total NCOs/provisions/allowance/
NPLs/balances as plain numpy arrays instead of having to parse this
engine's own CSV outputs. No function here touches the network or reads/
writes files -- see cli.py for orchestration.

UNITS: every dollar array is in $mm (this engine's own internal units
are thousands of dollars, per the FFIEC Call Report convention -- divide
by 1,000 once here rather than carrying that convention outward). Rate
arrays (`nco_rate`, `npl_ratio`) stay as decimal fractions (0.05, not 5%).

NPL SIMPLIFICATION: Stage 5 deliberately projects only the NCO rate
("CECL allowances fund future charge-offs, so the NCO rate is the
relevant loss-rate driver, not the NPL ratio" -- projection.py's own
docstring) -- there is no Stage 5 NPL *projection* to draw from. Rather
than build a second, parallel model-selection pipeline just for NPL,
`npl_ratio`/`npl_mm` here hold the REALIZED (jump-off, 2025Q4) NPL ratio
CONSTANT across the whole projection horizon -- a simple, explicitly
documented assumption (NPL is a slower-moving, lagging indicator than
NCO in practice, so "unchanged" is a far less wrong placeholder for it
than for NCO), not a second full projection effort.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from corefin.credit import backtest, models, monte_carlo, projection
from corefin.timeline import Timeline

DOLLARS_TO_MM = 1.0 / 1_000.0  # this engine's own units are thousands of dollars


@dataclass(frozen=True)
class CreditLossProjection:
    """One scenario's projection across `categories`, aligned to
    `timeline` (quarter 0 is the jump-off/last-actual quarter,
    `timeline.is_projection[0]` is False; the rest are the projected
    quarters). Every `..._mm` array has shape (n_categories, n_periods);
    every `..._total_mm` array has shape (n_periods,) (summed across
    categories). `nco_rate`/`npl_ratio` (shape (n_categories, n_periods))
    are decimal-fraction rates, not dollars. `monte_carlo_mean`/`
    monte_carlo_percentiles` (shape (n_categories,) each) are the
    cumulative loss RATE distribution (Stage 6) over the full projection
    horizon -- `monte_carlo.DEFAULT_N_DRAWS`/`MONTE_CARLO_PERCENTILES`.
    `bank_identifier`: None for the industry-wide projection, else the
    single bank's own `bank_id` (RSSD ID) this projection is for."""

    timeline: Timeline
    categories: tuple[str, ...]
    scenario_name: str
    bank_identifier: str | None

    balance_mm: np.ndarray
    net_charge_off_mm: np.ndarray
    provision_expense_mm: np.ndarray
    allowance_mm: np.ndarray
    npl_mm: np.ndarray
    nco_rate: np.ndarray
    npl_ratio: np.ndarray

    monte_carlo_mean: np.ndarray
    monte_carlo_percentiles: dict[int, np.ndarray]

    def __post_init__(self) -> None:
        n_categories = len(self.categories)
        n_periods = self.timeline.n_periods
        expected_shape = (n_categories, n_periods)
        for name in (
            "balance_mm",
            "net_charge_off_mm",
            "provision_expense_mm",
            "allowance_mm",
            "npl_mm",
            "nco_rate",
            "npl_ratio",
        ):
            array = getattr(self, name)
            if array.shape != expected_shape:
                raise ValueError(f"{name} has shape {array.shape}, expected {expected_shape}")
        if self.monte_carlo_mean.shape != (n_categories,):
            raise ValueError(
                f"monte_carlo_mean has shape {self.monte_carlo_mean.shape}, "
                f"expected ({n_categories},)"
            )
        for percentile, array in self.monte_carlo_percentiles.items():
            if array.shape != (n_categories,):
                raise ValueError(
                    f"monte_carlo_percentiles[{percentile}] has shape {array.shape}, "
                    f"expected ({n_categories},)"
                )

    @property
    def balance_total_mm(self) -> np.ndarray:
        return self.balance_mm.sum(axis=0)

    @property
    def net_charge_off_total_mm(self) -> np.ndarray:
        return np.nansum(self.net_charge_off_mm, axis=0)

    @property
    def provision_expense_total_mm(self) -> np.ndarray:
        return np.nansum(self.provision_expense_mm, axis=0)

    @property
    def allowance_total_mm(self) -> np.ndarray:
        return self.allowance_mm.sum(axis=0)

    @property
    def npl_total_mm(self) -> np.ndarray:
        return self.npl_mm.sum(axis=0)


def _build_timeline(cecl_frames: dict[str, pd.DataFrame]) -> tuple[Timeline, list[pd.Period]]:
    """All categories share the SAME quarter index by construction (the
    same jump-off quarter and the same scenario) -- take it from the
    first category's own CECL frame."""
    quarters = list(next(iter(cecl_frames.values())).index)
    first_quarter = quarters[0]
    timeline = Timeline.quarterly(
        n_periods=len(quarters),
        n_historical=1,  # quarter 0 is the jump-off, the rest are projected
        start_year=first_quarter.year,
        start_quarter=first_quarter.quarter,
    )
    return timeline, quarters


def build_credit_loss_projection(
    categories: list[str],
    selected_family_by_category: dict[str, str],
    training: pd.DataFrame,
    macro_history: pd.DataFrame,
    scenario: pd.DataFrame,
    scenario_name: str,
    long_history_frames: dict[tuple[str, str], pd.DataFrame],
    bank_id: str | None = None,
    n_monte_carlo_draws: int = monte_carlo.DEFAULT_N_DRAWS,
    monte_carlo_random_state: int = 0,
) -> CreditLossProjection:
    """Builds one `CreditLossProjection` covering every `categories`
    entry under `scenario`. `selected_family_by_category`: {category ->
    model family}, e.g. `project`'s own `model_selection.csv`'s
    "selected" rows. `training`: the FULL modeling dataset (all
    categories; filtered here per category) -- NPL columns are added if
    not already present (`backtest.add_npl_ratio_columns`). If `bank_id`
    is given, projects that ONE bank (`projection.
    project_single_bank_nco_rate`, using its own balance and relative
    level) instead of the industry; the Monte Carlo step is skipped for
    single-bank mode (Stage 6's macro/residual simulation is defined at
    the industry level only -- see the module docstring's NPL note for
    the same kind of documented simplification)."""
    if "winsorized_npl_ratio" not in training.columns:
        training = backtest.add_npl_ratio_columns(training)

    cecl_frames: dict[str, pd.DataFrame] = {}
    nco_rate_by_category: dict[str, pd.Series] = {}
    monte_carlo_summaries: dict[str, monte_carlo.LossDistributionSummary] = {}

    for category in categories:
        category_train = training[training["category"] == category]
        family = selected_family_by_category[category]
        long_history_frame = long_history_frames.get((category, "nco_rate"))

        if bank_id is None:
            forecast = projection.project_category_nco_rate(
                category, family, category_train, macro_history, scenario, long_history_frame
            )
            realized = models.build_industry_series(category_train, "winsorized_nco_rate").frame
            realized_rate = realized.set_index("quarter")["industry_rate"]
            last_quarter = category_train["quarter"].max()
            starting_balance = category_train[category_train["quarter"] == last_quarter][
                "average_balance"
            ].sum()
        else:
            forecast = projection.project_single_bank_nco_rate(
                category,
                family,
                bank_id,
                category_train,
                macro_history,
                scenario,
                long_history_frame,
            )
            bank_rows = category_train[category_train["bank_id"] == bank_id]
            realized_rate = bank_rows.set_index("quarter")["winsorized_nco_rate"]
            last_quarter = category_train["quarter"].max()
            starting_balance = bank_rows[bank_rows["quarter"] == last_quarter][
                "average_balance"
            ].sum()

        cecl_frames[category] = projection.build_cecl_projection(
            realized_rate, forecast, category, starting_balance
        )
        nco_rate_by_category[category] = forecast

        if bank_id is None:
            mc_model = monte_carlo.fit_monte_carlo_model(
                category, family, category_train, long_history_frame
            )
            shock_std = monte_carlo.estimate_macro_shock_std(macro_history)
            losses = monte_carlo.simulate_category_losses(
                mc_model,
                macro_history,
                scenario,
                shock_std,
                n_draws=n_monte_carlo_draws,
                random_state=monte_carlo_random_state,
            )
            monte_carlo_summaries[category] = monte_carlo.summarize_loss_distribution(
                category, scenario_name, family, losses
            )

    timeline, quarters = _build_timeline(cecl_frames)
    n_categories, n_periods = len(categories), len(quarters)

    balance_mm = np.empty((n_categories, n_periods))
    net_charge_off_mm = np.empty((n_categories, n_periods))
    provision_expense_mm = np.empty((n_categories, n_periods))
    allowance_mm = np.empty((n_categories, n_periods))
    npl_mm = np.empty((n_categories, n_periods))
    nco_rate_arr = np.empty((n_categories, n_periods))
    npl_ratio_arr = np.empty((n_categories, n_periods))

    for i, category in enumerate(categories):
        cecl = cecl_frames[category].loc[quarters]
        if bank_id is None:
            category_train = training[training["category"] == category]
        else:
            category_train = training[
                (training["category"] == category) & (training["bank_id"] == bank_id)
            ]
        last_quarter = category_train["quarter"].max()
        last_actual = category_train[category_train["quarter"] == last_quarter]
        jump_off_npl_ratio = float(
            (last_actual["winsorized_npl_ratio"] * last_actual["average_balance"]).sum()
            / last_actual["average_balance"].sum()
        )

        starting_balance = last_actual["average_balance"].sum()
        balance_mm[i] = np.full(n_periods, starting_balance * DOLLARS_TO_MM)
        net_charge_off_mm[i] = cecl["net_charge_off"].to_numpy() * DOLLARS_TO_MM
        provision_expense_mm[i] = cecl["provision_expense"].to_numpy() * DOLLARS_TO_MM
        allowance_mm[i] = cecl["allowance_required"].to_numpy() * DOLLARS_TO_MM
        npl_ratio_arr[i] = jump_off_npl_ratio
        npl_mm[i] = starting_balance * jump_off_npl_ratio * DOLLARS_TO_MM

        rate_series = nco_rate_by_category[category].reindex(quarters)
        nco_rate_arr[i] = rate_series.to_numpy()

    if bank_id is None:
        monte_carlo_mean = np.array(
            [monte_carlo_summaries[c].mean for c in categories]
        )
        percentile_set = monte_carlo.MONTE_CARLO_PERCENTILES
        monte_carlo_percentiles = {
            p: np.array([monte_carlo_summaries[c].percentiles[p] for c in categories])
            for p in percentile_set
        }
    else:
        monte_carlo_mean = np.full(n_categories, np.nan)
        monte_carlo_percentiles = {
            p: np.full(n_categories, np.nan) for p in monte_carlo.MONTE_CARLO_PERCENTILES
        }

    return CreditLossProjection(
        timeline=timeline,
        categories=tuple(categories),
        scenario_name=scenario_name,
        bank_identifier=bank_id,
        balance_mm=balance_mm,
        net_charge_off_mm=net_charge_off_mm,
        provision_expense_mm=provision_expense_mm,
        allowance_mm=allowance_mm,
        npl_mm=npl_mm,
        nco_rate=nco_rate_arr,
        npl_ratio=npl_ratio_arr,
        monte_carlo_mean=monte_carlo_mean,
        monte_carlo_percentiles=monte_carlo_percentiles,
    )
