"""Stage 6: sensitivities -- a tornado (vary one deal/PPNR assumption
at a time, holding the rest at the base case) and a 2D grid (price x
cost saves), both over the SAME four outputs: Year 2 EPS accretion
(excluding one-time charges -- the "economics, not accounting" view),
TBV dilution at close, TBV earnback (in years; `None` if beyond the
horizon), and the pro forma combined entity's minimum CET1 ratio under
the Fed's severely adverse scenario (Stage 5).

BASE CASE CONTEXT (`DealContext`): one fully-built baseline-scenario
`BankModelResult` per bank (already extended to the deal horizon) and
each bank's own severely-adverse `CreditLossProjection` (already
truncated to the Fed's 9-quarter window), plus the raw inputs needed to
CHEAPLY re-evaluate the deal at a DIFFERENT assumption -- all of this
is already built by `corefin.ma.cli.run`'s own pipeline by the time
Stage 5 finishes, so `DealContext` just packages it for reuse here
rather than re-fetching or re-parsing anything.

WHAT RE-EVALUATING COSTS: varying a `DealConfig` field (price_to_tbv,
credit_mark_pct, rate_mark_pct, cost_save_pct_of_target_noninterest_
expense, cdi_pct_of_core_deposits, stock_pct) only reruns `run_deal_
model`/the Stage 4 accretion functions/`run_stress_test` -- the
underlying bank models are UNCHANGED and reused as-is. Varying
`nim_beta` (PPNR stress) requires rebuilding those bank models too (a
new `BankConfig.ppnr_stress`), done here via `run_bank_model` directly
(the credit-loss projection is already extended/truncated once, in
`DealContext`, so this skips `horizon.py`'s own extend/truncate step on
every call) rather than the full `build_deal_horizon_bank_result`/
`build_stress_bank_result` wrappers. `nim_beta` overrides BOTH banks to
the SAME shared value (even though each bank's own calibrated beta
differs at the true base case) -- a single scalar driver, not two
independent ones, matching the brief's own "NIM beta" as ONE tornado/
Monte Carlo factor."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from corefin.bank.model import BankModelResult, run_bank_model
from corefin.bank.schema import BankConfig, BankOpeningBalance, PpnrStressConfig
from corefin.credit.interface import CreditLossProjection
from corefin.ma.model import run_deal_model
from corefin.ma.schema import DealConfig
from corefin.ma.stress import run_stress_test

DEFAULT_MONTE_CARLO_DRAWS = 500


@dataclass(frozen=True)
class DealContext:
    acquirer_opening: BankOpeningBalance
    target_opening: BankOpeningBalance
    acquirer_bank_config: BankConfig  # own calibrated ppnr_stress baked in
    target_bank_config: BankConfig

    acquirer_baseline_result: BankModelResult  # already extended to the deal horizon
    target_baseline_result: BankModelResult

    acquirer_baseline_projection: CreditLossProjection  # same horizon as acquirer_baseline_result
    target_baseline_projection: CreditLossProjection
    acquirer_severely_adverse_projection: CreditLossProjection  # truncated to the Fed's 9Q window
    target_severely_adverse_projection: CreditLossProjection

    baseline_rate_path_pp: np.ndarray  # aligned to acquirer/target_baseline_projection's horizon
    severely_adverse_rate_path_pp: (
        np.ndarray
    )  # aligned to the severely-adverse projections' horizon

    base_config: DealConfig
    acquirer_unexplained_cet1_residual_mm: float


@dataclass(frozen=True)
class SensitivityMetrics:
    year2_accretion_pct_excl_one_time: float | None  # None if not meaningful (standalone <= 0)
    tbv_dilution_at_close_pct: float
    earnback_years: float | None  # None if beyond the deal horizon
    minimum_stressed_cet1_ratio: float  # pro forma combined, severely adverse, Fed 9Q window


def _rebuild_baseline_results(
    ctx: DealContext, nim_beta: float
) -> tuple[BankModelResult, BankModelResult]:
    acquirer_config = ctx.acquirer_bank_config.model_copy(
        update={"ppnr_stress": PpnrStressConfig(nim_beta=nim_beta)}
    )
    target_config = ctx.target_bank_config.model_copy(
        update={"ppnr_stress": PpnrStressConfig(nim_beta=nim_beta)}
    )
    acquirer_result = run_bank_model(
        ctx.acquirer_opening,
        ctx.acquirer_baseline_projection,
        acquirer_config,
        ctx.acquirer_baseline_projection.timeline,
        rate_path_pp=ctx.baseline_rate_path_pp,
    )
    target_result = run_bank_model(
        ctx.target_opening,
        ctx.target_baseline_projection,
        target_config,
        ctx.target_baseline_projection.timeline,
        rate_path_pp=ctx.baseline_rate_path_pp,
    )
    return acquirer_result, target_result


def evaluate_deal(
    ctx: DealContext,
    price_to_tbv: float | None = None,
    credit_mark_pct: float | None = None,
    rate_mark_pct: float | None = None,
    cost_save_pct: float | None = None,
    cdi_pct: float | None = None,
    stock_pct: float | None = None,
    nim_beta: float | None = None,
) -> SensitivityMetrics:
    """Each `None` parameter stays at `ctx.base_config`'s own value (or,
    for `nim_beta`, at each bank's own calibrated value, unchanged --
    see module docstring). Only the ones actually given override the
    base case."""
    config = ctx.base_config
    if price_to_tbv is not None:
        config = config.model_copy(
            update={
                "consideration": config.consideration.model_copy(
                    update={"price_to_tbv": price_to_tbv}
                )
            }
        )
    if stock_pct is not None:
        config = config.model_copy(
            update={
                "consideration": config.consideration.model_copy(update={"stock_pct": stock_pct})
            }
        )
    if credit_mark_pct is not None:
        config = config.model_copy(
            update={
                "credit_mark": config.credit_mark.model_copy(
                    update={"credit_mark_pct": credit_mark_pct}
                )
            }
        )
    if rate_mark_pct is not None:
        config = config.model_copy(
            update={
                "rate_mark": config.rate_mark.model_copy(update={"rate_mark_pct": rate_mark_pct})
            }
        )
    if cost_save_pct is not None:
        config = config.model_copy(
            update={
                "cost_saves": config.cost_saves.model_copy(
                    update={"cost_save_pct_of_target_noninterest_expense": cost_save_pct}
                )
            }
        )
    if cdi_pct is not None:
        config = config.model_copy(
            update={"cdi": config.cdi.model_copy(update={"cdi_pct_of_core_deposits": cdi_pct})}
        )

    if nim_beta is not None:
        acquirer_result, target_result = _rebuild_baseline_results(ctx, nim_beta)
        acquirer_bank_config = ctx.acquirer_bank_config.model_copy(
            update={"ppnr_stress": PpnrStressConfig(nim_beta=nim_beta)}
        )
        target_bank_config = ctx.target_bank_config.model_copy(
            update={"ppnr_stress": PpnrStressConfig(nim_beta=nim_beta)}
        )
    else:
        acquirer_result, target_result = ctx.acquirer_baseline_result, ctx.target_baseline_result
        acquirer_bank_config, target_bank_config = ctx.acquirer_bank_config, ctx.target_bank_config

    deal_result = run_deal_model(
        acquirer_result, target_result, acquirer_bank_config, target_bank_config, config
    )

    year2 = deal_result.eps_accretion.annual()["Year 2"]
    year2_accretion = (
        year2.accretion_dilution_pct_excl_one_time if year2.accretion_pct_meaningful else None
    )

    stress_result = run_stress_test(
        ctx.acquirer_opening,
        ctx.target_opening,
        ctx.acquirer_severely_adverse_projection,
        ctx.target_severely_adverse_projection,
        acquirer_bank_config,
        target_bank_config,
        deal_result,
        config,
        acquirer_unexplained_cet1_residual_mm=ctx.acquirer_unexplained_cet1_residual_mm,
        jumpoff_rate_pp=float(ctx.severely_adverse_rate_path_pp[0]),
        projected_rate_path_pp=ctx.severely_adverse_rate_path_pp[1:],
    )

    return SensitivityMetrics(
        year2_accretion_pct_excl_one_time=year2_accretion,
        tbv_dilution_at_close_pct=deal_result.tbv_earnback.tbv_dilution_at_close_pct,
        earnback_years=deal_result.tbv_earnback.earnback_years,
        minimum_stressed_cet1_ratio=stress_result.pro_forma_combined.minimum_cet1_ratio,
    )


@dataclass(frozen=True)
class TornadoDriver:
    name: str
    param: str  # evaluate_deal's own kwarg name
    low: float
    high: float


def build_default_tornado_drivers(base_config: DealConfig) -> list[TornadoDriver]:
    """Illustrative +/- ranges around `base_config`'s own values -- not
    derived from any real deal-pricing distribution -- clipped to each
    field's own valid range (e.g. stock_pct can't exceed 1.0). `nim_beta`
    has no single base value to perturb (each bank is calibrated
    independently -- see module docstring), so its own range is a fixed
    illustrative [0.15, 0.6] regardless of the base config."""
    c = base_config
    return [
        TornadoDriver(
            "Price (P/TBV)",
            "price_to_tbv",
            max(c.consideration.price_to_tbv - 0.3, 0.1),
            c.consideration.price_to_tbv + 0.3,
        ),
        TornadoDriver(
            "Credit mark",
            "credit_mark_pct",
            max(c.credit_mark.credit_mark_pct - 0.01, 0.0),
            c.credit_mark.credit_mark_pct + 0.01,
        ),
        TornadoDriver(
            "Rate mark",
            "rate_mark_pct",
            c.rate_mark.rate_mark_pct - 0.02,
            c.rate_mark.rate_mark_pct + 0.02,
        ),
        TornadoDriver(
            "Cost saves",
            "cost_save_pct",
            max(c.cost_saves.cost_save_pct_of_target_noninterest_expense - 0.10, 0.0),
            min(c.cost_saves.cost_save_pct_of_target_noninterest_expense + 0.10, 1.0),
        ),
        TornadoDriver(
            "CDI",
            "cdi_pct",
            max(c.cdi.cdi_pct_of_core_deposits - 0.01, 0.0),
            c.cdi.cdi_pct_of_core_deposits + 0.01,
        ),
        TornadoDriver(
            "Stock % of consideration",
            "stock_pct",
            max(c.consideration.stock_pct - 0.2, 0.0),
            min(c.consideration.stock_pct + 0.2, 1.0),
        ),
        TornadoDriver("NIM beta", "nim_beta", 0.15, 0.6),
    ]


@dataclass(frozen=True)
class TornadoRow:
    driver_name: str
    low_value: float
    high_value: float
    low_metrics: SensitivityMetrics
    high_metrics: SensitivityMetrics


@dataclass(frozen=True)
class TornadoResult:
    base_metrics: SensitivityMetrics
    rows: list[TornadoRow]

    def sorted_by(self, metric: str) -> list[TornadoRow]:
        """`rows` sorted by that metric's own |high - low| range,
        descending (the standard tornado-chart ordering: the widest bar
        on top) -- rows where either end is `None` (not meaningful) sort
        last."""

        def _range(row: TornadoRow) -> float:
            low = getattr(row.low_metrics, metric)
            high = getattr(row.high_metrics, metric)
            if low is None or high is None:
                return -1.0
            return abs(high - low)

        return sorted(self.rows, key=_range, reverse=True)


def compute_tornado(ctx: DealContext, drivers: list[TornadoDriver] | None = None) -> TornadoResult:
    if drivers is None:
        drivers = build_default_tornado_drivers(ctx.base_config)
    base_metrics = evaluate_deal(ctx)
    rows = [
        TornadoRow(
            driver_name=driver.name,
            low_value=driver.low,
            high_value=driver.high,
            low_metrics=evaluate_deal(ctx, **{driver.param: driver.low}),
            high_metrics=evaluate_deal(ctx, **{driver.param: driver.high}),
        )
        for driver in drivers
    ]
    return TornadoResult(base_metrics=base_metrics, rows=rows)


@dataclass(frozen=True)
class GridResult:
    price_values: np.ndarray
    cost_save_values: np.ndarray
    year2_accretion_pct: (
        np.ndarray
    )  # shape (len(cost_save_values), len(price_values)); NaN where n/m
    earnback_years: np.ndarray  # same shape; NaN where beyond horizon


def compute_price_cost_save_grid(
    ctx: DealContext, price_values: np.ndarray, cost_save_values: np.ndarray
) -> GridResult:
    """Year-2 EPS accretion (excl. one-time) and TBV earnback across a
    price (P/TBV) x cost-saves grid, holding every other assumption at
    the base case."""
    n_rows, n_cols = len(cost_save_values), len(price_values)
    accretion = np.full((n_rows, n_cols), np.nan)
    earnback = np.full((n_rows, n_cols), np.nan)
    for i, cost_save_pct in enumerate(cost_save_values):
        for j, price_to_tbv in enumerate(price_values):
            metrics = evaluate_deal(
                ctx, price_to_tbv=float(price_to_tbv), cost_save_pct=float(cost_save_pct)
            )
            if metrics.year2_accretion_pct_excl_one_time is not None:
                accretion[i, j] = metrics.year2_accretion_pct_excl_one_time
            if metrics.earnback_years is not None:
                earnback[i, j] = metrics.earnback_years
    return GridResult(
        price_values=np.asarray(price_values),
        cost_save_values=np.asarray(cost_save_values),
        year2_accretion_pct=accretion,
        earnback_years=earnback,
    )


@dataclass(frozen=True)
class MonteCarloResult:
    n_draws: int
    seed: int
    credit_mark_pct: np.ndarray
    cost_save_pct: np.ndarray
    rate_mark_pct: np.ndarray
    nim_beta: np.ndarray
    year2_accretion_pct: np.ndarray  # NaN where not meaningful
    tbv_dilution_at_close_pct: np.ndarray
    earnback_years: np.ndarray  # NaN where beyond horizon
    minimum_stressed_cet1_ratio: np.ndarray

    def percentiles(
        self, metric: str, percentiles: tuple[int, ...] = (5, 25, 50, 75, 95)
    ) -> dict[int, float]:
        values = getattr(self, metric)
        finite = values[np.isfinite(values)]
        if finite.size == 0:
            return {p: float("nan") for p in percentiles}
        return {p: float(np.percentile(finite, p)) for p in percentiles}

    def probability_beyond_horizon(self) -> float:
        """Fraction of draws whose TBV earnback never happens within the
        deal horizon (`earnback_years` is NaN)."""
        return float(np.mean(~np.isfinite(self.earnback_years)))


def run_monte_carlo(
    ctx: DealContext,
    n_draws: int = DEFAULT_MONTE_CARLO_DRAWS,
    seed: int = 0,
    credit_mark_pct_range: tuple[float, float] | None = None,
    cost_save_pct_range: tuple[float, float] | None = None,
    rate_mark_pct_range: tuple[float, float] | None = None,
    nim_beta_range: tuple[float, float] | None = None,
) -> MonteCarloResult:
    """Draws `n_draws` i.i.d. UNIFORM samples -- the "shared scenario
    layer" convention this project's own `corefin.scenarios.generator`
    uses for its simple stochastic mode: a seeded `np.random.default_rng`,
    every draw's own array generated in ONE vectorized call, not a per-
    draw Python RNG call -- over each of the four deal assumptions
    INDEPENDENTLY (no assumed correlation between them; a real joint
    distribution would need historical deal data this project doesn't
    have, a documented simplification), then evaluates the deal once per
    draw (each evaluation itself fully vectorized across periods, via
    `evaluate_deal`)."""
    base = ctx.base_config
    cm_lo, cm_hi = credit_mark_pct_range or (
        max(base.credit_mark.credit_mark_pct - 0.01, 0.0),
        base.credit_mark.credit_mark_pct + 0.01,
    )
    cs_lo, cs_hi = cost_save_pct_range or (
        max(base.cost_saves.cost_save_pct_of_target_noninterest_expense - 0.10, 0.0),
        min(base.cost_saves.cost_save_pct_of_target_noninterest_expense + 0.10, 1.0),
    )
    rm_lo, rm_hi = rate_mark_pct_range or (
        base.rate_mark.rate_mark_pct - 0.02,
        base.rate_mark.rate_mark_pct + 0.02,
    )
    nb_lo, nb_hi = nim_beta_range or (0.15, 0.6)

    rng = np.random.default_rng(seed)
    credit_mark_draws = rng.uniform(cm_lo, cm_hi, size=n_draws)
    cost_save_draws = rng.uniform(cs_lo, cs_hi, size=n_draws)
    rate_mark_draws = rng.uniform(rm_lo, rm_hi, size=n_draws)
    nim_beta_draws = rng.uniform(nb_lo, nb_hi, size=n_draws)

    year2_accretion = np.full(n_draws, np.nan)
    tbv_dilution = np.full(n_draws, np.nan)
    earnback = np.full(n_draws, np.nan)
    min_cet1 = np.full(n_draws, np.nan)

    for i in range(n_draws):
        metrics = evaluate_deal(
            ctx,
            credit_mark_pct=float(credit_mark_draws[i]),
            cost_save_pct=float(cost_save_draws[i]),
            rate_mark_pct=float(rate_mark_draws[i]),
            nim_beta=float(nim_beta_draws[i]),
        )
        if metrics.year2_accretion_pct_excl_one_time is not None:
            year2_accretion[i] = metrics.year2_accretion_pct_excl_one_time
        tbv_dilution[i] = metrics.tbv_dilution_at_close_pct
        if metrics.earnback_years is not None:
            earnback[i] = metrics.earnback_years
        min_cet1[i] = metrics.minimum_stressed_cet1_ratio

    return MonteCarloResult(
        n_draws=n_draws,
        seed=seed,
        credit_mark_pct=credit_mark_draws,
        cost_save_pct=cost_save_draws,
        rate_mark_pct=rate_mark_draws,
        nim_beta=nim_beta_draws,
        year2_accretion_pct=year2_accretion,
        tbv_dilution_at_close_pct=tbv_dilution,
        earnback_years=earnback,
        minimum_stressed_cet1_ratio=min_cet1,
    )
