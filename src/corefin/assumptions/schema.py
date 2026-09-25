"""Pydantic assumption schema loaded from YAML.

Units (documented once, here, and nowhere else):
  - All dollar amounts are in $mm.
  - All rates, margins, growth figures, and percentages are decimal
    fractions (0.20, not 20 or "20%").
  - All multiples (entry/exit, leverage) are bare floats (5.5x -> 5.5).
  - Sign convention: cash inflows to the company are positive; interest
    expense, fees, and coverage-metric denominators are positive costs.

`ScalarOrSeries` fields accept either a single float (held flat across every
projection period) or an explicit list of one float per period. Expansion to
a full (n_periods,) array happens in `loader.expand_series`, once the number
of periods is known from `TimelineConfig`.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, model_validator

ScalarOrSeries = float | list[float]


class TimelineConfig(BaseModel):
    n_periods: int = Field(gt=0)
    n_historical: int = Field(default=0, ge=0)
    start_year: int | None = None

    @model_validator(mode="after")
    def _check_historical_bound(self) -> TimelineConfig:
        if self.n_historical > self.n_periods:
            raise ValueError("n_historical cannot exceed n_periods")
        return self


class CompanyAssumptions(BaseModel):
    revenue_base_mm: float = Field(gt=0)
    revenue_growth: ScalarOrSeries
    ebitda_margin: ScalarOrSeries
    da_pct_revenue: ScalarOrSeries
    capex_pct_revenue: ScalarOrSeries
    nwc_pct_revenue: ScalarOrSeries
    tax_rate: float = Field(ge=0, lt=1)
    nol_beginning_balance_mm: float = Field(default=0.0, ge=0)
    dividend_pct_of_ni: float = Field(default=0.0, ge=0, le=1)


class OpeningBalanceSheet(BaseModel):
    """The period -1 balance sheet the corporate model rolls forward from.

    Whether this must include debt depends on which model is run against it:
    `run_corporate_model_no_debt` always treats total_debt as zero, so these
    figures should describe a debt-free balance sheet in that case. Used with
    `run_corporate_model_with_debt`, the tranches are already funded at their
    face value as of period 0's start, so equity_mm must be sized to also
    cover that initial debt (assets = other_liabilities + equity + initial
    debt), the way the transaction layer's Sources & Uses will size it
    automatically once that module exists. Pydantic can't see the tranche
    list from here to enforce either invariant; `checks.check_balance_sheet_balances`
    catches a mismatch at runtime instead, for whichever model actually ran."""

    cash_mm: float = Field(ge=0)
    nwc_mm: float
    ppe_mm: float = Field(ge=0)
    goodwill_mm: float = Field(default=0.0, ge=0)
    deferred_financing_costs_mm: float = Field(default=0.0, ge=0)
    other_liabilities_mm: float = Field(default=0.0, ge=0)
    equity_mm: float


class TrancheType(StrEnum):
    REVOLVER = "revolver"
    TERM_LOAN_A = "term_loan_a"
    TERM_LOAN_B = "term_loan_b"
    SENIOR_NOTES = "senior_notes"
    SUBORDINATED_PIK = "subordinated_pik"


DEFAULT_SECURED_TRANCHE_TYPES = frozenset(
    {TrancheType.REVOLVER, TrancheType.TERM_LOAN_A, TrancheType.TERM_LOAN_B}
)


class RateType(StrEnum):
    FIXED = "fixed"
    FLOATING = "floating"


class TrancheConfig(BaseModel):
    name: str
    tranche_type: TrancheType
    size_mm: float = Field(ge=0)
    rate_type: RateType
    fixed_rate: float | None = Field(default=None, ge=0)
    spread: float | None = Field(default=None, ge=0)
    rate_floor: float | None = Field(default=None, ge=0)
    mandatory_amort_pct_of_original: ScalarOrSeries = 0.0
    cash_sweep_eligible: bool = False
    sweep_priority: int | None = None
    pik_fraction: float = Field(default=0.0, ge=0.0, le=1.0)
    upfront_fee_pct: float = Field(default=0.0, ge=0)
    oid_pct: float = Field(default=0.0, ge=0)
    fee_amortization_years: int = Field(default=1, gt=0)
    commitment_fee_pct: float | None = Field(default=None, ge=0)
    secured: bool | None = Field(
        default=None,
        description=(
            "None (default) uses the type-based default: revolver/term_loan_a/"
            "term_loan_b are secured, senior_notes/subordinated_pik are not. "
            "Set explicitly to override -- e.g. unsecured TLB or secured notes."
        ),
    )

    @property
    def is_revolver(self) -> bool:
        return self.tranche_type is TrancheType.REVOLVER

    @property
    def is_secured(self) -> bool:
        if self.secured is not None:
            return self.secured
        return self.tranche_type in DEFAULT_SECURED_TRANCHE_TYPES

    @model_validator(mode="after")
    def _check_rate_fields(self) -> TrancheConfig:
        if self.rate_type is RateType.FIXED and self.fixed_rate is None:
            raise ValueError(f"tranche '{self.name}': fixed_rate required for rate_type=fixed")
        if self.rate_type is RateType.FLOATING and self.spread is None:
            raise ValueError(f"tranche '{self.name}': spread required for rate_type=floating")
        if self.cash_sweep_eligible and self.sweep_priority is None:
            raise ValueError(
                f"tranche '{self.name}': sweep_priority required when cash_sweep_eligible=True"
            )
        if self.commitment_fee_pct is not None and not self.is_revolver:
            raise ValueError(
                f"tranche '{self.name}': commitment_fee_pct only applies to revolver tranches"
            )
        if self.is_revolver and self.pik_fraction != 0.0:
            raise ValueError(f"tranche '{self.name}': revolver cannot have a PIK component")
        return self


class CovenantMetric(StrEnum):
    TOTAL_NET_LEVERAGE = "total_net_leverage"
    SECURED_NET_LEVERAGE = "secured_net_leverage"
    INTEREST_COVERAGE = "interest_coverage"
    FCCR = "fccr"


class CovenantConfig(BaseModel):
    name: str
    metric: CovenantMetric
    threshold: ScalarOrSeries
    test_from_period: int = Field(default=0, ge=0)
    springing_revolver_draw_pct: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description=(
            "When set, the covenant is only tested in a period where the revolver's "
            "drawn balance / commitment strictly exceeds this fraction -- the usual "
            "covenant-lite structure, where a maintenance test applies only to the "
            "revolver and only once it's meaningfully drawn. None (default) means the "
            "covenant is tested in every period from test_from_period onward, matching "
            "prior behavior exactly."
        ),
    )


class TransactionConfig(BaseModel):
    entry_multiple: float = Field(gt=0)
    transaction_fees_pct: float = Field(default=0.0, ge=0)
    exit_year_index: int = Field(ge=0)


class InterestMode(StrEnum):
    AVERAGE_BALANCE = "average_balance"
    BEGINNING_BALANCE = "beginning_balance"


class WaterfallConfig(BaseModel):
    minimum_cash_mm: float = Field(ge=0)
    sweep_pct: float = Field(default=1.0, ge=0.0, le=1.0)
    interest_mode: InterestMode = InterestMode.AVERAGE_BALANCE
    circularity_tolerance: float = Field(default=1e-6, gt=0)
    circularity_max_iterations: int = Field(default=50, gt=0)


class DriverVolConfig(BaseModel):
    revenue_growth_std: float = Field(default=0.0, ge=0)
    ebitda_margin_std: float = Field(default=0.0, ge=0)
    capex_pct_revenue_std: float = Field(default=0.0, ge=0)
    nwc_pct_revenue_std: float = Field(default=0.0, ge=0)
    exit_multiple_std: float = Field(default=0.0, ge=0)
    base_rate_std: float = Field(default=0.0, ge=0)


DRIVER_ORDER = (
    "revenue_growth",
    "ebitda_margin",
    "capex_pct_revenue",
    "nwc_pct_revenue",
    "exit_multiple",
    "base_rate",
)


class PersistenceConfig(BaseModel):
    """AR(1) autocorrelation on the per-period shock feeding revenue_growth
    and ebitda_margin: shock_t = phi*shock_{t-1} + innovation_t. phi=0
    reproduces the i.i.d.-shock (simple mode) behavior for that driver
    exactly -- illustrative defaults, not calibrated to any real dataset."""

    revenue_growth_phi: float = Field(default=0.0, ge=0.0, lt=1.0)
    ebitda_margin_phi: float = Field(default=0.0, ge=0.0, lt=1.0)


class RecessionRegimeConfig(BaseModel):
    """Stochastic recession clustering: each period, while a scenario isn't
    already in a recession, a new one starts with `annual_probability`;
    once started it runs for a fixed `duration_years`, applying a constant
    hit to revenue_growth and ebitda_margin each period of the recession
    (no overlapping recessions). Illustrative defaults, not calibrated."""

    annual_probability: float = Field(default=0.0, ge=0.0, le=1.0)
    duration_years: int = Field(default=2, ge=1)
    revenue_growth_hit: float = Field(
        default=-0.10, description="Additive hit per recession period, e.g. -0.10 = 10pp lower."
    )
    ebitda_margin_hit: float = Field(
        default=-0.03, description="Additive hit per recession period, e.g. -0.03 = 300bps lower."
    )


class FatTailsConfig(BaseModel):
    """Student-t innovations (variance-normalized so driver_vol's std fields
    keep their meaning) instead of normal, for every stochastic driver."""

    degrees_of_freedom: float = Field(
        default=5.0, gt=2.0, description="Must be > 2 for finite variance; lower = fatter tails."
    )


class RateMeanReversionConfig(BaseModel):
    """Replaces base_rate's flat-deterministic-path-plus-noise default with
    a discrete AR(1)/Ornstein-Uhlenbeck path: rate_t = rate_{t-1} +
    kappa*(long_run_mean - rate_{t-1}) + shock_t, shock_t drawn at
    driver_vol.base_rate_std same as before. Period 0 is unaffected
    (anchored to the deterministic entry rate + its own shock, matching
    simple mode), reversion starts from period 1."""

    kappa: float = Field(
        default=0.3, ge=0.0, le=1.0, description="Speed of reversion per period; 0=random walk."
    )
    long_run_mean: float = Field(gt=0.0)


class ExitMultipleLinkConfig(BaseModel):
    """Structural exit-multiple sensitivity, replacing the pure-noise
    default: exit_multiple = scenario.exit_multiple + beta_growth*(EBITDA
    CAGR from entry to exit - reference_cagr) + beta_rate*(exit-year
    base_rate - reference_rate) + independent noise (still
    driver_vol.exit_multiple_std, drawn fresh rather than reused from the
    correlated shock cube, since the structural terms already capture the
    fundamentals link). beta_rate is typically negative (higher rates
    compress multiples); beta_growth typically positive."""

    beta_growth: float = Field(default=0.0)
    reference_cagr: float = Field(default=0.0)
    beta_rate: float = Field(default=0.0)
    reference_rate: float = Field(default=0.0)


class AdvancedScenarioConfig(BaseModel):
    """Opt-in richer scenario generation for the Monte Carlo engine. Every
    field defaults to None/off; `scenario.advanced` itself defaults to None,
    so `generate_stochastic_drivers` takes the exact same code path as
    before this existed whenever it's absent -- corefin run/optimize are
    unaffected unless a config explicitly adds this block."""

    persistence: PersistenceConfig | None = None
    regime: RecessionRegimeConfig | None = None
    fat_tails: FatTailsConfig | None = None
    rate_mean_reversion: RateMeanReversionConfig | None = None
    exit_multiple_link: ExitMultipleLinkConfig | None = None


class ScenarioConfig(BaseModel):
    n_scenarios: int = Field(default=1, gt=0)
    random_seed: int | None = None
    exit_multiple: float = Field(gt=0)
    base_rate: ScalarOrSeries
    driver_vol: DriverVolConfig = DriverVolConfig()
    driver_correlation: list[list[float]] | None = None
    advanced: AdvancedScenarioConfig | None = None

    @model_validator(mode="after")
    def _check_correlation_shape(self) -> ScenarioConfig:
        if self.driver_correlation is not None:
            n = len(DRIVER_ORDER)
            rows = self.driver_correlation
            if len(rows) != n or any(len(r) != n for r in rows):
                raise ValueError(
                    f"driver_correlation must be a {n}x{n} matrix ordered {DRIVER_ORDER}"
                )
        return self


class DecisionVariableConfig(BaseModel):
    """One tranche's size, expressed as a multiple of entry EBITDA, that the
    optimizer is free to vary. Revolver and PIK tranches can be decision
    variables too (e.g. min_multiple=0.0 lets a PIK tranche be switched off);
    nothing here is specific to term tranches."""

    tranche_name: str
    min_multiple: float = Field(ge=0)
    max_multiple: float = Field(gt=0)
    step_multiple: float | None = Field(
        default=None,
        gt=0,
        description=(
            "Grid step, in turns of EBITDA. If omitted, the step is derived from "
            "search.grid_points_per_dimension instead of a hardcoded default."
        ),
    )

    @model_validator(mode="after")
    def _check_bounds(self) -> DecisionVariableConfig:
        if self.max_multiple <= self.min_multiple:
            raise ValueError(
                f"decision variable '{self.tranche_name}': max_multiple must exceed min_multiple"
            )
        return self


class LeverageBasis(StrEnum):
    TOTAL = "total_leverage"
    SECURED = "secured_leverage"


class TranchePricingConfig(BaseModel):
    """Linear leverage-based pricing ramp for one tranche: above
    leverage_threshold, spread/fees step up by a fixed amount per turn of
    leverage (closing leverage, gross of cash -- a static, structure-only
    figure distinct from the period-by-period *net* leverage covenants use)."""

    tranche_name: str
    basis: LeverageBasis = LeverageBasis.TOTAL
    leverage_threshold: float = Field(ge=0)
    spread_bps_per_turn: float = Field(default=0.0, ge=0)
    upfront_fee_pct_per_turn: float = Field(default=0.0, ge=0)
    oid_pct_per_turn: float = Field(default=0.0, ge=0)
    market_capacity_mm: float | None = Field(default=None, ge=0)


class PricingGridConfig(BaseModel):
    tranches: list[TranchePricingConfig] = []
    total_market_capacity_mm: float | None = Field(default=None, ge=0)


class DeterministicConstraintsConfig(BaseModel):
    """At-close constraints, each optional (None = unconstrained).

    extra="forbid": this model previously had a `max_senior_leverage` field,
    renamed to `max_secured_leverage`. A leftover old key should raise a
    clear error rather than silently being ignored (pydantic's default
    behavior for unrecognized fields)."""

    model_config = ConfigDict(extra="forbid")

    max_total_leverage: float | None = Field(default=None, gt=0)
    max_secured_leverage: float | None = Field(default=None, gt=0)
    min_equity_pct_of_sources: float | None = Field(default=None, ge=0, le=1)
    min_interest_coverage_at_close: float | None = Field(default=None, gt=0)


class StochasticConstraintsConfig(BaseModel):
    """Monte Carlo constraints, each optional (None = unconstrained).
    max_covenant_breach_probability is one aggregate probability across every
    configured covenant (P(any tested period of any covenant breaches)), not
    a separate limit per covenant."""

    max_covenant_breach_probability: float | None = Field(default=None, ge=0, le=1)
    max_revolver_shortfall_probability: float | None = Field(default=None, ge=0, le=1)
    max_loss_of_capital_probability: float | None = Field(default=None, ge=0, le=1)


class ObjectiveKind(StrEnum):
    MEAN_IRR = "mean_irr"
    MEDIAN_IRR = "median_irr"
    PERCENTILE_IRR = "percentile_irr"
    MEAN_DOWNSIDE_BLEND = "mean_downside_blend"


class ObjectiveConfig(BaseModel):
    kind: ObjectiveKind = ObjectiveKind.MEAN_IRR
    percentile: float = Field(
        default=10.0,
        ge=0,
        le=100,
        description="Used by percentile_irr, and as the downside p in the blend.",
    )
    downside_lambda: float = Field(
        default=0.5,
        ge=0,
        description="mean_downside_blend: mean_irr - downside_lambda*(mean_irr - percentile_irr).",
    )


class SearchConfig(BaseModel):
    n_scenarios_search: int = Field(default=2000, gt=0)
    n_scenarios_confirm: int = Field(default=10000, gt=0)
    random_seed: int = 42
    grid_points_per_dimension: int = Field(default=15, gt=1)
    refine: bool = True
    refine_grid_points_per_dimension: int = Field(default=9, gt=1)
    max_confirmation_fallback_attempts: int = Field(
        default=5,
        ge=0,
        description=(
            "If the recommended structure fails its constraints on the larger "
            "confirmation sample, try up to this many of the next-best feasible "
            "(on the search sample) candidates, in objective order, until one "
            "also passes confirmation."
        ),
    )


class ToleranceConfig(BaseModel):
    """How close to a limit counts as 'binding' for reporting purposes.
    Probability-type constraints (breach/shortfall/loss-of-capital) use an
    absolute tolerance in percentage points, since a 2pp gap means something
    very different at a 5% limit than at a 50% one; leverage/coverage/equity%
    constraints and decision-variable bounds use a relative tolerance."""

    probability_tolerance_pp: float = Field(default=2.0, ge=0)
    relative_tolerance_pct: float = Field(default=5.0, ge=0)


class RelaxationConfig(BaseModel):
    """How much to loosen a binding constraint or bound when computing the
    relaxation sensitivity ("shadow price") table."""

    probability_relax_pp: float = Field(default=2.0, ge=0)
    leverage_relax_turns: float = Field(default=0.5, ge=0)
    equity_pct_relax_pp: float = Field(default=5.0, ge=0)
    coverage_relax_turns: float = Field(default=0.25, ge=0)


class OptimizerConfig(BaseModel):
    decision_variables: list[DecisionVariableConfig]
    pricing: PricingGridConfig = PricingGridConfig()
    deterministic_constraints: DeterministicConstraintsConfig = DeterministicConstraintsConfig()
    stochastic_constraints: StochasticConstraintsConfig = StochasticConstraintsConfig()
    objective: ObjectiveConfig = ObjectiveConfig()
    search: SearchConfig = SearchConfig()
    tolerances: ToleranceConfig = ToleranceConfig()
    relaxation: RelaxationConfig = RelaxationConfig()

    @model_validator(mode="after")
    def _check_decision_variables(self) -> OptimizerConfig:
        if not self.decision_variables:
            raise ValueError("optimizer.decision_variables must have at least one entry")
        names = [d.tranche_name for d in self.decision_variables]
        if len(names) != len(set(names)):
            raise ValueError("optimizer.decision_variables tranche_name values must be unique")
        return self


class RecessionStressConfig(BaseModel):
    """A one-time growth/margin shock at `start_year_index`, tapering
    linearly back to zero over `recovery_years` (e.g. start_year_index=2,
    recovery_years=2: full hit at year 2, half at year 3, fully recovered
    by year 4). Illustrative defaults, not calibrated to any real
    downturn."""

    start_year_index: int = Field(ge=0)
    revenue_growth_hit: float = Field(
        default=-0.10, description="Additive hit at the peak (start) year, e.g. -0.10 = 10pp."
    )
    ebitda_margin_hit: float = Field(
        default=-0.03, description="Additive hit at the peak (start) year, e.g. -0.03 = 300bps."
    )
    recovery_years: int = Field(default=2, ge=0, description="0 = one-period shock with no taper.")


class RateShockStressConfig(BaseModel):
    """A permanent, held base_rate shift from `start_year_index` onward."""

    start_year_index: int = Field(default=0, ge=0)
    bps: float = Field(default=0.03, description="Additive, e.g. 0.03 = +300bps.")


class MultipleCompressionStressConfig(BaseModel):
    """A shift applied to the exit multiple only."""

    delta: float = Field(default=-2.0, description="Additive, e.g. -2.0 = 2.0x lower.")


class StressShockConfig(BaseModel):
    """One or more shocks combined into a named stress scenario; any
    combination of the three is allowed (e.g. "combined downside" sets all
    three at once)."""

    recession: RecessionStressConfig | None = None
    rate_shock: RateShockStressConfig | None = None
    multiple_compression: MultipleCompressionStressConfig | None = None


class NamedStressScenarioConfig(BaseModel):
    name: str
    shocks: StressShockConfig


class SimulateConfig(BaseModel):
    """Settings for the Sponsor LBO Monte Carlo engine (`corefin simulate`)."""

    stress_scenarios: list[NamedStressScenarioConfig] = Field(default_factory=list)
    irr_hurdle: float = Field(
        default=0.15, description="Illustrative -- P(IRR below this) is reported by the engine."
    )


class RootConfig(BaseModel):
    timeline: TimelineConfig
    company: CompanyAssumptions
    opening_balance_sheet: OpeningBalanceSheet
    transaction: TransactionConfig
    tranches: list[TrancheConfig]
    covenants: list[CovenantConfig] = []
    scenario: ScenarioConfig
    waterfall: WaterfallConfig
    optimizer: OptimizerConfig | None = None
    simulate: SimulateConfig | None = None

    @model_validator(mode="after")
    def _check_tranches(self) -> RootConfig:
        if not self.tranches:
            raise ValueError("at least one debt tranche is required")
        n_revolvers = sum(1 for t in self.tranches if t.tranche_type is TrancheType.REVOLVER)
        if n_revolvers != 1:
            raise ValueError(
                f"exactly one revolver tranche is required (found {n_revolvers}); the cash "
                "waterfall always draws on/pays down a revolver, so include one with size_mm=0 "
                "if the deal has no revolving facility"
            )
        priorities = [t.sweep_priority for t in self.tranches if t.cash_sweep_eligible]
        if len(priorities) != len(set(priorities)):
            raise ValueError("sweep_priority values must be unique among sweep-eligible tranches")
        if self.transaction.exit_year_index >= self.timeline.n_periods:
            raise ValueError("transaction.exit_year_index must be within the timeline")
        if self.optimizer is not None:
            tranche_names = {t.name for t in self.tranches}
            for dv in self.optimizer.decision_variables:
                if dv.tranche_name not in tranche_names:
                    raise ValueError(
                        f"optimizer decision variable references unknown tranche "
                        f"'{dv.tranche_name}'; known tranches: {sorted(tranche_names)}"
                    )
            for tp in self.optimizer.pricing.tranches:
                if tp.tranche_name not in tranche_names:
                    raise ValueError(
                        f"optimizer pricing entry references unknown tranche "
                        f"'{tp.tranche_name}'; known tranches: {sorted(tranche_names)}"
                    )
        return self
