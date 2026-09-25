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

from pydantic import BaseModel, Field, model_validator

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

    @property
    def is_revolver(self) -> bool:
        return self.tranche_type is TrancheType.REVOLVER

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
    SENIOR_NET_LEVERAGE = "senior_net_leverage"
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


class ScenarioConfig(BaseModel):
    n_scenarios: int = Field(default=1, gt=0)
    random_seed: int | None = None
    exit_multiple: float = Field(gt=0)
    base_rate: ScalarOrSeries
    driver_vol: DriverVolConfig = DriverVolConfig()
    driver_correlation: list[list[float]] | None = None

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


class RootConfig(BaseModel):
    timeline: TimelineConfig
    company: CompanyAssumptions
    opening_balance_sheet: OpeningBalanceSheet
    transaction: TransactionConfig
    tranches: list[TrancheConfig]
    covenants: list[CovenantConfig] = []
    scenario: ScenarioConfig
    waterfall: WaterfallConfig

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
        return self
