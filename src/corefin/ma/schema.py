"""`DealConfig` -- every purchase-accounting assumption for one deal, all
in one pydantic model (loadable from YAML once Stage 7 wires up a CLI).
Units: dollar amounts are $mm; rates/percentages/shares are decimal
fractions (0.20, not 20); tax_rate defaults to 0.25 (combined federal +
state), matching `corefin.bank.schema.BankConfig`'s own default and the
same explicit user direction.

SIGN CONVENTIONS: `credit_mark_pct` is always a POSITIVE fraction of
gross loans (a write-DOWN -- the mark itself is subtracted in
`purchase_accounting.py`). `rate_mark_pct`/`securities_mark_pct` are
SIGNED (negative = a write-down, e.g. loans originated at below-market
rates in a rising-rate environment; positive = a write-up), since unlike
the credit mark, the fair-value interest-rate adjustment can go either
way depending on the rate environment at close."""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field


class Day2AllowanceMethod(StrEnum):
    CREDIT_MARK_RATE = "credit_mark_rate"  # Day-2 non-PCD allowance rate = credit_mark_pct
    TARGET_ACL_RATIO = "target_acl_ratio"  # Day-2 non-PCD allowance rate = target's own
    # existing allowance-to-loans ratio (pre-deal)


class ConsiderationConfig(BaseModel):
    """Deal price and funding mix. `price_to_tbv`: multiple of the
    target's tangible common book value (equity minus preferred stock
    minus goodwill minus other intangibles, all GAAP/book figures --
    `purchase_accounting.compute_target_tangible_common_equity`).
    `stock_pct`: fraction of total consideration paid in acquirer common
    stock (new shares issued); the remainder (`1 - stock_pct`) is cash,
    funded out of the acquirer's own balance sheet (no new acquisition
    debt modeled -- a documented Stage 3 simplification).

    SHARE DATA (Stage 4: EPS accretion/dilution, TBV per share, IRR) --
    explicit config inputs, never scraped market data, per the approved
    plan. `acquirer_share_price`: used only to convert the stock
    consideration's DOLLAR value into a number of new shares issued
    (`stock_consideration_mm / acquirer_share_price`); this model has no
    other use for a market price (it doesn't mark anything to market
    value). `target_shares_outstanding_mm` is optional and used only to
    report an illustrative exchange ratio -- not needed for EPS/TBV
    math, which only cares about the ACQUIRER's post-deal share count.
    `cash_funding_cost_rate`: the assumed forgone yield / cost of funds
    on cash consideration (cash paid out stops earning this rate) --
    Stage 3 assumes cash consideration is funded from the acquirer's own
    balance sheet, not new debt, so this is an OPPORTUNITY cost, not an
    interest expense on new borrowings; applied in Stage 4's pro forma
    income statement."""

    model_config = ConfigDict(extra="forbid")

    price_to_tbv: float = Field(gt=0.0)
    stock_pct: float = Field(ge=0.0, le=1.0)
    acquirer_share_price: float = Field(gt=0.0)
    acquirer_shares_outstanding_mm: float = Field(gt=0.0)
    target_shares_outstanding_mm: float | None = Field(default=None, gt=0.0)
    cash_funding_cost_rate: float = Field(default=0.04, ge=0.0, lt=1.0)

    @property
    def cash_pct(self) -> float:
        return 1.0 - self.stock_pct


class CreditMarkConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    credit_mark_pct: float = Field(ge=0.0, le=1.0)
    pcd_share: float = Field(
        ge=0.0,
        le=1.0,
        description="Fraction of the credit mark (and of "
        "target gross loans) attributed to purchased-credit-deteriorated (PCD) loans -- the "
        "rest is non-PCD.",
    )
    day2_allowance_method: Day2AllowanceMethod = Day2AllowanceMethod.CREDIT_MARK_RATE


class RateMarkConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    rate_mark_pct: float = Field(default=0.0, ge=-1.0, le=1.0)
    rate_mark_life_years: float = Field(default=5.0, gt=0.0)


class SecuritiesMarkConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    securities_mark_pct: float = Field(default=0.0, ge=-1.0, le=1.0)
    securities_mark_life_years: float = Field(default=3.0, gt=0.0)


class CdiConfig(BaseModel):
    """Core deposit intangible -- `cdi_pct_of_core_deposits` applied to
    the target's own deposits (`BankOpeningBalance.deposits_mm`; Stage 2
    doesn't break deposits into core vs. non-core/large-time, so all
    deposits are treated as the CDI base -- a documented simplification),
    amortized sum-of-years-digits (front-loaded, the standard convention
    for core deposit intangibles) over `cdi_amortization_years`."""

    model_config = ConfigDict(extra="forbid")

    cdi_pct_of_core_deposits: float = Field(default=0.02, ge=0.0, le=0.10)
    cdi_amortization_years: float = Field(default=10.0, gt=0.0)


class CostSaveConfig(BaseModel):
    """Ongoing cost saves don't affect the Day-0 pro forma balance sheet/
    capital (Stage 3's own deliverable) -- they matter for Stage 4's
    multi-year combined income statement, which consumes these same
    fields. `restructuring_charge_mm` DOES hit Day-0 (an after-tax
    reduction to pro forma retained earnings/CET1) and is this module's
    concern."""

    model_config = ConfigDict(extra="forbid")

    cost_save_pct_of_target_noninterest_expense: float = Field(default=0.0, ge=0.0, le=1.0)
    phase_in_year1_pct: float = Field(default=0.5, ge=0.0, le=1.0)
    restructuring_charge_mm: float = Field(default=0.0, ge=0.0)


class DealConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    consideration: ConsiderationConfig
    credit_mark: CreditMarkConfig
    rate_mark: RateMarkConfig = Field(default_factory=RateMarkConfig)
    securities_mark: SecuritiesMarkConfig = Field(default_factory=SecuritiesMarkConfig)
    cdi: CdiConfig = Field(default_factory=CdiConfig)
    cost_saves: CostSaveConfig = Field(default_factory=CostSaveConfig)
    tax_rate: float = Field(default=0.25, ge=0.0, lt=1.0)
