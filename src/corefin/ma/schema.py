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

from pydantic import BaseModel, ConfigDict, Field, model_validator


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


class DistributableCashMethod(StrEnum):
    """What counts as cash available to the acquirer's common shareholders
    each post-close quarter, for `accretion.compute_acquirer_irr`'s
    interim cash flows -- NOT the full net income available to common
    (the original Stage 4 bug: counting ALL net income as an interim
    flow AND the resulting retained tangible equity as a terminal value
    double-counts the retained portion)."""

    DIVIDENDS = "dividends"  # actual common dividends paid (BankConfig.dividend_payout_ratio)
    EXCESS_CAPITAL_ABOVE_TARGET_CET1 = "excess_capital_above_target_cet1"  # see
    # accretion._distributable_cash_above_target_mm


class ExitMultipleBasis(StrEnum):
    """What `IrrConfig.exit_multiple` is applied to, to produce
    `compute_acquirer_irr`'s terminal value (added to the LAST interim
    cash flow) -- the value of whatever wasn't already paid out as
    distributable cash during the horizon."""

    FORWARD_PE = "forward_pe"  # exit_multiple x the deal's INCREMENTAL net income available to
    # common over the horizon's final 4 quarters (annualized "forward earnings")
    PRICE_TO_TBV = "price_to_tbv"  # exit_multiple x the deal's INCREMENTAL tangible common
    # equity at the end of the horizon


class IrrConfig(BaseModel):
    """`accretion.compute_acquirer_irr`'s interim-cash-flow and terminal-
    value assumptions. `exit_multiple` defaults to 1.0x `PRICE_TO_TBV` --
    a NEUTRAL default (no assumed multiple expansion/contraction versus
    book value), matching this module's original (pre-fix) implicit
    behavior as a default, not a realistic market assumption: override
    with an actual market multiple for a real IRR estimate."""

    model_config = ConfigDict(extra="forbid")

    distributable_cash_method: DistributableCashMethod = DistributableCashMethod.DIVIDENDS
    target_cet1_ratio: float | None = Field(
        default=None,
        gt=0.0,
        lt=1.0,
        description="Required when distributable_cash_method is "
        "excess_capital_above_target_cet1; the CET1 ratio the acquirer/pro forma entity "
        "retains earnings to maintain, distributing everything generated above it. RWA is "
        "held flat at its close-date level for this calculation -- this project doesn't "
        "project RWA growth beyond close (see corefin.ma.capital's own module docstring).",
    )
    exit_multiple_basis: ExitMultipleBasis = ExitMultipleBasis.PRICE_TO_TBV
    exit_multiple: float = Field(default=1.0, gt=0.0)

    @model_validator(mode="after")
    def _require_target_cet1_ratio_for_excess_capital(self) -> IrrConfig:
        if (
            self.distributable_cash_method
            == DistributableCashMethod.EXCESS_CAPITAL_ABOVE_TARGET_CET1
            and self.target_cet1_ratio is None
        ):
            raise ValueError(
                "target_cet1_ratio is required when distributable_cash_method is "
                "excess_capital_above_target_cet1"
            )
        return self


class DealConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    consideration: ConsiderationConfig
    credit_mark: CreditMarkConfig
    rate_mark: RateMarkConfig = Field(default_factory=RateMarkConfig)
    securities_mark: SecuritiesMarkConfig = Field(default_factory=SecuritiesMarkConfig)
    cdi: CdiConfig = Field(default_factory=CdiConfig)
    cost_saves: CostSaveConfig = Field(default_factory=CostSaveConfig)
    irr: IrrConfig = Field(default_factory=IrrConfig)
    tax_rate: float = Field(default=0.25, ge=0.0, lt=1.0)
    deal_horizon_quarters: int = Field(
        default=20,
        gt=0,
        description="Stage 4 deal economics (EPS accretion, TBV earnback, IRR) run over their "
        "OWN horizon -- default 5 years (20 post-close quarters) -- separate from whatever "
        "shorter or longer horizon a credit-engine scenario natively covers. See "
        "corefin.ma.horizon.build_deal_horizon_bank_result, which extends each bank's "
        "CreditLossProjection to this length before corefin.bank.model.run_bank_model builds "
        "the BankModelResult that run_deal_model consumes.",
    )
