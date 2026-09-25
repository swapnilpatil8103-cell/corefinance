"""Leverage-dependent pricing and market capacity limits.

Pricing is a linear ramp: above `leverage_threshold` turns of (total or
senior) closing leverage, spread/fixed rate and fees step up by a fixed
amount per turn. Applied after sizing (`structure.py`), since pricing
depends on the resulting leverage but sizing/leverage don't depend on
pricing -- there's no circularity to solve here, just a one-way dependency.
"""

from __future__ import annotations

from dataclasses import dataclass

from corefin.assumptions.schema import (
    LeverageBasis,
    PricingGridConfig,
    RateType,
    RootConfig,
    TrancheConfig,
    TranchePricingConfig,
)
from corefin.optimize.structure import (
    CandidateStructure,
    ClosingLeverage,
    build_opening_balance_sheet,
    build_structure,
    compute_entry_ebitda_margin,
)
from corefin.transaction.sources_uses import compute_sources_and_uses


def _basis_leverage(basis: LeverageBasis, leverage: ClosingLeverage) -> float:
    return leverage.total_leverage if basis is LeverageBasis.TOTAL else leverage.senior_leverage


def _priced_tranche(
    tranche: TrancheConfig, turns_above: float, pricing: TranchePricingConfig
) -> TrancheConfig:
    if turns_above <= 0:
        return tranche
    updates: dict[str, float] = {}
    if pricing.spread_bps_per_turn:
        bump = pricing.spread_bps_per_turn * turns_above
        if tranche.rate_type is RateType.FIXED:
            updates["fixed_rate"] = tranche.fixed_rate + bump
        else:
            updates["spread"] = tranche.spread + bump
    if pricing.upfront_fee_pct_per_turn:
        updates["upfront_fee_pct"] = (
            tranche.upfront_fee_pct + pricing.upfront_fee_pct_per_turn * turns_above
        )
    if pricing.oid_pct_per_turn:
        updates["oid_pct"] = tranche.oid_pct + pricing.oid_pct_per_turn * turns_above
    return tranche.model_copy(update=updates) if updates else tranche


def apply_pricing_grid(
    tranches: list[TrancheConfig], leverage: ClosingLeverage, pricing: PricingGridConfig
) -> list[TrancheConfig]:
    by_name = {tp.tranche_name: tp for tp in pricing.tranches}
    priced = []
    for t in tranches:
        tp = by_name.get(t.name)
        if tp is None:
            priced.append(t)
            continue
        basis_leverage = _basis_leverage(tp.basis, leverage)
        turns_above = max(0.0, basis_leverage - tp.leverage_threshold)
        priced.append(_priced_tranche(t, turns_above, tp))
    return priced


def build_priced_structure(
    root_config: RootConfig, decision_values: dict[str, float]
) -> CandidateStructure:
    """build_structure() plus the leverage-dependent pricing ramp. Sizing and
    leverage are computed once (pricing doesn't change size_mm, so they're
    identical either way); sources & uses and the opening balance sheet are
    then re-derived against the priced tranches, since financing fees/OID
    scale with the (now bumped) fee percentages."""
    unpriced = build_structure(root_config, decision_values)
    pricing = root_config.optimizer.pricing
    if not pricing.tranches:
        return unpriced

    priced_tranches = apply_pricing_grid(unpriced.tranches, unpriced.leverage, pricing)
    entry_margin = compute_entry_ebitda_margin(root_config)
    sources_and_uses = compute_sources_and_uses(
        root_config.transaction, priced_tranches, root_config.company.revenue_base_mm, entry_margin
    )
    opening_balance_sheet = build_opening_balance_sheet(root_config, sources_and_uses)

    return CandidateStructure(
        decision_values=unpriced.decision_values,
        tranches=priced_tranches,
        entry_ebitda_mm=unpriced.entry_ebitda_mm,
        leverage=unpriced.leverage,
        sources_and_uses=sources_and_uses,
        opening_balance_sheet=opening_balance_sheet,
    )


@dataclass(frozen=True)
class MarketCapacityCheck:
    feasible: bool
    violations: list[str]


def check_market_capacity(
    candidate: CandidateStructure, pricing: PricingGridConfig
) -> MarketCapacityCheck:
    """Debt simply isn't available above these limits -- a hard infeasibility,
    checked independently of pricing (a tranche can be capacity-infeasible
    even with no pricing ramp configured for it at all)."""
    violations: list[str] = []
    by_name = {tp.tranche_name: tp for tp in pricing.tranches}
    for t in candidate.tranches:
        tp = by_name.get(t.name)
        if (
            tp is not None
            and tp.market_capacity_mm is not None
            and t.size_mm > tp.market_capacity_mm
        ):
            violations.append(
                f"{t.name} size {t.size_mm:.1f}mm exceeds "
                f"market capacity {tp.market_capacity_mm:.1f}mm"
            )
    if (
        pricing.total_market_capacity_mm is not None
        and candidate.leverage.total_debt_sources_mm > pricing.total_market_capacity_mm
    ):
        violations.append(
            f"total debt {candidate.leverage.total_debt_sources_mm:.1f}mm exceeds total "
            f"market capacity {pricing.total_market_capacity_mm:.1f}mm"
        )
    return MarketCapacityCheck(feasible=not violations, violations=violations)
