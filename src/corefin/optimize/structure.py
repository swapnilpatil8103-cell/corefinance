"""Structure builder: turn a candidate's decision-variable values into a
complete, internally consistent model input (sized tranches, sources & uses,
opening balance sheet). Pricing (leverage-dependent spread/fee adjustments,
market capacity limits) is applied separately in `pricing.py`, since sizing
and closing leverage don't depend on it -- pricing depends on leverage, not
the other way around.
"""

from __future__ import annotations

from dataclasses import dataclass

from corefin.assumptions.loader import expand_series
from corefin.assumptions.schema import OpeningBalanceSheet, RootConfig, TrancheConfig
from corefin.transaction.sources_uses import SourcesAndUses, compute_sources_and_uses


def compute_entry_ebitda_margin(root_config: RootConfig) -> float:
    """The deterministic base-case margin at period 0 -- deal sizing is fixed
    at close and doesn't vary by Monte Carlo scenario."""
    return expand_series(
        root_config.company.ebitda_margin, root_config.timeline.n_periods, "ebitda_margin"
    )[0]


def compute_entry_ebitda_mm(root_config: RootConfig) -> float:
    return root_config.company.revenue_base_mm * compute_entry_ebitda_margin(root_config)


def size_tranches(
    tranches: list[TrancheConfig], decision_values: dict[str, float], entry_ebitda_mm: float
) -> list[TrancheConfig]:
    """Apply each decision variable's chosen multiple (of entry EBITDA) as
    that tranche's new size_mm; every other tranche keeps its configured
    size unchanged."""
    sized = []
    for t in tranches:
        if t.name in decision_values:
            new_size_mm = decision_values[t.name] * entry_ebitda_mm
            sized.append(t.model_copy(update={"size_mm": new_size_mm}))
        else:
            sized.append(t)
    return sized


@dataclass(frozen=True)
class ClosingLeverage:
    total_debt_sources_mm: float
    total_leverage: float
    secured_debt_sources_mm: float
    secured_leverage: float


def compute_closing_leverage(
    tranches: list[TrancheConfig], entry_ebitda_mm: float
) -> ClosingLeverage:
    """Gross debt at close / entry EBITDA -- a static, structure-only figure,
    distinct from the period-by-period *net* leverage the covenant system
    tracks. The revolver is excluded: undrawn at close by convention."""
    total_debt_sources_mm = sum(t.size_mm for t in tranches if not t.is_revolver)
    secured_debt_sources_mm = sum(t.size_mm for t in tranches if not t.is_revolver and t.is_secured)
    return ClosingLeverage(
        total_debt_sources_mm=total_debt_sources_mm,
        total_leverage=total_debt_sources_mm / entry_ebitda_mm,
        secured_debt_sources_mm=secured_debt_sources_mm,
        secured_leverage=secured_debt_sources_mm / entry_ebitda_mm,
    )


def build_opening_balance_sheet(
    root_config: RootConfig, sources_and_uses: SourcesAndUses
) -> OpeningBalanceSheet:
    """Rebuilds cash, deferred financing costs, goodwill and equity from
    sources & uses. NWC, PP&E and other liabilities are held fixed -- company
    facts independent of financing structure -- taken from the config's own
    opening_balance_sheet. Goodwill is the plug that makes the balance sheet
    balance given the other five figures (and, since equity + total debt
    sources always sums to total uses regardless of the debt/equity split,
    goodwill algebraically reduces to a constant across every candidate:
    other_liabilities + purchase_price + transaction_fees - cash - nwc - ppe
    -- the full derivation below is kept explicit rather than using that
    shortcut, since it's what "rebuilt from sources & uses" literally means)."""
    base = root_config.opening_balance_sheet
    cash_mm = root_config.waterfall.minimum_cash_mm
    deferred_financing_costs_mm = sources_and_uses.financing_fees_and_oid_mm
    equity_mm = sources_and_uses.sponsor_equity_mm
    goodwill_mm = (
        (base.other_liabilities_mm + equity_mm + sources_and_uses.total_debt_sources_mm)
        - cash_mm
        - base.nwc_mm
        - base.ppe_mm
        - deferred_financing_costs_mm
    )
    return OpeningBalanceSheet(
        cash_mm=cash_mm,
        nwc_mm=base.nwc_mm,
        ppe_mm=base.ppe_mm,
        goodwill_mm=goodwill_mm,
        deferred_financing_costs_mm=deferred_financing_costs_mm,
        other_liabilities_mm=base.other_liabilities_mm,
        equity_mm=equity_mm,
    )


@dataclass(frozen=True)
class CandidateStructure:
    decision_values: dict[str, float]
    tranches: list[TrancheConfig]
    entry_ebitda_mm: float
    leverage: ClosingLeverage
    sources_and_uses: SourcesAndUses
    opening_balance_sheet: OpeningBalanceSheet


def build_structure(
    root_config: RootConfig, decision_values: dict[str, float]
) -> CandidateStructure:
    if root_config.optimizer is None:
        raise ValueError("root_config.optimizer must be set to build a candidate structure")

    entry_margin = compute_entry_ebitda_margin(root_config)
    entry_ebitda_mm = root_config.company.revenue_base_mm * entry_margin

    tranches = size_tranches(root_config.tranches, decision_values, entry_ebitda_mm)
    leverage = compute_closing_leverage(tranches, entry_ebitda_mm)

    sources_and_uses = compute_sources_and_uses(
        root_config.transaction, tranches, root_config.company.revenue_base_mm, entry_margin
    )
    opening_balance_sheet = build_opening_balance_sheet(root_config, sources_and_uses)

    return CandidateStructure(
        decision_values=dict(decision_values),
        tranches=tranches,
        entry_ebitda_mm=entry_ebitda_mm,
        leverage=leverage,
        sources_and_uses=sources_and_uses,
        opening_balance_sheet=opening_balance_sheet,
    )
