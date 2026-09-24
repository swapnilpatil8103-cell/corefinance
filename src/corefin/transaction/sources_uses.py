"""Sources & Uses at entry.

Deal sizing (purchase price, tranche sizes, sponsor equity) is fixed at
close and does not vary by Monte Carlo scenario -- only the projected
operating performance and exit outcome do. `entry_ebitda_margin` should
therefore be the deterministic base-case margin (e.g. the config's period-0
value), not a stochastic per-scenario draw.
"""

from __future__ import annotations

from dataclasses import dataclass

from corefin.assumptions.schema import TrancheConfig, TransactionConfig


@dataclass(frozen=True)
class SourcesAndUses:
    entry_ebitda_mm: float
    purchase_price_mm: float
    transaction_fees_mm: float
    financing_fees_and_oid_mm: float
    total_uses_mm: float
    total_debt_sources_mm: float
    sponsor_equity_mm: float


def compute_sources_and_uses(
    transaction: TransactionConfig,
    tranches: list[TrancheConfig],
    revenue_base_mm: float,
    entry_ebitda_margin: float,
) -> SourcesAndUses:
    entry_ebitda_mm = revenue_base_mm * entry_ebitda_margin
    purchase_price_mm = transaction.entry_multiple * entry_ebitda_mm
    transaction_fees_mm = transaction.transaction_fees_pct * purchase_price_mm
    financing_fees_and_oid_mm = sum((t.upfront_fee_pct + t.oid_pct) * t.size_mm for t in tranches)
    total_uses_mm = purchase_price_mm + transaction_fees_mm + financing_fees_and_oid_mm
    total_debt_sources_mm = sum(t.size_mm for t in tranches if not t.is_revolver)
    sponsor_equity_mm = total_uses_mm - total_debt_sources_mm
    return SourcesAndUses(
        entry_ebitda_mm=entry_ebitda_mm,
        purchase_price_mm=purchase_price_mm,
        transaction_fees_mm=transaction_fees_mm,
        financing_fees_and_oid_mm=financing_fees_and_oid_mm,
        total_uses_mm=total_uses_mm,
        total_debt_sources_mm=total_debt_sources_mm,
        sponsor_equity_mm=sponsor_equity_mm,
    )
