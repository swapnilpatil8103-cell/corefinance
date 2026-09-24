"""Non-path-dependent per-tranche sizing: rates, scheduled amortization, fee
amortization. Path-dependent items (actual repayment given available cash,
PIK accrual, ending balances) live in `waterfall.py` / `circularity.py`.
"""

from __future__ import annotations

import numpy as np

from corefin.assumptions.loader import expand_series
from corefin.assumptions.schema import RateType, TrancheConfig


def effective_rate(tranche: TrancheConfig, base_rate: np.ndarray) -> np.ndarray:
    """All-in coupon rate for one tranche. base_rate: (n_scenarios, n_periods)."""
    if tranche.rate_type is RateType.FIXED:
        return np.full_like(base_rate, tranche.fixed_rate)
    floored_base = (
        base_rate if tranche.rate_floor is None else np.maximum(base_rate, tranche.rate_floor)
    )
    return floored_base + tranche.spread


def scheduled_mandatory_amortization(tranche: TrancheConfig, n_periods: int) -> np.ndarray:
    """$mm scheduled per period, as a fraction of ORIGINAL principal. Shape (n_periods,).
    The waterfall caps this at the outstanding balance, since balance can already have
    been reduced by cash sweeps in prior periods."""
    pct = expand_series(
        tranche.mandatory_amort_pct_of_original, n_periods, f"{tranche.name} mandatory_amort"
    )
    return pct * tranche.size_mm


def fee_amortization_schedule(tranche: TrancheConfig, n_periods: int) -> np.ndarray:
    """Straight-line amortization of upfront fee + OID over fee_amortization_years.
    Shape (n_periods,)."""
    total_fee = (tranche.upfront_fee_pct + tranche.oid_pct) * tranche.size_mm
    schedule = np.zeros(n_periods)
    if total_fee <= 0:
        return schedule
    annual_amount = total_fee / tranche.fee_amortization_years
    periods_to_amortize = min(tranche.fee_amortization_years, n_periods)
    schedule[:periods_to_amortize] = annual_amount
    return schedule
