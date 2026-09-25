"""Returns attribution: per-scenario sponsor equity value-creation bridge.

Four components, summing EXACTLY to the change in equity value (realized
exit equity value + dividends received through exit - the sponsor's entry
equity check):

1. EBITDA growth -- entry multiple x (exit EBITDA - entry EBITDA).
2. Multiple change -- (exit multiple - entry multiple) x exit EBITDA.
   (1)+(2) together equal exactly (exit EV - entry EV); this is the
   "growth at entry multiple, then re-rating at exit EBITDA" convention --
   one of two standard orderings, both exact, this one chosen for
   consistency throughout this module.
3. Debt paydown / cash generation -- entry debt raised minus net debt at
   exit, plus dividends received through exit, plus a limited-liability
   floor adjustment (see below). Interest expense (cash and PIK) is not a
   separate bucket: it already reduced net income and therefore cash
   generation/debt paydown capacity, so its effect is already inside this
   bucket by construction, per the interest-attribution discussion for
   this stage.
4. Fees and other leakage -- transaction fees + financing fees/OID paid at
   entry (fixed across scenarios, since deal sizing doesn't vary by
   scenario).

Limited liability: `transaction.returns.compute_exit_and_returns` floors
the *realized* exit equity value at zero, but the unfloored value can be
negative (net debt exceeding enterprise value). To keep the four
components summing exactly to the *realized* change even for a wipeout
scenario, the floor adjustment (how much limited liability "rescued" the
sponsor from an even worse outcome) is folded into bucket 3 -- a
capital-structure/debt effect, the same bucket that already absorbs every
other cash/debt effect.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from corefin.assumptions.schema import RootConfig
from corefin.simulate.engine import SimulationResult


@dataclass(frozen=True)
class ValueCreationBridge:
    ebitda_growth: np.ndarray
    multiple_change: np.ndarray
    debt_paydown_and_cash: np.ndarray
    fees_and_leakage: np.ndarray
    total_value_creation: np.ndarray


def compute_value_creation_bridge(
    root_config: RootConfig, simulation_result: SimulationResult
) -> ValueCreationBridge:
    candidate = simulation_result.candidate
    exit_result = simulation_result.exit_result
    exit_idx = root_config.transaction.exit_year_index

    entry_multiple = root_config.transaction.entry_multiple
    entry_ebitda = candidate.entry_ebitda_mm
    exit_ebitda = exit_result.exit_ebitda
    exit_multiple = simulation_result.drivers.exit_multiple

    ebitda_growth = entry_multiple * (exit_ebitda - entry_ebitda)
    multiple_change = (exit_multiple - entry_multiple) * exit_ebitda

    sau = candidate.sources_and_uses
    dividends_cumulative = np.sum(
        simulation_result.model_result.cash_flow_statement.dividends[:, : exit_idx + 1], axis=1
    )
    floor_adjustment = np.maximum(0.0, -exit_result.exit_equity_value)
    debt_paydown_and_cash = (
        sau.total_debt_sources_mm
        - exit_result.net_debt_at_exit
        + dividends_cumulative
        + floor_adjustment
    )
    fees_and_leakage = np.full(
        exit_ebitda.shape, -(sau.transaction_fees_mm + sau.financing_fees_and_oid_mm)
    )

    total_value_creation = (
        exit_result.realized_exit_equity_value + dividends_cumulative - sau.sponsor_equity_mm
    )

    return ValueCreationBridge(
        ebitda_growth=ebitda_growth,
        multiple_change=multiple_change,
        debt_paydown_and_cash=debt_paydown_and_cash,
        fees_and_leakage=fees_and_leakage,
        total_value_creation=total_value_creation,
    )


@dataclass(frozen=True)
class BridgeSummary:
    label: str
    ebitda_growth: float
    multiple_change: float
    debt_paydown_and_cash: float
    fees_and_leakage: float
    total_value_creation: float


def _bridge_at_index(bridge: ValueCreationBridge, idx: int, label: str) -> BridgeSummary:
    return BridgeSummary(
        label=label,
        ebitda_growth=float(bridge.ebitda_growth[idx]),
        multiple_change=float(bridge.multiple_change[idx]),
        debt_paydown_and_cash=float(bridge.debt_paydown_and_cash[idx]),
        fees_and_leakage=float(bridge.fees_and_leakage[idx]),
        total_value_creation=float(bridge.total_value_creation[idx]),
    )


def _bridge_average(bridge: ValueCreationBridge) -> BridgeSummary:
    return BridgeSummary(
        label="Average",
        ebitda_growth=float(np.mean(bridge.ebitda_growth)),
        multiple_change=float(np.mean(bridge.multiple_change)),
        debt_paydown_and_cash=float(np.mean(bridge.debt_paydown_and_cash)),
        fees_and_leakage=float(np.mean(bridge.fees_and_leakage)),
        total_value_creation=float(np.mean(bridge.total_value_creation)),
    )


def summarize_value_creation_bridge(
    bridge: ValueCreationBridge, irr: np.ndarray
) -> list[BridgeSummary]:
    """Average bridge, plus the bridge for the specific scenarios whose own
    IRR is closest to the p10/median/p90 of the full distribution -- each
    of those three is a real scenario's exact bridge, not an interpolated
    or synthetic one."""
    summaries = [_bridge_average(bridge)]
    for label, percentile in (("P10", 10), ("Median", 50), ("P90", 90)):
        target = np.percentile(irr, percentile)
        idx = int(np.argmin(np.abs(irr - target)))
        summaries.append(_bridge_at_index(bridge, idx, label))
    return summaries
