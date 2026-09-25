"""Exit valuation and sponsor returns (IRR, MOIC), vectorized across scenarios.

Cash flows are assumed regular and annual: a single entry outflow at t=0,
any interim dividends to equity at each period end, and the exit equity
value (plus that period's dividend) at the exit period. IRR is solved with
a small vectorized Newton-Raphson routine rather than pulling in an extra
dependency for a two-or-few-cash-flow-point series.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _moic_based_guess(cash_flows: np.ndarray) -> np.ndarray:
    """A closed-form starting point: total inflows / total outflows, annualized
    over the cash flow's span. Exact (not just a starting guess) whenever the
    series has no interim flows, which is the common case here."""
    n_years = max(cash_flows.shape[1] - 1, 1)
    outflow = np.sum(np.where(cash_flows < 0, -cash_flows, 0.0), axis=1)
    inflow = np.sum(np.where(cash_flows > 0, cash_flows, 0.0), axis=1)
    safe_outflow = np.where(outflow <= 0, 1.0, outflow)
    moic = np.where(outflow <= 0, 1.0, inflow / safe_outflow)
    return moic ** (1.0 / n_years) - 1.0


def vectorized_irr(
    cash_flows: np.ndarray,
    guess: np.ndarray | None = None,
    tolerance: float = 1e-7,
    max_iterations: int = 100,
) -> np.ndarray:
    """cash_flows: (n_scenarios, n_points), evenly spaced annual cash flows at
    t=0..n_points-1. Returns the per-scenario rate r solving sum(cf_t/(1+r)^t)=0.
    Defaults to a MOIC-based starting guess (exact when there are no interim
    flows) rather than a fixed guess, since a fixed guess can diverge for
    scenarios with a strongly negative true IRR (e.g. equity near wiped out).

    A scenario with zero distributions of any kind after t=0 (a total loss,
    no interim dividends either) has NPV(r) = entry outflow for every r -- a
    constant, so the equation has no root at all, not just a very negative
    one. Newton's method has nothing to converge to there (the numerically
    ~0 derivative makes the solver take huge, unbounded steps); those
    scenarios are pinned to -99.9999% directly rather than iterated."""
    n_points = cash_flows.shape[1]
    t = np.arange(n_points)
    total_loss = np.sum(np.where(cash_flows > 0, cash_flows, 0.0), axis=1) <= 0.0
    if np.all(total_loss):
        return np.full(cash_flows.shape[0], -0.999999)
    rate = _moic_based_guess(cash_flows) if guess is None else np.asarray(guess, dtype=float)
    for _ in range(max_iterations):
        rate = np.maximum(rate, -0.999999)
        discount = (1.0 + rate[:, None]) ** t[None, :]
        npv = np.sum(cash_flows / discount, axis=1)
        d_discount = t[None, :] * (1.0 + rate[:, None]) ** (t[None, :] - 1)
        d_npv = np.sum(-cash_flows * d_discount, axis=1)
        d_npv = np.where(np.abs(d_npv) < 1e-12, 1e-12, d_npv)
        step = np.clip(npv / d_npv, -1.0, 1.0)
        rate = rate - step
        if np.max(np.abs(step[~total_loss])) < tolerance:
            break
    return np.where(total_loss, -0.999999, rate)


@dataclass(frozen=True)
class ExitResult:
    exit_ebitda: np.ndarray
    exit_enterprise_value: np.ndarray
    net_debt_at_exit: np.ndarray
    exit_equity_value: np.ndarray
    realized_exit_equity_value: np.ndarray
    equity_cash_flows: np.ndarray
    irr: np.ndarray
    moic: np.ndarray


def compute_exit_and_returns(
    sponsor_equity_mm: float,
    exit_year_index: int,
    exit_multiple: np.ndarray,
    ebitda: np.ndarray,
    total_debt: np.ndarray,
    cash: np.ndarray,
    dividends: np.ndarray,
) -> ExitResult:
    exit_ebitda = ebitda[:, exit_year_index]
    exit_enterprise_value = exit_multiple * exit_ebitda
    net_debt_at_exit = total_debt[:, exit_year_index] - cash[:, exit_year_index]
    # Can be negative when net debt exceeds enterprise value; kept as-is here for
    # diagnostics (how far underwater a scenario is). Sponsor equity has limited
    # liability, so what actually gets distributed is floored at zero below.
    exit_equity_value = exit_enterprise_value - net_debt_at_exit
    realized_exit_equity_value = np.maximum(exit_equity_value, 0.0)

    n_scenarios = ebitda.shape[0]
    n_flow_points = exit_year_index + 2
    equity_cash_flows = np.zeros((n_scenarios, n_flow_points))
    equity_cash_flows[:, 0] = -sponsor_equity_mm
    for t in range(exit_year_index + 1):
        equity_cash_flows[:, t + 1] += dividends[:, t]
    equity_cash_flows[:, exit_year_index + 1] += realized_exit_equity_value

    irr = vectorized_irr(equity_cash_flows)
    total_distributions = np.sum(equity_cash_flows[:, 1:], axis=1)
    moic = total_distributions / sponsor_equity_mm

    return ExitResult(
        exit_ebitda=exit_ebitda,
        exit_enterprise_value=exit_enterprise_value,
        net_debt_at_exit=net_debt_at_exit,
        exit_equity_value=exit_equity_value,
        realized_exit_equity_value=realized_exit_equity_value,
        equity_cash_flows=equity_cash_flows,
        irr=irr,
        moic=moic,
    )
