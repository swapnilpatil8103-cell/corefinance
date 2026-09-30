"""Stage 5: the projection engine -- forecasts each loan category's NCO
rate forward under the Fed's baseline and severely-adverse scenarios,
using the model Stage 4's own backtest found best for that category, fit
on the FULL sample through 2025Q4 (the training window used everywhere
else in this project is 2001Q1-2025Q4; Stage 5 is the one place that
full-sample fit is actually used for something -- a genuine, unknown-
outcome projection, not a backtest against an already-known history).
Then computes a SIMPLIFIED CECL allowance and quarterly provision
roll-forward from that projected NCO path. No function here touches the
network or reads/writes files -- see cli.py for orchestration.

SIMPLIFIED CECL, AND WHAT "SIMPLIFIED" MEANS HERE: real CECL (ASC 326)
estimates the full LIFETIME expected credit loss for each loan pool,
typically using cohort/vintage loss curves built from each bank's own
loan-level maturity, prepayment and amortization data discounted to
present value, revisited every period. This project has none of that
(no loan-level data at all -- only bank-quarter aggregates). The
simplification: approximate "lifetime expected loss rate" as the average
projected ANNUALIZED NCO rate over a fixed, documented WEIGHTED-AVERAGE-
LIFE window per category (`CATEGORY_WEIGHTED_AVERAGE_LIFE_QUARTERS`) --
industry rule-of-thumb figures (not derived from this project's own
data, since there's no loan-level maturity data to derive them from),
clearly flagged as an ASSUMPTION a real implementation would replace with
a bank's own prepayment-adjusted duration estimates. The BALANCE used
throughout is held STATIC at each category's last actual (2025Q4)
level -- the same "static balance sheet" convention real regulatory
stress tests (DFAST/CCAR) use for exactly this reason: it isolates the
loss-rate projection from a separate, unmodeled balance-growth forecast.
"""

from __future__ import annotations

import pandas as pd

# Weighted-average-life, in QUARTERS, per category -- industry rule-of-
# thumb figures for the "lifetime" horizon the simplified CECL allowance
# averages projected losses over (see module docstring: NOT derived from
# this project's own data). Revolving/short-duration books (credit card,
# auto, other consumer, CRE construction) get short horizons; long-
# duration term/mortgage books (CRE multifamily/nonfarm-nonresidential,
# residential mortgage, home equity) get longer ones reflecting typical
# prepayment-adjusted effective life, not contractual maturity (e.g. a
# 30-year mortgage's WAL after prepayment/refinancing is conventionally
# much shorter than 30 years).
CATEGORY_WEIGHTED_AVERAGE_LIFE_QUARTERS: dict[str, int] = {
    "commercial_and_industrial": 14,  # ~3.5 years: revolvers/term loans
    "cre_construction": 10,  # ~2.5 years: short-term, converts or pays off
    "cre_multifamily": 32,  # ~8 years: term mortgage-style loans
    "cre_nonfarm_nonresidential": 32,  # ~8 years: term mortgage-style loans
    "residential_mortgage": 30,  # ~7.5 years: post-prepayment effective life
    "home_equity": 24,  # ~6 years: revolving/term mix
    "credit_card": 8,  # ~2 years: revolving, short expected life
    "auto": 11,  # ~2.75 years: fixed term, but prepays
    "other_consumer": 10,  # ~2.5 years: similar to auto/personal loans
}


def lifetime_expected_loss_rate(
    annualized_nco_rate_path: pd.Series, quarter: pd.Period, wal_quarters: int
) -> float:
    """The simplified "lifetime expected loss rate" as of `quarter`: the
    average of `annualized_nco_rate_path` over the `wal_quarters`
    quarters starting at `quarter` (inclusive). If the path doesn't reach
    that far forward (either because the projection horizon ends first,
    or because `quarter` is the historical jump-off itself and only past
    data exists), the LAST available quarter's rate is held constant to
    fill the remainder -- a simple, documented terminal-value convention,
    not a real amortization schedule. Raises if `quarter` itself isn't in
    `annualized_nco_rate_path` at all."""
    if quarter not in annualized_nco_rate_path.index:
        raise ValueError(f"{quarter} is not in the provided rate path")
    window_quarters = [quarter + i for i in range(wal_quarters)]
    available = annualized_nco_rate_path.reindex(window_quarters)
    if available.notna().any():
        available = available.ffill()
    return float(available.mean())


def backward_looking_lifetime_expected_loss_rate(
    realized_annualized_nco_rate: pd.Series, as_of_quarter: pd.Period, wal_quarters: int
) -> float:
    """The simplified "lifetime expected loss rate" as of a REALIZED
    (non-projected) quarter, e.g. the 2025Q4 jump-off -- the average of
    `realized_annualized_nco_rate` over the `wal_quarters` quarters
    ENDING at `as_of_quarter` (backward-looking, since there is no
    forward projection to average yet at the jump-off itself). Used only
    to establish a baseline allowance level to roll the projection's
    provision forward from -- not a claim about what the real bank
    actually held as its allowance that quarter (this project has no
    per-category real allowance data -- see panel.py's allowance
    functions, which are bank-TOTAL, not split by category)."""
    window_quarters = [as_of_quarter - i for i in range(wal_quarters)]
    available = realized_annualized_nco_rate.reindex(window_quarters).dropna()
    if available.empty:
        raise ValueError(f"no realized data available on or before {as_of_quarter}")
    return float(available.mean())


def build_cecl_projection(
    realized_annualized_nco_rate: pd.Series,
    projected_annualized_nco_rate: pd.Series,
    category: str,
    starting_balance: float,
) -> pd.DataFrame:
    """The simplified CECL allowance/provision roll-forward for one
    category/scenario. `realized_annualized_nco_rate`: the real, actual
    annualized NCO rate history (quarter-indexed), used only to establish
    the jump-off quarter's baseline allowance level (via
    `backward_looking_lifetime_expected_loss_rate`). `projected_
    annualized_nco_rate`: the projected path (quarter-indexed) for the
    scenario/model being evaluated, covering the projection horizon
    ONLY (e.g. 2026Q1-2029Q1) -- must NOT include the jump-off quarter
    itself. `starting_balance`: the category's static (held-constant)
    balance, e.g. its last actual (jump-off) average balance.

    Returns a frame indexed by quarter (the jump-off quarter plus every
    projected quarter) with columns: "lifetime_expected_loss_rate",
    "allowance_required" (= starting_balance x that rate),
    "net_charge_off" (quarterly, not annualized: projected/realized rate
    / 4 x starting_balance -- NaN for the jump-off row, since that's a
    baseline, not a quarter being rolled forward from a prior one), and
    "provision_expense" (allowance_required's own quarter-over-quarter
    change plus that quarter's net_charge_off, via the standard
    allowance roll-forward identity: ending = beginning + provision -
    net_charge_offs => provision = (ending - beginning) + net_charge_offs)."""
    wal_quarters = CATEGORY_WEIGHTED_AVERAGE_LIFE_QUARTERS[category]
    jump_off_quarter = realized_annualized_nco_rate.index.max()
    if jump_off_quarter in projected_annualized_nco_rate.index:
        raise ValueError(
            "projected_annualized_nco_rate must not include the jump-off quarter "
            f"({jump_off_quarter}) -- it's supplied separately via realized_annualized_nco_rate"
        )

    jump_off_rate = backward_looking_lifetime_expected_loss_rate(
        realized_annualized_nco_rate, jump_off_quarter, wal_quarters
    )
    rows = [
        {
            "quarter": jump_off_quarter,
            "lifetime_expected_loss_rate": jump_off_rate,
            "allowance_required": starting_balance * jump_off_rate,
            "net_charge_off": float("nan"),
            "provision_expense": float("nan"),
        }
    ]

    projected_sorted = projected_annualized_nco_rate.sort_index()
    prior_allowance = rows[0]["allowance_required"]
    for quarter in projected_sorted.index:
        lifetime_rate = lifetime_expected_loss_rate(projected_sorted, quarter, wal_quarters)
        allowance_required = starting_balance * lifetime_rate
        net_charge_off = (projected_sorted.loc[quarter] / 4.0) * starting_balance
        provision_expense = (allowance_required - prior_allowance) + net_charge_off
        rows.append(
            {
                "quarter": quarter,
                "lifetime_expected_loss_rate": lifetime_rate,
                "allowance_required": allowance_required,
                "net_charge_off": net_charge_off,
                "provision_expense": provision_expense,
            }
        )
        prior_allowance = allowance_required

    return pd.DataFrame(rows).set_index("quarter")
