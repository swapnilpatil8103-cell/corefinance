"""Pure, testable transformations for the bank-quarter loan-loss panel.

Every function here takes and returns pandas Series/DataFrames (not yet
the (n_scenarios, n_periods) numpy arrays used downstream in the
projection engine -- this layer sits between raw MDRM-item-coded Call
Report rows and the modeling stage). No function in this module touches
the network; fetching is `sources/*.py`'s job, orchestration is the CLI's.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from corefin.credit.schema import CATEGORY_MDRM_CODES, LoanCategory

# CECL is mandatory for large SEC filers starting 2020Q1, phased in for
# smaller/non-SEC filers through 2023. This default is only a fallback for
# quick synthetic-data smoke tests -- real panel-building must pass each
# bank's own adoption quarter to `apply_cecl_regime_dummy`, not rely on it.
DEFAULT_CECL_TRANSITION_QUARTER = pd.Period("2020Q1", freq="Q")


def ytd_to_quarterly(ytd: pd.Series, quarter_period: pd.Series) -> pd.Series:
    """Call Report RIAD (charge-off/recovery) items are reported calendar
    year-to-date. Convert to a per-quarter flow: Q1's value is already a
    quarterly flow; Q2-Q4 are the YTD value minus the prior quarter's YTD
    value. Pass one bank's rows at a time, sorted by quarter -- this does
    not group by bank itself.
    """
    quarters = pd.PeriodIndex(quarter_period).asfreq("Q")
    is_q1 = quarters.quarter == 1
    prior_ytd = ytd.shift(1).to_numpy()
    quarterly = np.where(is_q1, ytd.to_numpy(), ytd.to_numpy() - prior_ytd)
    return pd.Series(quarterly, index=ytd.index)


def average_balance(balance: pd.Series) -> pd.Series:
    """Average of the current and prior quarter's period-end balance, the
    standard denominator for an annualized NCO rate. The first quarter in
    a series has no prior balance and is NaN, not zero-filled."""
    return (balance + balance.shift(1)) / 2.0


def annualized_nco_rate(
    gross_chargeoffs_quarterly: pd.Series,
    recoveries_quarterly: pd.Series,
    average_balance_: pd.Series,
) -> pd.Series:
    """NCO = gross charge-offs - recoveries, both already per-quarter
    flows (run `ytd_to_quarterly` first, not the raw YTD values).
    Annualized by multiplying the quarterly rate by 4 -- the standard
    convention, not a trailing-12-month sum, which would smooth over the
    single-quarter spikes this project needs to resolve."""
    nco = gross_chargeoffs_quarterly.to_numpy() - recoveries_quarterly.to_numpy()
    avg = average_balance_.to_numpy()
    with np.errstate(divide="ignore", invalid="ignore"):
        rate = np.where(avg > 0, (nco / avg) * 4.0, np.nan)
    return pd.Series(rate, index=gross_chargeoffs_quarterly.index)


def apply_category_mapping(item_frame: pd.DataFrame, category: LoanCategory) -> pd.DataFrame:
    """item_frame: one row per bank-quarter, one column per MDRM item code
    (e.g. "RCON1766"). Returns a frame with standardized columns (balance,
    past_due_30_89, past_due_90, nonaccrual, chargeoff_ytd, recovery_ytd)
    for `category`, summing across item codes where a category maps to
    more than one Call Report line. A field left empty in schema.py (e.g.
    other_consumer's past-due items) produces an all-NaN column -- "not
    available" and "zero" are not the same thing here. Any component item
    that is NaN makes the sum NaN rather than silently treating it as
    zero."""
    codes = CATEGORY_MDRM_CODES[category]

    def _sum_or_nan(items: tuple[str, ...]) -> pd.Series:
        if not items:
            return pd.Series(np.nan, index=item_frame.index)
        missing = [c for c in items if c not in item_frame.columns]
        if missing:
            raise KeyError(f"item_frame is missing required columns: {missing}")
        return item_frame[list(items)].sum(axis=1, skipna=False)

    return pd.DataFrame(
        {
            "balance": _sum_or_nan(codes.balance_items),
            "past_due_30_89": _sum_or_nan(codes.past_due_30_89_items),
            "past_due_90": _sum_or_nan(codes.past_due_90_items),
            "nonaccrual": _sum_or_nan(codes.nonaccrual_items),
            "chargeoff_ytd": _sum_or_nan(codes.chargeoff_items),
            "recovery_ytd": _sum_or_nan(codes.recovery_items),
        }
    )


def flag_merger_discontinuities(balance: pd.Series, jump_threshold: float = 0.5) -> pd.Series:
    """Flags (does not drop) bank-quarters where balance jumps more than
    `jump_threshold` (50% default) quarter-over-quarter -- a cheap proxy
    for an unreported merger/acquisition or a large loan-portfolio sale,
    pending a real merger-history join (FDIC's `institutions`/`financials`
    endpoints don't expose merger events directly; a dedicated M&A-history
    source is a follow-up, not built here). Returns a boolean Series
    aligned to `balance`'s index; the first observation in any series
    can't be evaluated and is False."""
    prior = balance.shift(1)
    pct_change = (balance - prior) / prior.replace(0, np.nan)
    return (pct_change.abs() > jump_threshold).fillna(False)


def apply_cecl_regime_dummy(
    quarter_period: pd.Series,
    cecl_adoption_quarter: pd.Period = DEFAULT_CECL_TRANSITION_QUARTER,
) -> pd.Series:
    """Returns a 0/1 regime dummy: 1 from `cecl_adoption_quarter` onward
    (inclusive), 0 before. Real per-bank adoption quarter varies (large
    SEC filers: 2020Q1; smaller/non-SEC filers phased through 2023) --
    pass each bank's own adoption quarter when building a real panel."""
    quarters = pd.PeriodIndex(quarter_period).asfreq("Q")
    return pd.Series((quarters >= cecl_adoption_quarter).astype(int), index=quarter_period.index)


def apply_min_balance_filter(balance: pd.Series, minimum: float) -> pd.Series:
    """Keep-mask: True where `balance` is at least `minimum` and not NaN.
    Filters out bank-quarters where a category's balance is too small for
    a charge-off rate to be meaningful -- a single defaulted loan in an
    otherwise-tiny book can produce a huge, noisy NCO rate."""
    return balance.notna() & (balance >= minimum)


def allowance_rollforward_residual(
    beginning_allowance: pd.Series,
    provision: pd.Series,
    net_chargeoffs: pd.Series,
    ending_allowance: pd.Series,
) -> pd.Series:
    """Residual of the standard allowance roll-forward identity: ending =
    beginning + provision - net_chargeoffs. Real Call Report data also has
    a small "other adjustments" line not modeled in this simplified
    identity, so a small nonzero residual against real data is expected;
    against synthetic fixtures built to satisfy the identity exactly, the
    residual should be ~0 (see checks.framework.check_close_to_zero)."""
    return ending_allowance - (beginning_allowance + provision - net_chargeoffs)
