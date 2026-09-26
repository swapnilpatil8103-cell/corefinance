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

from corefin.credit.schema import CATEGORY_MDRM_CODES, LoanCategory, MdrmCodeSet

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


def code_set_for_quarter(category: LoanCategory, quarter: pd.Period) -> MdrmCodeSet:
    """The MDRM code set covering `quarter` for `category`, per the
    date-effective mapping in schema.CATEGORY_MDRM_CODES. Raises if no
    code set covers that quarter (e.g. LoanCategory.AUTO before 2011Q1) --
    that is a real "this category has no data here" condition, not
    something to paper over with an empty/zero result."""
    for code_set in CATEGORY_MDRM_CODES[category]:
        if code_set.covers(quarter):
            return code_set
    raise ValueError(f"no MDRM code set covers {category} at {quarter}")


def apply_category_mapping(
    item_frame: pd.DataFrame, category: LoanCategory, quarter_period: pd.Series
) -> pd.DataFrame:
    """item_frame: one row per bank-quarter, one column per MDRM item code
    (e.g. "RCON1766"). `quarter_period`: aligned to item_frame's index,
    used to pick the right date-effective MdrmCodeSet per row (a category
    like cre_nonfarm_nonresidential uses different item codes before vs.
    after 2007Q1). Returns a frame with standardized columns (balance,
    past_due_30_89, past_due_90, nonaccrual, chargeoff_ytd, recovery_ytd).
    A field left empty in schema.py (e.g. other_consumer's past-due items)
    produces an all-NaN column -- "not available" and "zero" are not the
    same thing here. Any component item that is NaN makes the sum NaN
    rather than silently treating it as zero. Rows whose quarter isn't
    covered by any code set for `category` (e.g. auto loans before
    2011Q1) raise, rather than being silently dropped or zero-filled."""
    quarters = pd.PeriodIndex(quarter_period).asfreq("Q")
    columns = [
        "balance",
        "past_due_30_89",
        "past_due_90",
        "nonaccrual",
        "chargeoff_ytd",
        "recovery_ytd",
    ]
    result = pd.DataFrame(np.nan, index=item_frame.index, columns=columns)
    unmatched = np.ones(len(item_frame), dtype=bool)

    def _sum_or_nan(idx: pd.Index, items: tuple[str, ...]) -> pd.Series:
        if not items:
            return pd.Series(np.nan, index=idx)
        missing = [c for c in items if c not in item_frame.columns]
        if missing:
            raise KeyError(f"item_frame is missing required columns: {missing}")
        return item_frame.loc[idx, list(items)].sum(axis=1, skipna=False)

    for code_set in CATEGORY_MDRM_CODES[category]:
        mask = np.array([code_set.covers(q) for q in quarters]) & unmatched
        if not mask.any():
            continue
        idx = item_frame.index[mask]
        result.loc[idx, "balance"] = _sum_or_nan(idx, code_set.balance_items)
        result.loc[idx, "past_due_30_89"] = _sum_or_nan(idx, code_set.past_due_30_89_items)
        result.loc[idx, "past_due_90"] = _sum_or_nan(idx, code_set.past_due_90_items)
        result.loc[idx, "nonaccrual"] = _sum_or_nan(idx, code_set.nonaccrual_items)
        result.loc[idx, "chargeoff_ytd"] = _sum_or_nan(idx, code_set.chargeoff_items)
        result.loc[idx, "recovery_ytd"] = _sum_or_nan(idx, code_set.recovery_items)
        unmatched &= ~mask

    if unmatched.any():
        bad_quarters = sorted({str(q) for q in quarters[unmatched]})
        raise ValueError(f"no MDRM code set covers {category} for quarter(s) {bad_quarters}")

    return result


def apply_rcfd_fallback(
    rcon_balance: pd.Series, rcfd_balance: pd.Series, reporting_form: pd.Series
) -> pd.Series:
    """For FFIEC 031 (large/international bank) filers, a balance item
    could in principle be reported only on a consolidated (RCFD, domestic
    + foreign) basis rather than the domestic-only RCON basis this panel
    otherwise uses. Falls back to `rcfd_balance` only where `reporting_form
    == "FFIEC 031"` and `rcon_balance` is missing -- FFIEC 041/051 filers
    have no foreign offices and no RCFD equivalent, so they never fall
    back. As of schema.py's MDRM verification, no current category
    actually triggers this (RCON stays available on FFIEC 031 for every
    category's whole reporting history); this function is a defensive
    mechanism for whichever category eventually does, exercised in tests
    with a synthetic gap."""
    is_031 = reporting_form == "FFIEC 031"
    use_fallback = is_031 & rcon_balance.isna()
    return rcon_balance.where(~use_fallback, rcfd_balance)


def build_coverage_report(panel: pd.DataFrame) -> pd.DataFrame:
    """panel: long-format frame with columns "category", "quarter" (a
    pandas Period or "YYYYQN" string) and "balance" (one row per
    bank-category-quarter, already through `apply_category_mapping`).
    Returns one row per (category, quarter) with:
    - n_banks: number of bank-quarter rows in that group
    - coverage_share: fraction with a non-missing balance
    - aggregate_balance: sum of non-missing balances
    - aggregate_pct_change: quarter-over-quarter change in aggregate_balance
      within that category
    - code_switch: True if `quarter` is a code set's `valid_from` boundary
      for that category (per schema.CATEGORY_MDRM_CODES)
    - possible_break: True if `code_switch` is True AND
      abs(aggregate_pct_change) exceeds `break_threshold` (default 15%) --
      a large jump exactly at a known code-set boundary is the signature
      of an incomplete or mismatched mapping, not necessarily a real
      economic move.
    """
    quarters = pd.PeriodIndex(panel["quarter"].astype(str), freq="Q")
    working = panel.assign(quarter=quarters)

    grouped = working.groupby(["category", "quarter"], observed=True)["balance"]
    report = grouped.agg(
        n_banks="size",
        coverage_share=lambda s: s.notna().mean(),
        aggregate_balance=lambda s: s.sum(skipna=True),
    ).reset_index()
    report = report.sort_values(["category", "quarter"]).reset_index(drop=True)
    pct_change = report.groupby("category", observed=True)["aggregate_balance"].pct_change()
    report["aggregate_pct_change"] = pct_change

    switch_quarters: dict[str, set[pd.Period]] = {
        category: {pd.Period(code_set.valid_from, freq="Q") for code_set in code_sets}
        for category, code_sets in CATEGORY_MDRM_CODES.items()
    }
    report["code_switch"] = [
        row.quarter in switch_quarters.get(row.category, set()) for row in report.itertuples()
    ]

    break_threshold = 0.15
    report["possible_break"] = report["code_switch"] & (
        report["aggregate_pct_change"].abs() > break_threshold
    )
    return report


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
