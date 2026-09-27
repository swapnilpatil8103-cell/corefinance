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

from corefin.credit.schema import (
    CATEGORY_MDRM_CODES,
    PROVISION_EXPENSE_ITEM,
    TOTAL_ALLOWANCE_ITEM,
    LoanCategory,
    MdrmCodeSet,
)

# RCFD3123 (consolidated) fallback for TOTAL_ALLOWANCE_ITEM (RCON3123,
# domestic-only) -- confirmed a REAL, actually-used fallback against a real
# 2021Q4 bulk file: 84 of 4887 banks had RCON3123 missing, and every one of
# those 84 had RCFD3123 populated instead (likely FFIEC 031 filers whose
# allowance is reported only on a consolidated basis) -- unlike every
# per-category balance item in schema.py, where the RCFD fallback is purely
# defensive and never triggered in practice.
TOTAL_ALLOWANCE_RCFD_FALLBACK_ITEM = "RCFD3123"

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


def build_panel(
    item_frame: pd.DataFrame,
    categories: list[LoanCategory] | None = None,
    min_balance: float = 1_000.0,
    cecl_adoption_quarter: pd.Period = DEFAULT_CECL_TRANSITION_QUARTER,
    merger_jump_threshold: float = 0.5,
) -> pd.DataFrame:
    """Orchestrates the whole per-category panel build. `item_frame` needs
    "bank_id" and "quarter" (pandas Period, freq="Q") columns plus every
    raw MDRM item column any of `categories`' code sets reference across
    the quarters present (one row per bank-quarter). `categories` defaults
    to every LoanCategory (including AUTO_AND_OTHER_CONSUMER_COMBINED --
    pass an explicit list to exclude it, e.g. when totaling "all loan
    categories" and its intentional overlap with AUTO/OTHER_CONSUMER would
    double-count).

    A category/quarter with no covering MdrmCodeSet (e.g. AUTO before
    2011Q1) is simply excluded from that category's rows, not zero-filled.

    Returns one row per (bank_id, category, quarter) with balance,
    past_due_30_89, past_due_90, nonaccrual, chargeoff_ytd, recovery_ytd
    (straight from `apply_category_mapping`), plus chargeoff_quarterly/
    recovery_quarterly (`ytd_to_quarterly`, per bank), average_balance
    (`average_balance`, per bank), annualized_nco_rate
    (`annualized_nco_rate`), merger_flag (`flag_merger_discontinuities`,
    per bank), cecl_regime (`apply_cecl_regime_dummy`) and keep (the
    min-balance mask from `apply_min_balance_filter` -- filtering on it is
    the caller's choice, not applied here)."""
    categories = categories if categories is not None else list(LoanCategory)
    category_frames = []

    for category in categories:
        quarter_frames = []
        for quarter, group in item_frame.groupby("quarter", sort=True):
            try:
                code_set_for_quarter(category, quarter)
            except ValueError:
                continue
            mapped = apply_category_mapping(group, category, group["quarter"])
            mapped["bank_id"] = group["bank_id"].to_numpy()
            mapped["quarter"] = quarter
            quarter_frames.append(mapped)
        if not quarter_frames:
            continue

        cat_panel = pd.concat(quarter_frames, ignore_index=True)
        cat_panel = cat_panel.sort_values(["bank_id", "quarter"]).reset_index(drop=True)

        # Vectorized per-bank "prior quarter" lookups via groupby().shift() --
        # NOT per-group .apply(pure_function), which scales terribly across
        # the tens of thousands of (bank, category) groups a full historical
        # build produces. Every formula below is identical to the
        # corresponding pure function (ytd_to_quarterly/average_balance/
        # flag_merger_discontinuities), just applied across all banks in one
        # vectorized pass instead of one Python-level call per bank.
        by_bank = cat_panel.groupby("bank_id", sort=False)
        is_q1 = pd.PeriodIndex(cat_panel["quarter"]).quarter == 1
        prior_chargeoff_ytd = by_bank["chargeoff_ytd"].shift(1)
        prior_recovery_ytd = by_bank["recovery_ytd"].shift(1)
        cat_panel["chargeoff_quarterly"] = np.where(
            is_q1, cat_panel["chargeoff_ytd"], cat_panel["chargeoff_ytd"] - prior_chargeoff_ytd
        )
        cat_panel["recovery_quarterly"] = np.where(
            is_q1, cat_panel["recovery_ytd"], cat_panel["recovery_ytd"] - prior_recovery_ytd
        )

        prior_balance = by_bank["balance"].shift(1)
        cat_panel["average_balance"] = (cat_panel["balance"] + prior_balance) / 2.0
        cat_panel["annualized_nco_rate"] = annualized_nco_rate(
            cat_panel["chargeoff_quarterly"],
            cat_panel["recovery_quarterly"],
            cat_panel["average_balance"],
        )

        pct_change = (cat_panel["balance"] - prior_balance) / prior_balance.replace(0, np.nan)
        cat_panel["merger_flag"] = (pct_change.abs() > merger_jump_threshold).fillna(False)

        cat_panel["cecl_regime"] = apply_cecl_regime_dummy(
            cat_panel["quarter"], cecl_adoption_quarter
        )
        cat_panel["keep"] = apply_min_balance_filter(cat_panel["balance"], min_balance)
        cat_panel["category"] = category
        category_frames.append(cat_panel)

    if not category_frames:
        return pd.DataFrame(
            columns=[
                "bank_id",
                "category",
                "quarter",
                "balance",
                "past_due_30_89",
                "past_due_90",
                "nonaccrual",
                "chargeoff_ytd",
                "recovery_ytd",
                "chargeoff_quarterly",
                "recovery_quarterly",
                "average_balance",
                "annualized_nco_rate",
                "merger_flag",
                "cecl_regime",
                "keep",
            ]
        )
    return pd.concat(category_frames, ignore_index=True)


def build_industry_nco_rate_report(panel: pd.DataFrame) -> pd.DataFrame:
    """panel: long-format frame (as produced by `build_panel`) with
    columns "category", "quarter", "chargeoff_quarterly",
    "recovery_quarterly", "average_balance". Returns one row per
    (category, quarter) with the aggregate industry annualized NCO rate --
    sum(chargeoff_quarterly) - sum(recovery_quarterly), divided by
    sum(average_balance), annualized by x4 -- and `has_chargeoff_data`
    (True if at least one bank has a non-missing chargeoff_quarterly that
    quarter; see `flag_chargeoff_gaps`)."""
    grouped = panel.groupby(["category", "quarter"], observed=True)
    agg = grouped.agg(
        aggregate_chargeoff=("chargeoff_quarterly", lambda s: s.sum(skipna=True)),
        aggregate_recovery=("recovery_quarterly", lambda s: s.sum(skipna=True)),
        aggregate_average_balance=("average_balance", lambda s: s.sum(skipna=True)),
        has_chargeoff_data=("chargeoff_quarterly", lambda s: bool(s.notna().any())),
    ).reset_index()

    nco = agg["aggregate_chargeoff"] - agg["aggregate_recovery"]
    avg = agg["aggregate_average_balance"].to_numpy()
    with np.errstate(divide="ignore", invalid="ignore"):
        rate = np.where(avg > 0, (nco.to_numpy() / avg) * 4.0, np.nan)
    agg["industry_nco_rate"] = rate
    return agg.sort_values(["category", "quarter"]).reset_index(drop=True)


def flag_chargeoff_gaps(
    industry_nco_report: pd.DataFrame,
    start: str = "2008Q1",
    end: str = "2010Q4",
    categories: list[LoanCategory] | None = None,
) -> pd.Series:
    """industry_nco_report: as produced by `build_industry_nco_rate_report`.
    Returns a Series indexed by `categories` (default: every LoanCategory):
    True if ANY quarter in [start, end] (inclusive "YYYYQN" strings) has no
    charge-off data for that category -- a gap in exactly the window a
    crisis-period backtest needs is a hard blocker, not a cosmetic issue.
    A category with NO rows at all in the window (e.g. AUTO before its
    2011Q1 start) is flagged True too, not silently omitted -- "missing
    every quarter" is the most extreme case of "has a gap," and dropping
    it from the output would look like "no gap" to a reader skimming the
    result."""
    categories = categories if categories is not None else list(LoanCategory)
    start_q = pd.Period(start, freq="Q")
    end_q = pd.Period(end, freq="Q")
    expected_quarters = pd.period_range(start_q, end_q, freq="Q")
    window = industry_nco_report[
        (industry_nco_report["quarter"] >= start_q) & (industry_nco_report["quarter"] <= end_q)
    ]
    by_category = {
        name: group["has_chargeoff_data"]
        for name, group in window.groupby("category", observed=True)
    }

    gaps = {}
    for category in categories:
        category_data = by_category.get(category)
        if category_data is None or len(category_data) < len(expected_quarters):
            gaps[category] = True  # missing quarter(s) entirely -- also a gap
        else:
            gaps[category] = not category_data.all()
    return pd.Series(gaps)


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


def build_allowance_panel(item_frame: pd.DataFrame) -> pd.DataFrame:
    """Bank-quarter (not per-category) allowance and provision figures.
    `item_frame` needs "bank_id", "quarter" plus TOTAL_ALLOWANCE_ITEM
    (RCON3123), TOTAL_ALLOWANCE_RCFD_FALLBACK_ITEM (RCFD3123, optional --
    used only where RCON3123 is missing) and PROVISION_EXPENSE_ITEM
    (RIAD4230, year-to-date) columns. Returns bank_id, quarter,
    allowance_balance, provision_ytd, provision_quarterly (the last via
    `ytd_to_quarterly`, per bank, vectorized the same way `build_panel`
    computes chargeoff_quarterly)."""
    rcon = item_frame[TOTAL_ALLOWANCE_ITEM]
    if TOTAL_ALLOWANCE_RCFD_FALLBACK_ITEM in item_frame.columns:
        rcfd = item_frame[TOTAL_ALLOWANCE_RCFD_FALLBACK_ITEM]
        allowance_balance = rcon.where(rcon.notna(), rcfd)
    else:
        allowance_balance = rcon

    result = pd.DataFrame(
        {
            "bank_id": item_frame["bank_id"],
            "quarter": item_frame["quarter"],
            "allowance_balance": allowance_balance,
            "provision_ytd": item_frame[PROVISION_EXPENSE_ITEM],
        }
    )
    result = result.sort_values(["bank_id", "quarter"]).reset_index(drop=True)

    quarters = pd.PeriodIndex(result["quarter"])
    is_q1 = quarters.quarter == 1
    prior_provision_ytd = result.groupby("bank_id", sort=False)["provision_ytd"].shift(1)
    result["provision_quarterly"] = np.where(
        is_q1, result["provision_ytd"], result["provision_ytd"] - prior_provision_ytd
    )
    return result


def compute_bank_allowance_rollforward(
    allowance_panel: pd.DataFrame,
    category_panel: pd.DataFrame,
    categories: list[LoanCategory] | None = None,
) -> pd.DataFrame:
    """Joins `allowance_panel` (from `build_allowance_panel`) with total
    net charge-offs aggregated ACROSS categories from `category_panel`
    (from `build_panel`) to compute the bank-quarter allowance
    roll-forward residual. `categories` defaults to every LoanCategory
    EXCEPT AUTO_AND_OTHER_CONSUMER_COMBINED (summing it alongside AUTO/
    OTHER_CONSUMER would double-count, per schema.py's module docstring) --
    pass an explicit list to change that.

    Returns bank_id, quarter, allowance_balance, beginning_allowance (the
    prior quarter's allowance_balance, per bank -- NaN for each bank's
    first quarter), provision_quarterly, total_net_chargeoffs, and
    residual (`allowance_rollforward_residual`). A nonzero residual is
    expected against real data (a small "other adjustments" Call Report
    line isn't modeled here); large residuals point to a mapping or
    differencing error, not just real-world noise."""
    categories = (
        categories
        if categories is not None
        else [c for c in LoanCategory if c != LoanCategory.AUTO_AND_OTHER_CONSUMER_COMBINED]
    )
    scoped = category_panel[category_panel["category"].isin(categories)]
    net_chargeoffs = scoped.assign(
        net_chargeoff=scoped["chargeoff_quarterly"] - scoped["recovery_quarterly"]
    )
    total_net_chargeoffs = (
        net_chargeoffs.groupby(["bank_id", "quarter"], observed=True)["net_chargeoff"]
        .sum(min_count=1)
        .reset_index()
        .rename(columns={"net_chargeoff": "total_net_chargeoffs"})
    )

    merged = allowance_panel.merge(total_net_chargeoffs, on=["bank_id", "quarter"], how="left")
    merged = merged.sort_values(["bank_id", "quarter"]).reset_index(drop=True)
    prior_allowance = merged.groupby("bank_id", sort=False)["allowance_balance"]
    merged["beginning_allowance"] = prior_allowance.shift(1)

    merged["residual"] = allowance_rollforward_residual(
        merged["beginning_allowance"],
        merged["provision_quarterly"],
        merged["total_net_chargeoffs"],
        merged["allowance_balance"],
    )
    return merged[
        [
            "bank_id",
            "quarter",
            "allowance_balance",
            "beginning_allowance",
            "provision_quarterly",
            "total_net_chargeoffs",
            "residual",
        ]
    ]
