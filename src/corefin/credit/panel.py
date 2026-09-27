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
    CECL_ADOPTION_INDICATOR_ITEMS,
    PROVISION_EXPENSE_ITEM,
    TOTAL_ALLOWANCE_ITEM,
    TOTAL_CHARGEOFF_ITEM,
    TOTAL_RECOVERY_ITEM,
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
# smaller/non-SEC filers through 2023Q1. This is the fallback used for
# banks `derive_cecl_adoption_quarters` never detects a signal for --
# real panel-building should always try that first (see `build_panel`),
# using this default only for banks the signal doesn't cover.
#
# 2023Q1, not 2020Q1, on purpose: verified against real data (a full
# 2001Q1-2026Q2 build), only 2,503 of the ~4,700-8,000 banks per quarter
# ever show a detectable RIADJJ26/JJ28 signal at all, split into two
# waves matching ASU 2016-13's known rollout (222 banks in 2020Q1, the
# large-SEC-filer mandatory date; 1,981 in 2023Q1, the final mandatory
# date for smaller/private companies -- see
# `summarize_cecl_adoption_counts`). The large majority of ALL banks
# (undetected ones included) are small/private institutions with no SEC
# filing obligation, so 2023Q1 is the correct default for them -- their
# adoption typically had little or no day-one allowance change to
# register as a nonzero RIADJJ26/JJ28 value, which is presumably why they
# go undetected in the first place, not because they adopted early.
#
# A total-assets-based rule to identify "large SEC filer -> use 2020Q1
# instead" was tried and rejected: the 222 CONFIRMED 2020Q1 adopters span
# total assets from $293 million to $93 billion (median ~$7.0 billion),
# with no threshold that meaningfully separates them from the broader
# population (42% of ALL banks in 2020Q1 already exceed the confirmed
# adopters' minimum asset size). Call Report data has no direct SEC-filer
# or public-company-status item at all (searched MDRM; none exists), and
# asset size alone is evidently too weak a proxy for actual SEC-filer
# status to justify a differentiated default -- so a single default
# (2023Q1) is used uniformly for every undetected bank instead.
DEFAULT_CECL_TRANSITION_QUARTER = pd.Period("2023Q1", freq="Q")


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


def derive_cecl_adoption_quarters(item_frame: pd.DataFrame) -> pd.Series:
    """Returns a Series indexed by bank_id: each bank's own CECL adoption
    quarter, derived as the FIRST quarter in `item_frame` where either
    CECL_ADOPTION_INDICATOR_ITEMS column (RIADJJ26/RIADJJ28, only ever
    populated 2019Q1-2023Q4) is reported non-null and nonzero -- see
    schema.py's note for the real-data verification behind this. Banks
    with no such quarter in `item_frame` (not covered by this window, or
    the columns are entirely absent) are simply not included in the
    result -- callers should fall back to a default adoption quarter for
    them (see `apply_cecl_regime_dummy`'s `default_adoption_quarter`),
    not treat their absence as "never adopted.\""""
    available = [c for c in CECL_ADOPTION_INDICATOR_ITEMS if c in item_frame.columns]
    if not available:
        return pd.Series(dtype="object")
    is_adoption_signal = (item_frame[available].fillna(0) != 0).any(axis=1)
    signal_rows = item_frame.loc[is_adoption_signal, ["bank_id", "quarter"]]
    return signal_rows.groupby("bank_id")["quarter"].min()


def summarize_cecl_adoption_counts(adoption_quarters: pd.Series) -> pd.Series:
    """adoption_quarters: from `derive_cecl_adoption_quarters`. Returns
    the number of banks whose derived adoption quarter falls in each
    quarter, sorted chronologically -- expect two clusters, one around
    2020Q1 (large SEC filers' mandatory date) and a larger one around
    2023Q1 (the final mandatory date for smaller/private companies)."""
    return adoption_quarters.value_counts().sort_index()


def resolve_cecl_adoption_quarters(
    bank_ids: pd.Series,
    detected_adoption_quarters: pd.Series,
    default_adoption_quarter: pd.Period = DEFAULT_CECL_TRANSITION_QUARTER,
) -> pd.DataFrame:
    """bank_ids: every bank_id `apply_cecl_regime_dummy` will be asked
    about (e.g. item_frame["bank_id"] or category_panel["bank_id"]).
    detected_adoption_quarters: from `derive_cecl_adoption_quarters`.
    Returns one row per unique bank_id in `bank_ids`, with columns
    "adoption_quarter" (the detected quarter, or `default_adoption_quarter`
    if undetected) and "detected" (True if it came from a real RIADJJ26/
    JJ28 signal, False if defaulted). This is a reporting utility -- it
    doesn't change `apply_cecl_regime_dummy`'s own behavior (which already
    applies the same fallback internally), it just makes the detected-vs-
    defaulted split visible; see `summarize_cecl_adoption_detected_vs_defaulted`."""
    unique_banks = pd.Index(bank_ids.unique(), name="bank_id")
    detected = detected_adoption_quarters.reindex(unique_banks)
    resolved = detected.where(detected.notna(), default_adoption_quarter)
    return pd.DataFrame(
        {
            "bank_id": unique_banks,
            "adoption_quarter": resolved.to_numpy(),
            "detected": detected.notna().to_numpy(),
        }
    )


def summarize_cecl_adoption_detected_vs_defaulted(resolved: pd.DataFrame) -> pd.DataFrame:
    """resolved: from `resolve_cecl_adoption_quarters`. Returns one row per
    adoption_quarter with n_detected (banks whose adoption quarter came
    from a real RIADJJ26/JJ28 signal) and n_defaulted (banks that never
    showed the signal and fell back to `default_adoption_quarter`),
    sorted chronologically."""
    counts = resolved.groupby(["adoption_quarter", "detected"]).size().unstack("detected")
    counts = counts.rename(columns={True: "n_detected", False: "n_defaulted"})
    for col in ("n_detected", "n_defaulted"):
        if col not in counts.columns:
            counts[col] = 0
    counts = counts[["n_detected", "n_defaulted"]].fillna(0).astype(int)
    return counts.sort_index()


def apply_cecl_regime_dummy(
    bank_id: pd.Series,
    quarter_period: pd.Series,
    adoption_quarters: pd.Series | None = None,
    default_adoption_quarter: pd.Period = DEFAULT_CECL_TRANSITION_QUARTER,
) -> pd.Series:
    """Returns a 0/1 regime dummy: 1 from each bank's OWN adoption quarter
    onward (inclusive), 0 before. `adoption_quarters` (e.g. from
    `derive_cecl_adoption_quarters`): a Series indexed by bank_id giving
    each bank's real adoption quarter. A bank missing from
    `adoption_quarters` (or `adoption_quarters=None` entirely) uses
    `default_adoption_quarter` instead -- a real approximation applied
    uniformly, not a per-bank fact, so prefer passing
    `derive_cecl_adoption_quarters`'s output whenever `item_frame` covers
    2019Q1-2023Q4."""
    quarters = pd.PeriodIndex(quarter_period).asfreq("Q")
    if adoption_quarters is not None:
        mapped = bank_id.map(adoption_quarters)
        per_row_adoption = mapped.where(mapped.notna(), default_adoption_quarter)
    else:
        per_row_adoption = pd.Series(default_adoption_quarter, index=quarter_period.index)
    regime = quarters.to_numpy() >= per_row_adoption.to_numpy()
    return pd.Series(regime.astype(int), index=quarter_period.index)


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
    default_cecl_adoption_quarter: pd.Period = DEFAULT_CECL_TRANSITION_QUARTER,
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
    per bank), cecl_regime and keep (the min-balance mask from
    `apply_min_balance_filter` -- filtering on it is the caller's choice,
    not applied here).

    cecl_regime uses each bank's OWN CECL adoption quarter, derived from
    `item_frame` via `derive_cecl_adoption_quarters` (needs RIADJJ26/JJ28,
    only present 2019Q1-2023Q4 -- if `item_frame` doesn't cover that
    window, or a specific bank never shows the signal within it, that
    bank falls back to `default_cecl_adoption_quarter` uniformly)."""
    categories = categories if categories is not None else list(LoanCategory)
    adoption_quarters = derive_cecl_adoption_quarters(item_frame)
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
            cat_panel["bank_id"],
            cat_panel["quarter"],
            adoption_quarters,
            default_cecl_adoption_quarter,
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
    (RIAD4230, year-to-date), TOTAL_CHARGEOFF_ITEM (RIADC079, year-to-date)
    and TOTAL_RECOVERY_ITEM (RIAD4605, year-to-date) columns. Returns
    bank_id, quarter, allowance_balance, provision_ytd, provision_quarterly,
    total_chargeoff_ytd, total_chargeoff_quarterly, total_recovery_ytd,
    total_recovery_quarterly, and allowance_merger_flag
    (`flag_merger_discontinuities` on allowance_balance, per bank -- a
    merger/acquisition adds the acquired bank's allowance as a lump sum
    that breaks the roll-forward identity by construction; see
    `compute_bank_allowance_rollforward`, which excludes flagged
    bank-quarters from its summary statistics)."""
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
            "total_chargeoff_ytd": item_frame[TOTAL_CHARGEOFF_ITEM],
            "total_recovery_ytd": item_frame[TOTAL_RECOVERY_ITEM],
        }
    )
    result = result.sort_values(["bank_id", "quarter"]).reset_index(drop=True)

    quarters = pd.PeriodIndex(result["quarter"])
    is_q1 = quarters.quarter == 1
    by_bank = result.groupby("bank_id", sort=False)
    for ytd_col, quarterly_col in (
        ("provision_ytd", "provision_quarterly"),
        ("total_chargeoff_ytd", "total_chargeoff_quarterly"),
        ("total_recovery_ytd", "total_recovery_quarterly"),
    ):
        prior_ytd = by_bank[ytd_col].shift(1)
        result[quarterly_col] = np.where(
            is_q1, result[ytd_col], result[ytd_col] - prior_ytd
        )

    prior_allowance = by_bank["allowance_balance"].shift(1)
    pct_change = (result["allowance_balance"] - prior_allowance) / prior_allowance.replace(
        0, np.nan
    )
    result["allowance_merger_flag"] = (pct_change.abs() > 0.5).fillna(False)
    return result


def compute_bank_allowance_rollforward(
    allowance_panel: pd.DataFrame,
    category_panel: pd.DataFrame,
    categories: list[LoanCategory] | None = None,
    cecl_adoption_quarters: pd.Series | None = None,
) -> pd.DataFrame:
    """Joins `allowance_panel` (from `build_allowance_panel`) with
    `category_panel` (from `build_panel`) to compute the bank-quarter
    allowance roll-forward residual. The residual uses `allowance_panel`'s
    own bank-level total_chargeoff_quarterly/total_recovery_quarterly
    (RIADC079/RIAD4605 -- the actual Schedule RI-B Part II totals, which
    include several charge-off items, e.g. lease financing and farmland
    loans, this project doesn't map into any LoanCategory), NOT the sum of
    this project's mapped categories, which understates the true total by
    construction. That mapped-category sum is kept as a separate column
    (`mapped_category_net_chargeoffs`) -- see
    `summarize_mapped_category_coverage` for how much of the true total it
    captures, by quarter. `categories` (for the mapped-category sum only)
    defaults to every LoanCategory EXCEPT AUTO_AND_OTHER_CONSUMER_COMBINED
    (summing it alongside AUTO/OTHER_CONSUMER would double-count, per
    schema.py's module docstring) -- pass an explicit list to change that.

    `cecl_adoption_quarters` (e.g. from `derive_cecl_adoption_quarters`):
    a Series indexed by bank_id giving each bank's CECL adoption quarter.
    When given, `cecl_adoption_flag` is True for exactly the bank-quarter
    row that IS that bank's own adoption quarter -- a bank's one-time
    CECL transition adjustment to the allowance isn't organic provision/
    charge-off activity either, so (like a merger) it breaks the
    roll-forward identity for a known, different reason and should be
    excluded from summary statistics the same way (see
    `split_by_cecl_adoption_flag`). Omit it (or pass None) to get an
    all-False column.

    Returns bank_id, quarter, allowance_balance, beginning_allowance (the
    prior quarter's allowance_balance, per bank -- NaN for each bank's
    first quarter), provision_quarterly, total_net_chargeoffs (bank-level,
    used in the residual), mapped_category_net_chargeoffs (this project's
    categories, informational only), allowance_merger_flag,
    cecl_adoption_flag, and residual. A nonzero residual is expected
    against real data (a small "other adjustments" Call Report line isn't
    modeled here); large residuals for bank-quarters flagged by NEITHER
    allowance_merger_flag NOR cecl_adoption_flag point to a mapping or
    differencing error, not just real-world noise -- flagged bank-quarters
    break the identity for a known, different reason and should be
    excluded from summary statistics on the residual, not blamed on the
    mapping."""
    categories = (
        categories
        if categories is not None
        else [c for c in LoanCategory if c != LoanCategory.AUTO_AND_OTHER_CONSUMER_COMBINED]
    )
    scoped = category_panel[category_panel["category"].isin(categories)]
    net_chargeoffs = scoped.assign(
        net_chargeoff=scoped["chargeoff_quarterly"] - scoped["recovery_quarterly"]
    )
    mapped_category_net_chargeoffs = (
        net_chargeoffs.groupby(["bank_id", "quarter"], observed=True)["net_chargeoff"]
        .sum(min_count=1)
        .reset_index()
        .rename(columns={"net_chargeoff": "mapped_category_net_chargeoffs"})
    )

    merged = allowance_panel.merge(
        mapped_category_net_chargeoffs, on=["bank_id", "quarter"], how="left"
    )
    merged = merged.sort_values(["bank_id", "quarter"]).reset_index(drop=True)
    prior_allowance = merged.groupby("bank_id", sort=False)["allowance_balance"]
    merged["beginning_allowance"] = prior_allowance.shift(1)

    merged["total_net_chargeoffs"] = (
        merged["total_chargeoff_quarterly"] - merged["total_recovery_quarterly"]
    )
    if cecl_adoption_quarters is not None:
        expected_adoption_quarter = merged["bank_id"].map(cecl_adoption_quarters)
        merged["cecl_adoption_flag"] = (merged["quarter"] == expected_adoption_quarter).fillna(
            False
        )
    else:
        merged["cecl_adoption_flag"] = False
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
            "mapped_category_net_chargeoffs",
            "allowance_merger_flag",
            "cecl_adoption_flag",
            "residual",
        ]
    ]


def split_by_merger_flag(rollforward: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Returns (clean, flagged) -- `rollforward` (from
    `compute_bank_allowance_rollforward`) split by allowance_merger_flag.
    Summary statistics computed on `clean` reflect the roll-forward
    identity's actual fit; `flagged` rows are excluded from such summaries
    because a merger/acquisition-driven allowance jump breaks the identity
    by design (the acquired allowance arrives as a lump sum, not through
    organic provision/charge-off activity), not because of a mapping bug."""
    flagged_mask = rollforward["allowance_merger_flag"].fillna(False)
    return rollforward[~flagged_mask], rollforward[flagged_mask]


def split_by_cecl_adoption_flag(rollforward: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Returns (clean, flagged) -- `rollforward` split by
    cecl_adoption_flag (present when `compute_bank_allowance_rollforward`
    was given `cecl_adoption_quarters`). A bank's one-time CECL transition
    adjustment to the allowance (RIADJJ26/JJ28) isn't organic provision/
    charge-off activity either, so it breaks the roll-forward identity for
    the same structural reason a merger does -- exclude it from summary
    statistics the same way. The two flags are independent (rarely but not
    never both true for the same bank-quarter); apply both splits (this
    one and `split_by_merger_flag`) to get a "clean" set excluding either
    reason."""
    flagged_mask = rollforward["cecl_adoption_flag"].fillna(False)
    return rollforward[~flagged_mask], rollforward[flagged_mask]


def summarize_mapped_category_coverage(rollforward: pd.DataFrame) -> pd.DataFrame:
    """rollforward: from `compute_bank_allowance_rollforward`. Returns one
    row per quarter: what share of the bank-level TOTAL net charge-offs
    (total_net_chargeoffs, from RIADC079/RIAD4605) is captured by summing
    this project's mapped loan categories (mapped_category_net_chargeoffs)?
    Schedule RI-B has several charge-off items (lease financing, farmland
    loans, loans to foreign governments, and more) that aren't mapped into
    any LoanCategory -- this share is expected to be well under 100%, not
    a data-quality bug."""
    grouped = rollforward.groupby("quarter", observed=True)[
        ["mapped_category_net_chargeoffs", "total_net_chargeoffs"]
    ].sum(min_count=1)
    grouped["mapped_category_share"] = (
        grouped["mapped_category_net_chargeoffs"] / grouped["total_net_chargeoffs"]
    )
    return grouped.reset_index().sort_values("quarter").reset_index(drop=True)
