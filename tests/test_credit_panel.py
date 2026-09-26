"""Offline tests for the credit panel's pure transformation functions,
against synthetic fixtures only -- no network, no real Call Report data."""

import numpy as np
import pandas as pd
import pytest

from corefin.checks.framework import check_close_to_zero
from corefin.credit import panel
from corefin.credit.schema import CATEGORY_MDRM_CODES, LoanCategory


def _quarters(labels: list[str]) -> pd.Series:
    return pd.Series([pd.Period(label, freq="Q") for label in labels])


# ------------------------------------------------------------- YTD -> Q ----


def test_ytd_to_quarterly_q1_is_passthrough_and_later_quarters_are_diffed():
    ytd = pd.Series([10.0, 25.0, 45.0, 60.0])  # Q1..Q4 cumulative
    quarters = _quarters(["2019Q1", "2019Q2", "2019Q3", "2019Q4"])
    quarterly = panel.ytd_to_quarterly(ytd, quarters)
    assert quarterly.tolist() == [10.0, 15.0, 20.0, 15.0]


def test_ytd_to_quarterly_resets_at_new_calendar_year():
    ytd = pd.Series([10.0, 60.0, 12.0])  # 2019Q1, 2019Q4 YTD, then 2020Q1 YTD (new year)
    quarters = _quarters(["2019Q1", "2019Q4", "2020Q1"])
    quarterly = panel.ytd_to_quarterly(ytd, quarters)
    # 2020Q1 must be its own value (12.0), not diffed against 2019Q4's 60.0
    assert quarterly.tolist() == [10.0, 50.0, 12.0]


# --------------------------------------------------------- NCO rate --------


def test_annualized_nco_rate_correctness():
    # $1,000 average balance, $10 net quarterly NCO -> 1%/quarter -> 4%/year
    chargeoffs = pd.Series([15.0])
    recoveries = pd.Series([5.0])
    avg_balance = pd.Series([1000.0])
    rate = panel.annualized_nco_rate(chargeoffs, recoveries, avg_balance)
    assert rate.iloc[0] == pytest.approx(0.04)


def test_annualized_nco_rate_handles_zero_or_negative_balance_as_nan():
    chargeoffs = pd.Series([10.0, 10.0])
    recoveries = pd.Series([0.0, 0.0])
    avg_balance = pd.Series([0.0, -5.0])
    rate = panel.annualized_nco_rate(chargeoffs, recoveries, avg_balance)
    assert rate.isna().all()


def test_average_balance_first_observation_is_nan():
    avg = panel.average_balance(pd.Series([100.0, 200.0, 300.0]))
    assert avg.isna().iloc[0]
    assert avg.iloc[1] == 150.0
    assert avg.iloc[2] == 250.0


# ----------------------------------------------- category mapping totals ---


def test_category_mapping_sums_multi_item_categories_within_tolerance():
    (code_set,) = CATEGORY_MDRM_CODES[LoanCategory.RESIDENTIAL_MORTGAGE]
    first_lien, junior_lien = code_set.balance_items
    item_frame = pd.DataFrame({first_lien: [700.0], junior_lien: [130.0]})
    other_items = (
        *code_set.past_due_30_89_items,
        *code_set.past_due_90_items,
        *code_set.nonaccrual_items,
        *code_set.chargeoff_items,
        *code_set.recovery_items,
    )
    for col in other_items:
        item_frame[col] = 0.0
    quarters = _quarters(["2015Q1"])
    mapped = panel.apply_category_mapping(item_frame, LoanCategory.RESIDENTIAL_MORTGAGE, quarters)
    expected_total = 830.0
    assert mapped["balance"].iloc[0] == pytest.approx(expected_total, rel=1e-9)


def test_category_mapping_missing_component_is_nan_not_zero():
    (code_set,) = CATEGORY_MDRM_CODES[LoanCategory.OTHER_CONSUMER]
    item_frame = pd.DataFrame({col: [500.0] for col in code_set.balance_items})
    for col in (*code_set.chargeoff_items, *code_set.recovery_items):
        item_frame[col] = 1.0
    quarters = _quarters(["2015Q1"])
    mapped = panel.apply_category_mapping(item_frame, LoanCategory.OTHER_CONSUMER, quarters)
    # other_consumer has no past-due/nonaccrual item codes in schema.py -- must be NaN
    assert mapped["past_due_30_89"].isna().all()
    assert mapped["nonaccrual"].isna().all()


def test_category_mapping_raises_on_missing_required_column():
    quarters = _quarters(["2015Q1"])
    with pytest.raises(KeyError):
        panel.apply_category_mapping(pd.DataFrame({"RCON9999": [1.0]}), LoanCategory.CI, quarters)


def test_category_mapping_raises_for_a_quarter_with_no_code_set():
    # auto loans have no code set before 2011Q1 -- this must raise, not
    # silently return zeros or NaN.
    (code_set,) = CATEGORY_MDRM_CODES[LoanCategory.AUTO]
    item_frame = pd.DataFrame({col: [500.0] for col in code_set.balance_items})
    other_items = (
        *code_set.past_due_30_89_items,
        *code_set.past_due_90_items,
        *code_set.nonaccrual_items,
        *code_set.chargeoff_items,
        *code_set.recovery_items,
    )
    for col in other_items:
        item_frame[col] = 0.0
    quarters = _quarters(["2005Q1"])
    with pytest.raises(ValueError, match="no MDRM code set covers"):
        panel.apply_category_mapping(item_frame, LoanCategory.AUTO, quarters)


def test_category_mapping_picks_the_right_code_set_on_each_side_of_a_switch_date():
    # cre_nonfarm_nonresidential switches from RCON1480 to RCONF160+RCONF161 at 2007Q1.
    item_frame = pd.DataFrame(
        {
            "RCON1480": [1000.0, np.nan],
            "RCONF160": [np.nan, 600.0],
            "RCONF161": [np.nan, 450.0],
            "RCON3502": [0.0, np.nan],
            "RCON3503": [0.0, np.nan],
            "RCON3504": [0.0, np.nan],
            "RIAD3590": [0.0, np.nan],
            "RIAD3591": [0.0, np.nan],
            "RCONF178": [np.nan, 0.0],
            "RCONF179": [np.nan, 0.0],
            "RCONF180": [np.nan, 0.0],
            "RCONF181": [np.nan, 0.0],
            "RCONF182": [np.nan, 0.0],
            "RCONF183": [np.nan, 0.0],
        }
    )
    quarters = _quarters(["2006Q4", "2007Q1"])
    category = LoanCategory.CRE_NONFARM_NONRESIDENTIAL
    mapped = panel.apply_category_mapping(item_frame, category, quarters)
    assert mapped["balance"].iloc[0] == pytest.approx(1000.0)
    assert mapped["balance"].iloc[1] == pytest.approx(1050.0)


def test_aggregate_balance_is_continuous_across_a_synthetic_code_switch():
    # Two banks reporting the SAME true total nonfarm-nonresidential balance
    # on both sides of the 2007Q1 code switch -- the mapped aggregate must
    # not show a spurious break just because the underlying item changed.
    item_frame = pd.DataFrame(
        {
            "RCON1480": [1000.0, 2000.0, np.nan, np.nan],
            "RCONF160": [np.nan, np.nan, 600.0, 1200.0],
            "RCONF161": [np.nan, np.nan, 400.0, 800.0],
            "RCON3502": [0.0, 0.0, np.nan, np.nan],
            "RCON3503": [0.0, 0.0, np.nan, np.nan],
            "RCON3504": [0.0, 0.0, np.nan, np.nan],
            "RIAD3590": [0.0, 0.0, np.nan, np.nan],
            "RIAD3591": [0.0, 0.0, np.nan, np.nan],
            "RCONF178": [np.nan, np.nan, 0.0, 0.0],
            "RCONF179": [np.nan, np.nan, 0.0, 0.0],
            "RCONF180": [np.nan, np.nan, 0.0, 0.0],
            "RCONF181": [np.nan, np.nan, 0.0, 0.0],
            "RCONF182": [np.nan, np.nan, 0.0, 0.0],
            "RCONF183": [np.nan, np.nan, 0.0, 0.0],
        }
    )
    quarters = _quarters(["2006Q4", "2006Q4", "2007Q1", "2007Q1"])
    category = LoanCategory.CRE_NONFARM_NONRESIDENTIAL
    mapped = panel.apply_category_mapping(item_frame, category, quarters)
    aggregate_before = mapped["balance"].iloc[:2].sum()
    aggregate_after = mapped["balance"].iloc[2:].sum()
    assert aggregate_before == pytest.approx(3000.0)
    assert aggregate_after == pytest.approx(3000.0)


# --------------------------------------------------------- RCFD fallback ---


def test_rcfd_fallback_only_applies_to_ffiec_031_with_missing_rcon():
    rcon = pd.Series([100.0, np.nan, np.nan])
    rcfd = pd.Series([999.0, 999.0, 999.0])
    reporting_form = pd.Series(["FFIEC 031", "FFIEC 031", "FFIEC 041"])
    resolved = panel.apply_rcfd_fallback(rcon, rcfd, reporting_form)
    # row 0: RCON present -> keep RCON. row 1: FFIEC031 + missing RCON -> use RCFD.
    # row 2: FFIEC041 + missing RCON -> stays missing (no RCFD equivalent exists).
    assert resolved.iloc[0] == pytest.approx(100.0)
    assert resolved.iloc[1] == pytest.approx(999.0)
    assert pd.isna(resolved.iloc[2])


# ------------------------------------------------------------ mergers ------


def test_flag_merger_discontinuities_flags_large_jump_only():
    balance = pd.Series([100.0, 105.0, 400.0, 410.0])  # 3rd quarter ~4x jump
    flags = panel.flag_merger_discontinuities(balance, jump_threshold=0.5)
    assert flags.tolist() == [False, False, True, False]


# --------------------------------------------------------------- CECL ------


def test_cecl_regime_dummy_flips_at_adoption_quarter():
    quarters = _quarters(["2019Q4", "2020Q1", "2020Q2"])
    adoption_quarter = pd.Period("2020Q1", freq="Q")
    dummy = panel.apply_cecl_regime_dummy(quarters, cecl_adoption_quarter=adoption_quarter)
    assert dummy.tolist() == [0, 1, 1]


# --------------------------------------------------------- min balance -----


def test_min_balance_filter_excludes_small_and_nan_balances():
    balance = pd.Series([1_000_000.0, 500.0, np.nan])
    keep = panel.apply_min_balance_filter(balance, minimum=1000.0)
    assert keep.tolist() == [True, False, False]


# --------------------------------------------------- allowance roll-forward -


def test_allowance_rollforward_residual_is_zero_for_a_consistent_synthetic_series():
    beginning = pd.Series([100.0, 110.0, 108.0])
    provision = pd.Series([20.0, 15.0, 25.0])
    nco = pd.Series([10.0, 17.0, 13.0])
    ending = beginning + provision - nco
    residual = panel.allowance_rollforward_residual(beginning, provision, nco, ending)
    result = check_close_to_zero(
        "allowance_rollforward", residual.to_numpy().reshape(1, -1), tolerance=1e-9
    )
    assert result.passed


def test_allowance_rollforward_residual_detects_a_broken_identity():
    residual = panel.allowance_rollforward_residual(
        pd.Series([100.0]), pd.Series([20.0]), pd.Series([10.0]), pd.Series([999.0])
    )
    assert residual.iloc[0] == pytest.approx(999.0 - 110.0)


# --------------------------------------------------------- coverage report -


def test_coverage_report_computes_share_and_aggregate_per_category_quarter():
    long_panel = pd.DataFrame(
        {
            "category": [LoanCategory.CI, LoanCategory.CI, LoanCategory.CI, LoanCategory.CI],
            "quarter": ["2019Q1", "2019Q1", "2019Q2", "2019Q2"],
            "balance": [100.0, np.nan, 150.0, 250.0],
        }
    )
    report = panel.build_coverage_report(long_panel)
    q1 = report[report["quarter"] == pd.Period("2019Q1", freq="Q")].iloc[0]
    q2 = report[report["quarter"] == pd.Period("2019Q2", freq="Q")].iloc[0]
    assert q1["n_banks"] == 2
    assert q1["coverage_share"] == pytest.approx(0.5)
    assert q1["aggregate_balance"] == pytest.approx(100.0)
    assert q2["coverage_share"] == pytest.approx(1.0)
    assert q2["aggregate_balance"] == pytest.approx(400.0)
    assert q2["aggregate_pct_change"] == pytest.approx((400.0 - 100.0) / 100.0)


def test_coverage_report_flags_a_break_at_a_code_switch_quarter():
    # cre_nonfarm_nonresidential switches at 2007Q1 -- a large jump exactly
    # there should be flagged; the same-size jump one quarter later (not a
    # switch boundary) should not be.
    long_panel = pd.DataFrame(
        {
            "category": [LoanCategory.CRE_NONFARM_NONRESIDENTIAL] * 3,
            "quarter": ["2006Q4", "2007Q1", "2007Q2"],
            "balance": [1000.0, 2000.0, 2000.0],
        }
    )
    report = panel.build_coverage_report(long_panel).set_index("quarter")
    assert bool(report.loc[pd.Period("2007Q1", freq="Q"), "code_switch"])
    assert bool(report.loc[pd.Period("2007Q1", freq="Q"), "possible_break"])
    assert not bool(report.loc[pd.Period("2007Q2", freq="Q"), "code_switch"])
    assert not bool(report.loc[pd.Period("2007Q2", freq="Q"), "possible_break"])
