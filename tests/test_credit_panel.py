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
    codes = CATEGORY_MDRM_CODES[LoanCategory.RESIDENTIAL_MORTGAGE]
    first_lien, junior_lien = codes.balance_items
    item_frame = pd.DataFrame({first_lien: [700.0], junior_lien: [130.0]})
    # fill the rest of the required columns with zeros/empty so mapping doesn't KeyError
    other_items = (
        *codes.past_due_30_89_items,
        *codes.past_due_90_items,
        *codes.chargeoff_items,
        *codes.recovery_items,
    )
    for col in other_items:
        item_frame[col] = 0.0
    mapped = panel.apply_category_mapping(item_frame, LoanCategory.RESIDENTIAL_MORTGAGE)
    expected_total = 830.0
    assert mapped["balance"].iloc[0] == pytest.approx(expected_total, rel=1e-9)


def test_category_mapping_missing_component_is_nan_not_zero():
    codes = CATEGORY_MDRM_CODES[LoanCategory.OTHER_CONSUMER]
    (balance_item,) = codes.balance_items
    item_frame = pd.DataFrame({balance_item: [500.0]})
    for col in (*codes.chargeoff_items, *codes.recovery_items):
        item_frame[col] = 1.0
    mapped = panel.apply_category_mapping(item_frame, LoanCategory.OTHER_CONSUMER)
    # other_consumer has no past-due/nonaccrual item codes in schema.py -- must be NaN
    assert mapped["past_due_30_89"].isna().all()
    assert mapped["nonaccrual"].isna().all()


def test_category_mapping_raises_on_missing_required_column():
    with pytest.raises(KeyError):
        panel.apply_category_mapping(pd.DataFrame({"RCON9999": [1.0]}), LoanCategory.CI)


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
