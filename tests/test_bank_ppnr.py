"""Offline tests for corefin.bank.ppnr -- synthetic data only."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from corefin.bank.ppnr import (
    align_rate_path_to_timeline,
    calibrate_nim_beta,
    compute_stressed_nii_mm,
    compute_stressed_noninterest_income_mm,
)
from corefin.bank.schema import NET_INTEREST_INCOME_ITEMS, TOTAL_ASSETS_ITEM

_INCOME_ITEM, _EXPENSE_ITEM = NET_INTEREST_INCOME_ITEMS


def _call_report_row(income_ytd: float, expense_ytd: float, total_assets: float) -> pd.Series:
    return pd.Series(
        {_INCOME_ITEM: income_ytd, _EXPENSE_ITEM: expense_ytd, TOTAL_ASSETS_ITEM: total_assets}
    )


def _quarters(start: str, n: int) -> list[pd.Period]:
    first = pd.Period(start, freq="Q")
    return [first + i for i in range(n)]


def test_calibrate_nim_beta_matches_hand_computed_value():
    # Flat $40mm quarterly NII (YTD cumulates 40, 80, 120, 160 within each year) on a
    # constant $10,000mm total assets at 2020Q1 (NIM = 40*4/10000 = 1.6%); NII steps down to
    # a flat $25mm/quarter from 2021Q1 onward (NIM = 25*4/10000 = 1.0%) -- a realized NIM
    # decline of -0.6pp while the 3-month Treasury rate falls 1.0pp (1.1 -> 0.1), an
    # asset-sensitive bank's expected sign (NIM falls when rates fall).
    quarters_2020 = _quarters("2020Q1", 4)
    quarters_2021 = _quarters("2021Q1", 4)
    rows = {}
    for i, q in enumerate(quarters_2020):
        rows[q] = _call_report_row(
            income_ytd=40.0 * (i + 1), expense_ytd=0.0, total_assets=10_000.0
        )
    for i, q in enumerate(quarters_2021):
        rows[q] = _call_report_row(
            income_ytd=25.0 * (i + 1), expense_ytd=0.0, total_assets=10_000.0
        )

    rate_by_quarter = pd.Series(
        {pd.Period("2020Q1", freq="Q"): 1.1, pd.Period("2021Q4", freq="Q"): 0.1}
    )
    result = calibrate_nim_beta("111", rows, rate_by_quarter)

    assert result is not None
    assert result.start_nim == pytest.approx(0.016)
    assert result.end_nim == pytest.approx(0.010)
    assert result.realized_nim_change_pp == pytest.approx(-0.6)
    assert result.realized_rate_change_pp == pytest.approx(-1.0)
    assert result.nim_beta == pytest.approx(0.6)  # asset-sensitive: positive, same direction


def test_calibrate_nim_beta_returns_none_when_window_incomplete():
    quarters_2020 = _quarters("2020Q1", 4)
    rows = {
        q: _call_report_row(40.0 * (i + 1), 0.0, 10_000.0) for i, q in enumerate(quarters_2020)
    }  # missing all of 2021
    rate_by_quarter = pd.Series(
        {pd.Period("2020Q1", freq="Q"): 1.1, pd.Period("2021Q4", freq="Q"): 0.1}
    )
    assert calibrate_nim_beta("111", rows, rate_by_quarter) is None


def test_calibrate_nim_beta_returns_none_when_rate_barely_moves():
    quarters_2020 = _quarters("2020Q1", 4)
    quarters_2021 = _quarters("2021Q1", 4)
    rows = {}
    for i, q in enumerate(quarters_2020):
        rows[q] = _call_report_row(40.0 * (i + 1), 0.0, 10_000.0)
    for i, q in enumerate(quarters_2021):
        rows[q] = _call_report_row(25.0 * (i + 1), 0.0, 10_000.0)
    rate_by_quarter = pd.Series(
        {pd.Period("2020Q1", freq="Q"): 1.1, pd.Period("2021Q4", freq="Q"): 1.1}
    )
    result = calibrate_nim_beta("111", rows, rate_by_quarter)
    assert result is not None
    assert result.nim_beta is None


def test_align_rate_path_to_timeline_prepends_jumpoff_and_pads_flat():
    path = align_rate_path_to_timeline(3.7, np.array([2.5, 0.1, 0.1]), n_periods=6)
    assert path[0] == pytest.approx(3.7)
    assert np.allclose(path[1:4], [2.5, 0.1, 0.1])
    assert np.allclose(path[4:], [0.1, 0.1])  # flat continuation at the last value


def test_align_rate_path_to_timeline_truncates():
    path = align_rate_path_to_timeline(3.7, np.array([2.5, 0.1, 0.1, 0.1]), n_periods=3)
    assert len(path) == 3
    assert np.allclose(path, [3.7, 2.5, 0.1])


def test_compute_stressed_nii_mm_matches_formula():
    rate_path = np.array([3.7, 0.1, 0.1])
    nii = compute_stressed_nii_mm(
        nii_jumpoff_mm=100.0,
        earning_assets_jumpoff_mm=10_000.0,
        rate_path_pp=rate_path,
        nim_beta=0.3,
    )
    assert nii[0] == pytest.approx(100.0)
    rate_change_pp = -3.6
    expected_delta = (0.3 * rate_change_pp / 100.0) * 10_000.0 / 4.0
    assert nii[1] == pytest.approx(100.0 + expected_delta)
    assert nii[2] == pytest.approx(100.0 + expected_delta)


def test_compute_stressed_nii_mm_is_asset_sensitive_when_beta_positive():
    # positive beta, rate FALLS -- NII should FALL too (asset-sensitive bank loses margin)
    rate_path = np.array([3.7, 0.1])
    nii = compute_stressed_nii_mm(100.0, 10_000.0, rate_path, nim_beta=0.3)
    assert nii[1] < nii[0]


def test_compute_stressed_noninterest_income_mm_steps_down_from_period_1():
    path = compute_stressed_noninterest_income_mm(
        noninterest_income_jumpoff_mm=20.0, n_periods=4, decline_pct=0.10
    )
    assert path[0] == pytest.approx(20.0)
    assert np.allclose(path[1:], 18.0)
