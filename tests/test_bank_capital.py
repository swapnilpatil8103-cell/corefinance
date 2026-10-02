"""Offline tests for corefin.bank.capital -- synthetic data only. Covers
the required Stage 2 check: jump-off CET1 ratio matches the bank's own
reported ratio within tolerance, after RWA calibration."""

from __future__ import annotations

import numpy as np
import pytest

from corefin.bank.balance_sheet import project_balance_sheet
from corefin.bank.capital import compute_capital
from corefin.bank.schema import BankConfig, BankOpeningBalance
from corefin.credit.interface import CreditLossProjection
from corefin.timeline import Timeline

_CATEGORIES = ("commercial_and_industrial", "residential_mortgage", "home_equity")


def _timeline(n_periods: int = 9) -> Timeline:
    return Timeline.quarterly(n_periods=n_periods, n_historical=1, start_year=2025, start_quarter=4)


def _synthetic_credit_projection(
    timeline: Timeline, bank_id: str = "99999"
) -> CreditLossProjection:
    n_cat = len(_CATEGORIES)
    n = timeline.n_periods
    return CreditLossProjection(
        timeline=timeline,
        categories=_CATEGORIES,
        scenario_name="baseline",
        bank_identifier=bank_id,
        balance_mm=np.full((n_cat, n), 1000.0),
        net_charge_off_mm=np.full((n_cat, n), 2.0),
        provision_expense_mm=np.full((n_cat, n), 2.0),
        allowance_mm=np.full((n_cat, n), 20.0),
        npl_mm=np.full((n_cat, n), 10.0),
        nco_rate=np.full((n_cat, n), 0.002),
        npl_ratio=np.full((n_cat, n), 0.01),
        monte_carlo_mean=np.zeros(n_cat),
        monte_carlo_percentiles={},
    )


def _synthetic_opening(**overrides) -> BankOpeningBalance:
    kwargs = dict(
        name="Test Bank",
        bank_id="99999",
        hc_rssd_id="88888",
        cash_mm=500.0,
        securities_afs_mm=300.0,
        securities_htm_mm=200.0,
        other_assets_mm=100.0,
        goodwill_mm=50.0,
        other_intangibles_mm=10.0,
        deposits_mm=3000.0,
        borrowings_mm=200.0,
        other_liabilities_mm=50.0,
        equity_mm=850.0,
        net_interest_income_jumpoff_mm=40.0,
        noninterest_income_jumpoff_mm=10.0,
        noninterest_expense_jumpoff_mm=30.0,
        reported_cet1_capital_mm=790.0,
        reported_cet1_ratio=0.12,
        reported_rwa_mm=6583.33,
        reported_tier1_leverage_ratio=0.09,
    )
    kwargs.update(overrides)
    return BankOpeningBalance(**kwargs)


def _build_balance_sheet(opening, timeline, config, net_income_mm=None, dividends_mm=None):
    credit_projection = _synthetic_credit_projection(timeline)
    n = timeline.n_periods
    if net_income_mm is None:
        net_income_mm = np.full(n, 5.0)
    if dividends_mm is None:
        dividends_mm = np.zeros(n)
    return project_balance_sheet(
        opening, credit_projection, net_income_mm, dividends_mm, config, timeline
    )


def test_jumpoff_cet1_ratio_matches_reported_within_tolerance():
    timeline = _timeline()
    opening = _synthetic_opening()
    config = BankConfig()
    bs = _build_balance_sheet(opening, timeline, config)

    capital = compute_capital(bs, opening, config, timeline)

    assert capital.cet1_ratio[0] == pytest.approx(opening.reported_cet1_ratio, abs=1e-6)


def test_jumpoff_cet1_capital_matches_reported_exactly():
    timeline = _timeline()
    opening = _synthetic_opening()
    config = BankConfig()
    bs = _build_balance_sheet(opening, timeline, config)

    capital = compute_capital(bs, opening, config, timeline)

    assert capital.cet1_capital_mm[0] == pytest.approx(opening.reported_cet1_capital_mm, abs=1e-6)


def test_jumpoff_rwa_matches_reported_exactly():
    timeline = _timeline()
    opening = _synthetic_opening()
    config = BankConfig()
    bs = _build_balance_sheet(opening, timeline, config)

    capital = compute_capital(bs, opening, config, timeline)

    assert capital.rwa_mm[0] == pytest.approx(opening.reported_rwa_mm, abs=1e-6)


def test_rwa_calibration_factor_is_positive_and_finite():
    timeline = _timeline()
    opening = _synthetic_opening()
    config = BankConfig()
    bs = _build_balance_sheet(opening, timeline, config)

    capital = compute_capital(bs, opening, config, timeline)
    assert capital.rwa_calibration_factor > 0
    assert np.isfinite(capital.rwa_calibration_factor)


def test_an_increase_in_goodwill_after_jumpoff_reduces_cet1_capital_dollar_for_dollar():
    # Two banks identical except one's goodwill steps up by 100 starting period 1 (e.g. what a
    # purchase-accounting goodwill addition would do, Stage 3's concern) -- both are calibrated
    # to the SAME reported jump-off CET1 capital, so a static jump-off goodwill LEVEL alone
    # doesn't move the projected path (the calibration constant absorbs it), but a CHANGE in
    # goodwill after jump-off must reduce CET1 capital exactly dollar-for-dollar.
    timeline = _timeline()
    opening = _synthetic_opening()
    config = BankConfig()
    bs = _build_balance_sheet(opening, timeline, config)
    capital = compute_capital(bs, opening, config, timeline)

    from dataclasses import replace

    goodwill_step_up_mm = 100.0
    bumped_goodwill = bs.goodwill_mm.copy()
    bumped_goodwill[1:] += goodwill_step_up_mm
    bs_with_goodwill_step_up = replace(bs, goodwill_mm=bumped_goodwill)
    capital_with_step_up = compute_capital(bs_with_goodwill_step_up, opening, config, timeline)

    assert capital_with_step_up.cet1_capital_mm[0] == pytest.approx(capital.cet1_capital_mm[0])
    assert np.allclose(
        capital.cet1_capital_mm[1:] - capital_with_step_up.cet1_capital_mm[1:],
        goodwill_step_up_mm,
    )


def test_capital_result_rejects_wrong_shape_array():
    from corefin.bank.capital import CapitalResult

    timeline = _timeline(n_periods=4)
    n = timeline.n_periods
    with pytest.raises(ValueError, match="rwa_mm"):
        CapitalResult(
            timeline=timeline,
            cet1_capital_mm=np.zeros(n),
            rwa_mm=np.zeros(n + 1),
            average_assets_mm=np.zeros(n),
            rwa_calibration_factor=1.0,
            other_cet1_adjustments_mm=0.0,
        )
