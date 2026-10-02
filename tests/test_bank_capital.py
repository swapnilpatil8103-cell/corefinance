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
        goodwill_net_of_dtl_mm=50.0,
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


def test_higher_goodwill_net_of_dtl_deduction_widens_the_unexplained_residual_by_the_same_amount():
    # Holding reported_cet1_capital_mm fixed, a larger goodwill_net_of_dtl_mm CET1 deduction
    # makes the explicit bridge's OWN (pre-residual) formula capital lower by the same amount --
    # both openings still tie to the same reported jump-off CET1 capital by construction (the
    # residual plug absorbs the difference), but the residual itself must widen by exactly the
    # goodwill delta, proving goodwill_net_of_dtl_mm is actually wired into the bridge formula.
    timeline = _timeline()
    goodwill_delta_mm = 100.0
    low_goodwill = _synthetic_opening(goodwill_net_of_dtl_mm=50.0)
    high_goodwill = _synthetic_opening(goodwill_net_of_dtl_mm=50.0 + goodwill_delta_mm)
    config = BankConfig()

    bs_low = _build_balance_sheet(low_goodwill, timeline, config)
    bs_high = _build_balance_sheet(high_goodwill, timeline, config)
    capital_low = compute_capital(bs_low, low_goodwill, config, timeline)
    capital_high = compute_capital(bs_high, high_goodwill, config, timeline)

    assert capital_low.cet1_capital_mm[0] == pytest.approx(low_goodwill.reported_cet1_capital_mm)
    assert capital_high.cet1_capital_mm[0] == pytest.approx(high_goodwill.reported_cet1_capital_mm)
    assert (
        capital_high.unexplained_cet1_residual_mm - capital_low.unexplained_cet1_residual_mm
    ) == pytest.approx(goodwill_delta_mm)


def test_preferred_stock_reduces_cet1_before_adjustments():
    # Preferred stock is part of total equity but excluded from CET1 (Additional Tier 1
    # instead) -- holding reported_cet1_capital_mm fixed, adding preferred stock should widen
    # the unexplained residual by exactly the preferred stock amount, same mechanism as the
    # goodwill test above.
    timeline = _timeline()
    preferred_mm = 80.0
    no_preferred = _synthetic_opening(preferred_stock_mm=0.0)
    with_preferred = _synthetic_opening(preferred_stock_mm=preferred_mm)
    config = BankConfig()

    bs_no_preferred = _build_balance_sheet(no_preferred, timeline, config)
    bs_with_preferred = _build_balance_sheet(with_preferred, timeline, config)
    capital_no_preferred = compute_capital(bs_no_preferred, no_preferred, config, timeline)
    capital_with_preferred = compute_capital(bs_with_preferred, with_preferred, config, timeline)

    assert (
        capital_with_preferred.unexplained_cet1_residual_mm
        - capital_no_preferred.unexplained_cet1_residual_mm
    ) == pytest.approx(preferred_mm)


def test_aoci_afs_loss_is_added_back_not_subtracted():
    # A loss is stored as a NEGATIVE value (RC-R Part I convention); subtracting a negative
    # ADDS it back to CET1 -- the AOCI opt-out mechanism (see schema.py's CET1 bridge
    # docstring). Holding reported_cet1_capital_mm fixed, a more negative (larger loss) AOCI
    # figure should NARROW the unexplained residual (the formula capital is now higher, closer
    # to or past the reported figure), not widen it.
    timeline = _timeline()
    small_loss = _synthetic_opening(aoci_afs_unrealized_mm=-10.0)
    large_loss = _synthetic_opening(aoci_afs_unrealized_mm=-50.0)
    config = BankConfig()

    bs_small = _build_balance_sheet(small_loss, timeline, config)
    bs_large = _build_balance_sheet(large_loss, timeline, config)
    capital_small = compute_capital(bs_small, small_loss, config, timeline)
    capital_large = compute_capital(bs_large, large_loss, config, timeline)

    assert (
        capital_large.unexplained_cet1_residual_mm - capital_small.unexplained_cet1_residual_mm
    ) == pytest.approx(-40.0)


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
            unexplained_cet1_residual_mm=0.0,
        )
