"""Offline tests for corefin.bank.allowance -- synthetic data only.
Covers the required Stage 4 check: a bank with steady losses anchored to
a reported allowance that differs from the model's own raw jump-off
estimate shows NO jump-off provision spike."""

from __future__ import annotations

import numpy as np
import pytest

from corefin.bank.allowance import anchor_allowance_to_reported
from corefin.bank.model import run_bank_model
from corefin.bank.schema import BankConfig, BankOpeningBalance
from corefin.credit.interface import CreditLossProjection
from corefin.timeline import Timeline

_TIMELINE = Timeline.quarterly(5, n_historical=1, start_year=2025, start_quarter=4)
_CATEGORIES = ("commercial_and_industrial", "residential_mortgage")


def _projection(allowance_by_category: list[list[float]], nco_by_category: list[list[float]]):
    n_cat = len(_CATEGORIES)
    n = _TIMELINE.n_periods
    allowance = np.array(allowance_by_category)
    net_charge_off = np.array(nco_by_category)
    provision = np.full((n_cat, n), np.nan)
    provision[:, 1:] = (allowance[:, 1:] - allowance[:, :-1]) + net_charge_off[:, 1:]
    return CreditLossProjection(
        timeline=_TIMELINE,
        categories=_CATEGORIES,
        scenario_name="baseline",
        bank_identifier="111",
        balance_mm=np.full((n_cat, n), 1000.0),
        net_charge_off_mm=net_charge_off,
        provision_expense_mm=provision,
        allowance_mm=allowance,
        npl_mm=np.full((n_cat, n), 10.0),
        nco_rate=np.full((n_cat, n), 0.01),
        npl_ratio=np.full((n_cat, n), 0.01),
        monte_carlo_mean=np.zeros(n_cat),
        monte_carlo_percentiles={},
    )


def test_anchoring_sets_jumpoff_allowance_to_the_reported_figure_exactly():
    # model's own raw jump-off estimate (100) disagrees with the bank's real reported
    # allowance (150) -- the exact real-world mismatch this module fixes
    projection = _projection(
        allowance_by_category=[[60.0, 70.0, 72.0, 71.0, 70.0], [40.0, 50.0, 52.0, 51.0, 50.0]],
        nco_by_category=[[np.nan, 5.0, 5.0, 5.0, 5.0], [np.nan, 3.0, 3.0, 3.0, 3.0]],
    )
    anchored = anchor_allowance_to_reported(projection, reported_allowance_mm=150.0)
    assert anchored.allowance_total_mm[0] == pytest.approx(150.0)


def test_anchoring_preserves_the_models_own_relative_path():
    projection = _projection(
        allowance_by_category=[[60.0, 70.0, 72.0, 71.0, 70.0], [40.0, 50.0, 52.0, 51.0, 50.0]],
        nco_by_category=[[np.nan, 5.0, 5.0, 5.0, 5.0], [np.nan, 3.0, 3.0, 3.0, 3.0]],
    )
    anchored = anchor_allowance_to_reported(projection, reported_allowance_mm=150.0)
    model_ratio = projection.allowance_total_mm / projection.allowance_total_mm[0]
    anchored_ratio = anchored.allowance_total_mm / anchored.allowance_total_mm[0]
    assert np.allclose(anchored_ratio, model_ratio)


def test_anchoring_preserves_the_models_own_category_mix():
    projection = _projection(
        allowance_by_category=[[60.0, 70.0, 72.0, 71.0, 70.0], [40.0, 50.0, 52.0, 51.0, 50.0]],
        nco_by_category=[[np.nan, 5.0, 5.0, 5.0, 5.0], [np.nan, 3.0, 3.0, 3.0, 3.0]],
    )
    anchored = anchor_allowance_to_reported(projection, reported_allowance_mm=150.0)
    model_share = projection.allowance_mm / projection.allowance_total_mm
    anchored_share = anchored.allowance_mm / anchored.allowance_total_mm
    assert np.allclose(anchored_share, model_share)


def test_steady_losses_anchored_to_a_different_reported_level_has_no_provision_spike():
    # the model's OWN path is flat (steady losses, both jump-off and every projected quarter
    # at the SAME level per category) -- anchoring to a reported figure that's quite different
    # from the model's own raw level (150 vs 100) must NOT create a spike: the allowance
    # roll-forward's delta term stays zero in every period, since the model's own relative
    # path never moves.
    projection = _projection(
        allowance_by_category=[[60.0, 60.0, 60.0, 60.0, 60.0], [40.0, 40.0, 40.0, 40.0, 40.0]],
        nco_by_category=[[np.nan, 5.0, 5.0, 5.0, 5.0], [np.nan, 3.0, 3.0, 3.0, 3.0]],
    )
    anchored = anchor_allowance_to_reported(projection, reported_allowance_mm=150.0)

    assert anchored.allowance_total_mm[0] == pytest.approx(150.0)
    assert np.allclose(anchored.allowance_total_mm, 150.0)  # flat at every period
    # provision = net charge-off alone, no allowance-build spike anywhere
    assert np.allclose(anchored.provision_expense_mm[:, 1:], projection.net_charge_off_mm[:, 1:])


def test_anchoring_raises_when_model_jumpoff_allowance_is_non_positive():
    projection = _projection(
        allowance_by_category=[[0.0, 70.0, 72.0, 71.0, 70.0], [0.0, 50.0, 52.0, 51.0, 50.0]],
        nco_by_category=[[np.nan, 5.0, 5.0, 5.0, 5.0], [np.nan, 3.0, 3.0, 3.0, 3.0]],
    )
    with pytest.raises(ValueError):
        anchor_allowance_to_reported(projection, reported_allowance_mm=150.0)


def _opening(reported_allowance_mm: float | None) -> BankOpeningBalance:
    return BankOpeningBalance(
        name="Acquirer Bank A",
        bank_id="111",
        hc_rssd_id="222",
        cash_mm=200.0,
        securities_afs_mm=100.0,
        securities_htm_mm=100.0,
        other_assets_mm=50.0,
        goodwill_mm=20.0,
        other_intangibles_mm=5.0,
        deposits_mm=1200.0,
        borrowings_mm=50.0,
        other_liabilities_mm=25.0,
        equity_mm=200.0,
        net_interest_income_jumpoff_mm=20.0,
        noninterest_income_jumpoff_mm=5.0,
        noninterest_expense_jumpoff_mm=15.0,
        reported_allowance_mm=reported_allowance_mm,
        goodwill_net_of_dtl_mm=20.0,
        other_intangibles_net_of_dtl_mm=3.5,
        dta_nol_deduction_mm=0.5,
        aoci_afs_unrealized_mm=-2.0,
        reported_cet1_capital_mm=200.0 - 20.0 - 3.5 - 0.5 + 2.0,
        reported_cet1_ratio=0.12,
        reported_rwa_mm=1000.0,
        reported_tier1_leverage_ratio=0.09,
    )


def test_run_bank_model_anchors_when_reported_allowance_is_set():
    projection = _projection(
        allowance_by_category=[[60.0, 70.0, 72.0, 71.0, 70.0], [40.0, 50.0, 52.0, 51.0, 50.0]],
        nco_by_category=[[np.nan, 5.0, 5.0, 5.0, 5.0], [np.nan, 3.0, 3.0, 3.0, 3.0]],
    )
    result = run_bank_model(_opening(150.0), projection, BankConfig(), _TIMELINE)
    assert result.balance_sheet.total_allowance_mm[0] == pytest.approx(150.0)


def test_run_bank_model_leaves_allowance_unanchored_when_reported_allowance_is_none():
    projection = _projection(
        allowance_by_category=[[60.0, 70.0, 72.0, 71.0, 70.0], [40.0, 50.0, 52.0, 51.0, 50.0]],
        nco_by_category=[[np.nan, 5.0, 5.0, 5.0, 5.0], [np.nan, 3.0, 3.0, 3.0, 3.0]],
    )
    result = run_bank_model(_opening(None), projection, BankConfig(), _TIMELINE)
    assert result.balance_sheet.total_allowance_mm[0] == pytest.approx(100.0)  # model's own raw
