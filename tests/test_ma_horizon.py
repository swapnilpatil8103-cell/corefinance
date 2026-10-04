"""Offline tests for corefin.ma.horizon -- synthetic data only. Covers
the required Stage 4 check: credit losses extended beyond a scenario's
own horizon follow the documented flat-rate/flat-allowance rule, and the
convenience wrapper rebuilds a BankModelResult over the extended
horizon."""

from __future__ import annotations

import numpy as np
import pytest

from corefin.bank.schema import BankConfig, BankOpeningBalance
from corefin.credit.interface import CreditLossProjection
from corefin.ma.horizon import (
    build_deal_horizon_bank_result,
    build_stress_bank_result,
    extend_credit_loss_projection,
    truncate_credit_loss_projection,
)
from corefin.timeline import Timeline

_TIMELINE = Timeline.quarterly(5, n_historical=1, start_year=2025, start_quarter=4)
_CATEGORIES = ("commercial_and_industrial", "residential_mortgage")


def _projection(nco_rate_path, allowance_path) -> CreditLossProjection:
    n_cat = len(_CATEGORIES)
    n = _TIMELINE.n_periods
    balance = 1000.0
    nco_rate = np.tile(np.array(nco_rate_path), (n_cat, 1))
    allowance = np.tile(np.array(allowance_path), (n_cat, 1))
    net_charge_off = np.full((n_cat, n), np.nan)
    net_charge_off[:, 1:] = (nco_rate[:, 1:] / 4.0) * balance
    provision = np.full((n_cat, n), np.nan)
    provision[:, 1:] = (allowance[:, 1:] - allowance[:, :-1]) + net_charge_off[:, 1:]
    return CreditLossProjection(
        timeline=_TIMELINE,
        categories=_CATEGORIES,
        scenario_name="baseline",
        bank_identifier="111",
        balance_mm=np.full((n_cat, n), balance),
        net_charge_off_mm=net_charge_off,
        provision_expense_mm=provision,
        allowance_mm=allowance,
        npl_mm=np.full((n_cat, n), balance * 0.01),
        nco_rate=nco_rate,
        npl_ratio=np.full((n_cat, n), 0.01),
        monte_carlo_mean=np.zeros(n_cat),
        monte_carlo_percentiles={},
    )


def test_extend_is_a_no_op_when_n_periods_already_matches():
    projection = _projection([0.0, 0.02, 0.03, 0.025, 0.02], [0.0, 20.0, 25.0, 23.0, 20.0])
    extended = extend_credit_loss_projection(projection, 5)
    assert extended is projection


def test_extend_raises_when_n_periods_is_shorter():
    projection = _projection([0.0, 0.02, 0.03, 0.025, 0.02], [0.0, 20.0, 25.0, 23.0, 20.0])
    with pytest.raises(ValueError):
        extend_credit_loss_projection(projection, 3)


def test_extend_holds_nco_rate_and_allowance_flat_beyond_the_scenario():
    projection = _projection([0.0, 0.02, 0.03, 0.025, 0.02], [0.0, 20.0, 25.0, 23.0, 20.0])
    extended = extend_credit_loss_projection(projection, 9)

    assert extended.timeline.n_periods == 9
    assert extended.timeline.start_year == 2025
    assert extended.timeline.start_quarter == 4
    # the original 5 periods are untouched
    assert np.allclose(extended.nco_rate[:, :5], projection.nco_rate)
    assert np.allclose(extended.allowance_mm[:, :5], projection.allowance_mm)
    # the 4 extended periods hold the LAST scenario quarter's rate/allowance flat
    assert np.allclose(extended.nco_rate[:, 5:], 0.02)
    assert np.allclose(extended.allowance_mm[:, 5:], 20.0)


def test_extend_provision_equals_net_charge_off_beyond_the_scenario():
    projection = _projection([0.0, 0.02, 0.03, 0.025, 0.02], [0.0, 20.0, 25.0, 23.0, 20.0])
    extended = extend_credit_loss_projection(projection, 9)

    expected_extra_nco = (0.02 / 4.0) * 1000.0
    assert np.allclose(extended.net_charge_off_mm[:, 5:], expected_extra_nco)
    # allowance is flat beyond the scenario, so provision has no roll-forward delta term left
    assert np.allclose(extended.provision_expense_mm[:, 5:], expected_extra_nco)


def test_extend_holds_balance_and_npl_consistent_beyond_the_scenario():
    projection = _projection([0.0, 0.02, 0.03, 0.025, 0.02], [0.0, 20.0, 25.0, 23.0, 20.0])
    extended = extend_credit_loss_projection(projection, 7)
    assert np.allclose(extended.balance_mm[:, 5:], 1000.0)
    assert np.allclose(extended.npl_mm[:, 5:], 1000.0 * 0.01)


def _opening(net_loans_mm=1000.0) -> BankOpeningBalance:
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
        goodwill_net_of_dtl_mm=20.0,
        other_intangibles_net_of_dtl_mm=3.5,
        dta_nol_deduction_mm=0.5,
        aoci_afs_unrealized_mm=-2.0,
        reported_cet1_capital_mm=200.0 - 20.0 - 3.5 - 0.5 + 2.0,
        reported_cet1_ratio=0.12,
        reported_rwa_mm=net_loans_mm,
        reported_tier1_leverage_ratio=0.09,
    )


def test_build_deal_horizon_bank_result_rebuilds_over_the_requested_horizon():
    opening = _opening()
    projection = _projection([0.0, 0.02, 0.03, 0.025, 0.02], [0.0, 20.0, 25.0, 23.0, 20.0])
    result = build_deal_horizon_bank_result(
        opening, projection, BankConfig(), deal_horizon_quarters=8
    )
    assert result.income_statement.timeline.n_periods == 9
    assert result.balance_sheet.timeline.n_periods == 9
    # period-0 (jump-off) figures are unaffected by the horizon extension
    assert result.capital.cet1_capital_mm[0] == pytest.approx(opening.reported_cet1_capital_mm)


def test_truncate_is_a_no_op_when_n_periods_already_matches():
    projection = _projection([0.0, 0.02, 0.03, 0.025, 0.02], [0.0, 20.0, 25.0, 23.0, 20.0])
    truncated = truncate_credit_loss_projection(projection, 5)
    assert truncated is projection


def test_truncate_raises_when_n_periods_is_longer():
    projection = _projection([0.0, 0.02, 0.03, 0.025, 0.02], [0.0, 20.0, 25.0, 23.0, 20.0])
    with pytest.raises(ValueError):
        truncate_credit_loss_projection(projection, 7)


def test_truncate_slices_every_array_and_keeps_the_timeline_convention():
    projection = _projection([0.0, 0.02, 0.03, 0.025, 0.02], [0.0, 20.0, 25.0, 23.0, 20.0])
    truncated = truncate_credit_loss_projection(projection, 3)

    assert truncated.timeline.n_periods == 3
    assert truncated.timeline.start_year == 2025
    assert truncated.timeline.start_quarter == 4
    assert np.sum(~truncated.timeline.is_projection) == 1  # jump-off only
    assert np.allclose(truncated.nco_rate, projection.nco_rate[:, :3])
    assert np.allclose(truncated.allowance_mm, projection.allowance_mm[:, :3])
    assert np.allclose(truncated.balance_mm, projection.balance_mm[:, :3])


def test_build_stress_bank_result_rebuilds_over_the_requested_horizon():
    opening = _opening()
    projection = _projection([0.0, 0.02, 0.03, 0.025, 0.02], [0.0, 20.0, 25.0, 23.0, 20.0])
    result = build_stress_bank_result(opening, projection, BankConfig(), n_quarters=3)
    assert result.income_statement.timeline.n_periods == 4
    assert result.capital.cet1_capital_mm[0] == pytest.approx(opening.reported_cet1_capital_mm)
