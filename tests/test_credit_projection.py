"""Offline tests for Stage 5's simplified CECL allowance/provision math
-- synthetic data only, no network."""

from __future__ import annotations

import pandas as pd
import pytest

from corefin.credit import projection


def _quarters(start: str, n: int) -> pd.PeriodIndex:
    return pd.period_range(pd.Period(start, freq="Q"), periods=n, freq="Q")


# --------------------------------------------- lifetime expected loss rate ---


def test_lifetime_expected_loss_rate_averages_the_forward_window():
    quarters = _quarters("2026Q1", 4)
    path = pd.Series([0.02, 0.03, 0.04, 0.05], index=quarters)
    rate = projection.lifetime_expected_loss_rate(path, pd.Period("2026Q1", freq="Q"), 4)
    assert rate == pytest.approx((0.02 + 0.03 + 0.04 + 0.05) / 4)


def test_lifetime_expected_loss_rate_holds_last_value_beyond_the_path():
    quarters = _quarters("2026Q1", 2)
    path = pd.Series([0.02, 0.04], index=quarters)
    # wal_quarters=4 reaches 2 quarters past the path's end -- both held at 0.04
    rate = projection.lifetime_expected_loss_rate(path, pd.Period("2026Q1", freq="Q"), 4)
    assert rate == pytest.approx((0.02 + 0.04 + 0.04 + 0.04) / 4)


def test_lifetime_expected_loss_rate_raises_if_quarter_not_in_path():
    path = pd.Series([0.02], index=_quarters("2026Q1", 1))
    with pytest.raises(ValueError, match="not in"):
        projection.lifetime_expected_loss_rate(path, pd.Period("2027Q1", freq="Q"), 4)


def test_backward_looking_lifetime_expected_loss_rate_averages_the_trailing_window():
    quarters = _quarters("2025Q1", 4)
    path = pd.Series([0.01, 0.02, 0.03, 0.04], index=quarters)
    rate = projection.backward_looking_lifetime_expected_loss_rate(
        path, pd.Period("2025Q4", freq="Q"), 4
    )
    assert rate == pytest.approx((0.01 + 0.02 + 0.03 + 0.04) / 4)


def test_backward_looking_lifetime_expected_loss_rate_uses_whatever_history_exists():
    # Only 2 quarters of real history exist even though wal_quarters=4 --
    # average over what's actually there, don't invent data for the rest.
    quarters = _quarters("2025Q3", 2)
    path = pd.Series([0.02, 0.06], index=quarters)
    rate = projection.backward_looking_lifetime_expected_loss_rate(
        path, pd.Period("2025Q4", freq="Q"), 4
    )
    assert rate == pytest.approx((0.02 + 0.06) / 2)


def test_backward_looking_lifetime_expected_loss_rate_raises_with_no_history():
    path = pd.Series([0.02], index=_quarters("2020Q1", 1))
    with pytest.raises(ValueError, match="no realized data"):
        projection.backward_looking_lifetime_expected_loss_rate(
            path, pd.Period("2025Q4", freq="Q"), 4
        )


# --------------------------------------------------------- CECL projection ---


def test_build_cecl_projection_jump_off_row_uses_backward_looking_rate():
    realized = pd.Series([0.01, 0.02, 0.03, 0.04], index=_quarters("2025Q1", 4))
    projected = pd.Series([0.05], index=_quarters("2026Q1", 1))
    result = projection.build_cecl_projection(
        realized, projected, "credit_card", starting_balance=1_000.0
    )
    jump_off = result.loc[pd.Period("2025Q4", freq="Q")]
    wal = projection.CATEGORY_WEIGHTED_AVERAGE_LIFE_QUARTERS["credit_card"]
    expected_rate = projection.backward_looking_lifetime_expected_loss_rate(
        realized, pd.Period("2025Q4", freq="Q"), wal
    )
    assert jump_off["lifetime_expected_loss_rate"] == pytest.approx(expected_rate)
    assert jump_off["allowance_required"] == pytest.approx(1_000.0 * expected_rate)
    assert pd.isna(jump_off["net_charge_off"])
    assert pd.isna(jump_off["provision_expense"])


def test_build_cecl_projection_rejects_projected_path_including_jump_off():
    realized = pd.Series([0.02], index=_quarters("2025Q4", 1))
    projected = pd.Series([0.03], index=_quarters("2025Q4", 1))  # overlaps jump-off
    with pytest.raises(ValueError, match="jump-off"):
        projection.build_cecl_projection(realized, projected, "auto", starting_balance=1_000.0)


def test_build_cecl_projection_provision_follows_the_rollforward_identity():
    # ending = beginning + provision - net_charge_offs
    # => provision = (ending - beginning) + net_charge_offs
    realized = pd.Series([0.02, 0.02, 0.02, 0.02], index=_quarters("2025Q1", 4))
    projected = pd.Series([0.02, 0.02], index=_quarters("2026Q1", 2))  # flat rate throughout
    result = projection.build_cecl_projection(
        realized, projected, "auto", starting_balance=10_000.0
    )
    row_q1 = result.loc[pd.Period("2026Q1", freq="Q")]
    prior_allowance = result.loc[pd.Period("2025Q4", freq="Q"), "allowance_required"]
    expected_provision = (row_q1["allowance_required"] - prior_allowance) + row_q1["net_charge_off"]
    assert row_q1["provision_expense"] == pytest.approx(expected_provision)


def test_build_cecl_projection_net_charge_off_is_quarterly_not_annualized():
    realized = pd.Series([0.02], index=_quarters("2025Q4", 1))
    projected = pd.Series([0.08], index=_quarters("2026Q1", 1))  # 8% annualized
    result = projection.build_cecl_projection(
        realized, projected, "auto", starting_balance=1_000.0
    )
    row = result.loc[pd.Period("2026Q1", freq="Q")]
    assert row["net_charge_off"] == pytest.approx((0.08 / 4.0) * 1_000.0)


def test_build_cecl_projection_allowance_rises_when_projected_losses_rise():
    realized = pd.Series([0.01] * 4, index=_quarters("2025Q1", 4))
    projected_rising = pd.Series([0.02, 0.05, 0.08], index=_quarters("2026Q1", 3))
    result = projection.build_cecl_projection(
        realized, projected_rising, "auto", starting_balance=1_000.0
    )
    allowances = result["allowance_required"]
    assert allowances.iloc[-1] > allowances.iloc[0]


def test_category_weighted_average_life_covers_every_modeling_category():
    from corefin.credit.backtest import MODELING_CATEGORIES

    for category in MODELING_CATEGORIES:
        assert str(category) in projection.CATEGORY_WEIGHTED_AVERAGE_LIFE_QUARTERS
        assert projection.CATEGORY_WEIGHTED_AVERAGE_LIFE_QUARTERS[str(category)] > 0
