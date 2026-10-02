"""Offline tests for corefin.ma.purchase_accounting -- synthetic data
only. Covers the required Stage 3 checks: goodwill = consideration minus
fair value of net assets acquired exactly; PCD has no net effect at
close; mark accretion/CDI amortization schedules sum to the original
amount over their life."""

from __future__ import annotations

import pytest

from corefin.bank.schema import BankOpeningBalance
from corefin.ma.purchase_accounting import (
    compute_fair_value_marks,
    compute_sources_and_uses,
    compute_straight_line_schedule,
    compute_sum_of_years_digits_schedule,
    compute_target_tangible_common_equity_mm,
)
from corefin.ma.schema import (
    CdiConfig,
    ConsiderationConfig,
    CreditMarkConfig,
    Day2AllowanceMethod,
    DealConfig,
    RateMarkConfig,
    SecuritiesMarkConfig,
)


def _target(**overrides) -> BankOpeningBalance:
    kwargs = dict(
        name="Target Bank B",
        bank_id="333",
        hc_rssd_id="444",
        cash_mm=200.0,
        securities_afs_mm=150.0,
        securities_htm_mm=50.0,
        other_assets_mm=40.0,
        goodwill_mm=20.0,
        other_intangibles_mm=5.0,
        deposits_mm=1200.0,
        borrowings_mm=50.0,
        other_liabilities_mm=20.0,
        equity_mm=400.0,
        net_interest_income_jumpoff_mm=15.0,
        noninterest_income_jumpoff_mm=4.0,
        noninterest_expense_jumpoff_mm=12.0,
        goodwill_net_of_dtl_mm=20.0,
        reported_cet1_capital_mm=370.0,
        reported_cet1_ratio=0.15,
        reported_rwa_mm=2400.0,
        reported_tier1_leverage_ratio=0.10,
    )
    kwargs.update(overrides)
    return BankOpeningBalance(**kwargs)


def _deal_config(**overrides) -> DealConfig:
    kwargs: dict = dict(
        consideration=ConsiderationConfig(price_to_tbv=1.5, stock_pct=0.8),
        credit_mark=CreditMarkConfig(credit_mark_pct=0.02, pcd_share=0.3),
    )
    kwargs.update(overrides)
    return DealConfig(**kwargs)


def test_target_tangible_common_equity_excludes_preferred_goodwill_and_intangibles():
    target = _target(
        equity_mm=400.0, preferred_stock_mm=50.0, goodwill_mm=20.0, other_intangibles_mm=5.0
    )
    assert compute_target_tangible_common_equity_mm(target) == pytest.approx(325.0)


def test_pcd_gross_up_exactly_offsets_the_pcd_credit_mark():
    target = _target()
    config = _deal_config(credit_mark=CreditMarkConfig(credit_mark_pct=0.03, pcd_share=0.4))
    marks = compute_fair_value_marks(
        target, target_gross_loans_mm=800.0, target_existing_allowance_mm=16.0, config=config
    )

    assert marks.credit_mark_pcd_mm == pytest.approx(-800.0 * 0.03 * 0.4)
    # the gross-up and the PCD mark are equal and opposite -- net zero effect
    assert marks.pcd_gross_up_mm + marks.credit_mark_pcd_mm == pytest.approx(0.0)


def test_day2_allowance_credit_mark_rate_method():
    target = _target()
    config = _deal_config(
        credit_mark=CreditMarkConfig(
            credit_mark_pct=0.025,
            pcd_share=0.25,
            day2_allowance_method=Day2AllowanceMethod.CREDIT_MARK_RATE,
        )
    )
    marks = compute_fair_value_marks(
        target, target_gross_loans_mm=1000.0, target_existing_allowance_mm=10.0, config=config
    )
    non_pcd_loans_mm = 1000.0 * 0.75
    assert marks.day2_allowance_non_pcd_mm == pytest.approx(0.025 * non_pcd_loans_mm)


def test_day2_allowance_target_acl_ratio_method():
    target = _target()
    config = _deal_config(
        credit_mark=CreditMarkConfig(
            credit_mark_pct=0.025,
            pcd_share=0.25,
            day2_allowance_method=Day2AllowanceMethod.TARGET_ACL_RATIO,
        )
    )
    marks = compute_fair_value_marks(
        target, target_gross_loans_mm=1000.0, target_existing_allowance_mm=12.0, config=config
    )
    non_pcd_loans_mm = 1000.0 * 0.75
    expected_rate = 12.0 / 1000.0
    assert marks.day2_allowance_non_pcd_mm == pytest.approx(expected_rate * non_pcd_loans_mm)


def test_cdi_net_of_dtl():
    target = _target(deposits_mm=1200.0)
    config = _deal_config(cdi=CdiConfig(cdi_pct_of_core_deposits=0.025), tax_rate=0.25)
    marks = compute_fair_value_marks(
        target, target_gross_loans_mm=800.0, target_existing_allowance_mm=16.0, config=config
    )

    expected_gross = 1200.0 * 0.025
    assert marks.cdi_gross_mm == pytest.approx(expected_gross)
    assert marks.cdi_dtl_mm == pytest.approx(expected_gross * 0.25)
    assert marks.cdi_net_of_dtl_mm == pytest.approx(expected_gross * 0.75)


def test_goodwill_equals_consideration_minus_fair_value_of_net_assets_exactly():
    target = _target()
    config = _deal_config(
        consideration=ConsiderationConfig(price_to_tbv=1.4, stock_pct=0.7),
        rate_mark=RateMarkConfig(rate_mark_pct=-0.01),
        securities_mark=SecuritiesMarkConfig(securities_mark_pct=-0.004),
    )
    marks = compute_fair_value_marks(
        target, target_gross_loans_mm=800.0, target_existing_allowance_mm=16.0, config=config
    )
    sources_and_uses = compute_sources_and_uses(target, marks, config)

    assert sources_and_uses.goodwill_mm == pytest.approx(
        sources_and_uses.consideration_mm - sources_and_uses.fair_value_of_net_assets_acquired_mm
    )


def test_zero_premium_zero_mark_zero_cdi_deal_produces_zero_goodwill():
    # price_to_tbv=1.0 (no premium), every mark and CDI at zero, and NO existing allowance to
    # eliminate (so "eliminating the allowance" has no effect either) -- fair value of net
    # assets acquired should exactly equal target's tangible book value, and goodwill zero.
    target = _target()
    config = _deal_config(
        consideration=ConsiderationConfig(price_to_tbv=1.0, stock_pct=1.0),
        credit_mark=CreditMarkConfig(credit_mark_pct=0.0, pcd_share=0.0),
        cdi=CdiConfig(cdi_pct_of_core_deposits=0.0),
    )
    marks = compute_fair_value_marks(
        target, target_gross_loans_mm=800.0, target_existing_allowance_mm=0.0, config=config
    )
    sources_and_uses = compute_sources_and_uses(target, marks, config)

    assert sources_and_uses.goodwill_mm == pytest.approx(0.0, abs=1e-9)


def test_eliminating_a_nonzero_existing_allowance_with_no_offsetting_mark_raises_fair_value():
    # If the target DOES carry an existing allowance but the deal assumes zero credit mark,
    # eliminating that allowance (step 1 of purchase accounting) genuinely raises the fair
    # value of net assets acquired above target's net book tangible equity -- a bargain
    # purchase (negative goodwill) in this edge case, not a bug: book net loans already
    # deducted the allowance, and a zero credit mark means "no further fair-value haircut is
    # needed," which is only consistent with loans being worth MORE than net book value.
    target = _target()
    config = _deal_config(
        consideration=ConsiderationConfig(price_to_tbv=1.0, stock_pct=1.0),
        credit_mark=CreditMarkConfig(credit_mark_pct=0.0, pcd_share=0.0),
        cdi=CdiConfig(cdi_pct_of_core_deposits=0.0),
    )
    marks = compute_fair_value_marks(
        target, target_gross_loans_mm=800.0, target_existing_allowance_mm=16.0, config=config
    )
    sources_and_uses = compute_sources_and_uses(target, marks, config)

    assert sources_and_uses.goodwill_mm == pytest.approx(-16.0)


def test_higher_price_produces_higher_goodwill():
    target = _target()
    low_price_config = _deal_config(
        consideration=ConsiderationConfig(price_to_tbv=1.2, stock_pct=0.8)
    )
    high_price_config = _deal_config(
        consideration=ConsiderationConfig(price_to_tbv=1.8, stock_pct=0.8)
    )

    marks_low = compute_fair_value_marks(target, 800.0, 16.0, low_price_config)
    marks_high = compute_fair_value_marks(target, 800.0, 16.0, high_price_config)
    su_low = compute_sources_and_uses(target, marks_low, low_price_config)
    su_high = compute_sources_and_uses(target, marks_high, high_price_config)

    assert su_high.goodwill_mm > su_low.goodwill_mm


def test_sum_of_years_digits_schedule_sums_to_total_over_the_life():
    schedule = compute_sum_of_years_digits_schedule(
        total_mm=100.0, life_years=10.0, n_periods=40, period_length_years=0.25
    )
    assert len(schedule) == 40
    assert sum(schedule) == pytest.approx(100.0)
    # front-loaded -- first period's amount must exceed the last within-life period's
    assert schedule[0] > schedule[39]


def test_sum_of_years_digits_schedule_is_zero_after_the_life_ends():
    schedule = compute_sum_of_years_digits_schedule(
        total_mm=100.0, life_years=2.0, n_periods=12, period_length_years=0.25
    )
    assert all(v == pytest.approx(0.0) for v in schedule[8:])


def test_straight_line_schedule_sums_to_total_and_is_constant_within_the_life():
    schedule = compute_straight_line_schedule(
        total_mm=50.0, life_years=5.0, n_periods=20, period_length_years=0.25
    )
    assert sum(schedule) == pytest.approx(50.0)
    assert schedule[0] == pytest.approx(schedule[19])
    assert schedule[0] == pytest.approx(50.0 / 20)
