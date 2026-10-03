"""Offline tests for corefin.ma.schema -- DealConfig validation."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from corefin.ma.schema import (
    ConsiderationConfig,
    CreditMarkConfig,
    Day2AllowanceMethod,
    DealConfig,
    DistributableCashMethod,
    ExitMultipleBasis,
    IrrConfig,
)


def _deal_config(**overrides) -> DealConfig:
    kwargs = dict(
        consideration=ConsiderationConfig(
            price_to_tbv=1.5,
            stock_pct=0.8,
            acquirer_share_price=25.0,
            acquirer_shares_outstanding_mm=50.0,
        ),
        credit_mark=CreditMarkConfig(credit_mark_pct=0.02, pcd_share=0.3),
    )
    kwargs.update(overrides)
    return DealConfig(**kwargs)


def test_consideration_config_cash_pct_is_complement_of_stock_pct():
    consideration = ConsiderationConfig(
        price_to_tbv=1.5,
        stock_pct=0.8,
        acquirer_share_price=25.0,
        acquirer_shares_outstanding_mm=50.0,
    )
    assert consideration.cash_pct == pytest.approx(0.2)


def test_deal_config_defaults_tax_rate_to_25_percent():
    config = _deal_config()
    assert config.tax_rate == pytest.approx(0.25)


def test_deal_config_defaults_day2_method_to_credit_mark_rate():
    config = _deal_config()
    assert config.credit_mark.day2_allowance_method == Day2AllowanceMethod.CREDIT_MARK_RATE


def test_deal_config_rejects_unknown_fields():
    with pytest.raises(ValidationError):
        _deal_config(not_a_real_field=1.0)


def test_consideration_config_rejects_stock_pct_out_of_range():
    with pytest.raises(ValidationError):
        ConsiderationConfig(
            price_to_tbv=1.5,
            stock_pct=1.5,
            acquirer_share_price=25.0,
            acquirer_shares_outstanding_mm=50.0,
        )


def test_consideration_config_rejects_non_positive_price_to_tbv():
    with pytest.raises(ValidationError):
        ConsiderationConfig(
            price_to_tbv=0.0,
            stock_pct=0.5,
            acquirer_share_price=25.0,
            acquirer_shares_outstanding_mm=50.0,
        )


def test_rate_mark_config_allows_negative_pct_for_a_writedown():
    config = _deal_config()
    assert config.rate_mark.rate_mark_pct == pytest.approx(0.0)  # default
    from corefin.ma.schema import RateMarkConfig

    writedown = RateMarkConfig(rate_mark_pct=-0.03)
    assert writedown.rate_mark_pct == pytest.approx(-0.03)


def test_cdi_config_defaults_match_the_approved_plan():
    config = _deal_config()
    assert config.cdi.cdi_pct_of_core_deposits == pytest.approx(0.02)
    assert config.cdi.cdi_amortization_years == pytest.approx(10.0)


def test_deal_config_defaults_horizon_to_5_years_of_quarters():
    config = _deal_config()
    assert config.deal_horizon_quarters == 20


def test_deal_config_horizon_is_overridable():
    config = _deal_config(deal_horizon_quarters=12)
    assert config.deal_horizon_quarters == 12


def test_irr_config_defaults_to_dividends_and_1x_price_to_tbv():
    config = _deal_config()
    assert config.irr.distributable_cash_method == DistributableCashMethod.DIVIDENDS
    assert config.irr.exit_multiple_basis == ExitMultipleBasis.PRICE_TO_TBV
    assert config.irr.exit_multiple == pytest.approx(1.0)


def test_irr_config_excess_capital_method_requires_target_cet1_ratio():
    with pytest.raises(ValidationError):
        IrrConfig(
            distributable_cash_method=DistributableCashMethod.EXCESS_CAPITAL_ABOVE_TARGET_CET1
        )


def test_irr_config_excess_capital_method_accepts_target_cet1_ratio():
    config = IrrConfig(
        distributable_cash_method=DistributableCashMethod.EXCESS_CAPITAL_ABOVE_TARGET_CET1,
        target_cet1_ratio=0.10,
    )
    assert config.target_cet1_ratio == pytest.approx(0.10)


def test_irr_config_exit_multiple_must_be_positive():
    with pytest.raises(ValidationError):
        IrrConfig(exit_multiple=0.0)
