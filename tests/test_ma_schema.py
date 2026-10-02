"""Offline tests for corefin.ma.schema -- DealConfig validation."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from corefin.ma.schema import (
    ConsiderationConfig,
    CreditMarkConfig,
    Day2AllowanceMethod,
    DealConfig,
)


def _deal_config(**overrides) -> DealConfig:
    kwargs = dict(
        consideration=ConsiderationConfig(price_to_tbv=1.5, stock_pct=0.8),
        credit_mark=CreditMarkConfig(credit_mark_pct=0.02, pcd_share=0.3),
    )
    kwargs.update(overrides)
    return DealConfig(**kwargs)


def test_consideration_config_cash_pct_is_complement_of_stock_pct():
    consideration = ConsiderationConfig(price_to_tbv=1.5, stock_pct=0.8)
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
        ConsiderationConfig(price_to_tbv=1.5, stock_pct=1.5)


def test_consideration_config_rejects_non_positive_price_to_tbv():
    with pytest.raises(ValidationError):
        ConsiderationConfig(price_to_tbv=0.0, stock_pct=0.5)


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
