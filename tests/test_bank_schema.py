"""Offline tests for corefin.bank.schema -- BankConfig/BankOpeningBalance
validation and the default risk-weight table."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from corefin.bank.schema import (
    DEFAULT_RISK_WEIGHTS,
    AssetRiskCategory,
    BankConfig,
    BankOpeningBalance,
)


def _opening_kwargs(**overrides):
    kwargs = dict(
        name="Acquirer Bank A",
        bank_id="34537",
        hc_rssd_id="1085013",
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
    return kwargs


def test_bank_config_default_tax_rate_is_25_percent_combined():
    # Explicit user direction: 25% combined federal+state default, not the 21% federal
    # statutory rate alone.
    config = BankConfig()
    assert config.tax_rate == pytest.approx(0.25)


def test_bank_config_default_aoci_opt_out_is_true():
    config = BankConfig()
    assert config.aoci_opt_out is True


def test_bank_config_rejects_unknown_fields():
    with pytest.raises(ValidationError):
        BankConfig(not_a_real_field=1.0)


def test_bank_config_rejects_tax_rate_out_of_range():
    with pytest.raises(ValidationError):
        BankConfig(tax_rate=1.5)


def test_default_risk_weights_cover_every_asset_risk_category():
    for category in AssetRiskCategory:
        assert category in DEFAULT_RISK_WEIGHTS
        assert 0.0 <= DEFAULT_RISK_WEIGHTS[category] <= 1.0


def test_bank_opening_balance_valid_construction():
    opening = BankOpeningBalance(**_opening_kwargs())
    assert opening.name == "Acquirer Bank A"
    assert opening.bank_id == "34537"


def test_bank_opening_balance_rejects_cet1_ratio_outside_unit_interval():
    with pytest.raises(ValidationError):
        BankOpeningBalance(**_opening_kwargs(reported_cet1_ratio=1.5))


def test_bank_opening_balance_rejects_negative_cash():
    with pytest.raises(ValidationError):
        BankOpeningBalance(**_opening_kwargs(cash_mm=-1.0))


def test_bank_opening_balance_rejects_unknown_fields():
    with pytest.raises(ValidationError):
        BankOpeningBalance(**_opening_kwargs(not_a_real_field=1.0))
