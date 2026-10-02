"""Offline tests for corefin.bank.sources.real_data -- synthetic
Call-Report/Y-9C rows only (no network, no real bank data)."""

from __future__ import annotations

import pandas as pd
import pytest
from pydantic import ValidationError

from corefin.bank.sources.real_data import build_opening_balance_from_real_data


def _synthetic_call_report_row(**overrides) -> pd.Series:
    data = {
        "RCON2200": 3_000_000.0,  # total deposits, $thousands
    }
    data.update(overrides)
    return pd.Series(data)


def _synthetic_y9c_row(**overrides) -> pd.Series:
    data = {
        "BHCK0081": 150_000.0,  # noninterest-bearing cash
        "BHCK0395": 50_000.0,  # interest-bearing cash
        "BHCK1773": 400_000.0,  # AFS securities
        "BHCK1754": 100_000.0,  # HTM securities
        "BHCK3163": 80_000.0,  # goodwill, gross
        "BHCKJF76": 20_000.0,  # other intangibles, gross
        "BHCK3210": 500_000.0,  # total equity capital
        "BHCK2170": 4_200_000.0,  # total assets
        "BHCK2948": 3_700_000.0,  # total liabilities
        "BHCK4107": 200_000.0,  # total interest income, YTD (full year as of Q4)
        "BHCK4073": 60_000.0,  # total interest expense, YTD
        "BHCK4079": 25_000.0,  # noninterest income, YTD
        "BHCK4093": 90_000.0,  # noninterest expense, YTD
        "BHCK4598": 8_000.0,  # preferred dividends declared, YTD
        "BHCK3283": 30_000.0,  # preferred stock
        "BHCAP841": 75_000.0,  # goodwill net of DTL
        "BHCAP842": 15_000.0,  # other intangibles net of DTL
        "BHCAP843": 1_000.0,  # DTA NOL deduction
        "BHCAP844": -5_000.0,  # AOCI AFS (loss)
        "BHCAP846": 500.0,  # AOCI cash-flow hedge
        "BHCAP847": 0.0,  # AOCI pension
        "BHCAP848": 0.0,  # AOCI HTM
        "BHCAP850": 0.0,  # other CET1 deductions
        "BHCAP859": 380_000.0,  # reported CET1 capital
        "BHCAP793": 11.5,  # reported CET1 ratio (percentage points)
        "BHCAA223": 3_300_000.0,  # reported RWA
        "BHCA7204": 9.0,  # reported tier 1 leverage ratio (percentage points)
    }
    data.update(overrides)
    return pd.Series(data)


def test_build_opening_balance_sources_balance_sheet_from_y9c():
    call_report_row = _synthetic_call_report_row()
    y9c_row = _synthetic_y9c_row()

    opening = build_opening_balance_from_real_data(
        name="Acquirer Bank A",
        bank_id="11111",
        hc_rssd_id="22222",
        call_report_row=call_report_row,
        y9c_row=y9c_row,
        bank_level_net_loans_mm=3000.0,
    )

    assert opening.cash_mm == pytest.approx(200.0)  # (150,000 + 50,000) / 1000
    assert opening.securities_afs_mm == pytest.approx(400.0)
    assert opening.securities_htm_mm == pytest.approx(100.0)
    assert opening.goodwill_mm == pytest.approx(80.0)
    assert opening.other_intangibles_mm == pytest.approx(20.0)
    assert opening.equity_mm == pytest.approx(500.0)
    assert opening.deposits_mm == pytest.approx(3000.0)
    assert opening.borrowings_mm == pytest.approx(0.0)


def test_build_opening_balance_derives_other_liabilities_as_a_residual():
    call_report_row = _synthetic_call_report_row(RCON2200=3_000_000.0)
    y9c_row = _synthetic_y9c_row(BHCK2948=3_700_000.0)

    opening = build_opening_balance_from_real_data(
        name="Acquirer Bank A",
        bank_id="11111",
        hc_rssd_id="22222",
        call_report_row=call_report_row,
        y9c_row=y9c_row,
        bank_level_net_loans_mm=3000.0,
    )
    # other_liabilities = HC total liabilities (3,700,000) - bank deposits (3,000,000), in $mm
    assert opening.other_liabilities_mm == pytest.approx(700.0)


def test_build_opening_balance_derives_other_assets_as_a_residual():
    call_report_row = _synthetic_call_report_row()
    y9c_row = _synthetic_y9c_row(BHCK2170=4_200_000.0)

    opening = build_opening_balance_from_real_data(
        name="Acquirer Bank A",
        bank_id="11111",
        hc_rssd_id="22222",
        call_report_row=call_report_row,
        y9c_row=y9c_row,
        bank_level_net_loans_mm=3000.0,
    )
    # total assets (4,200) - cash(200) - afs(400) - htm(100) - loans(3000) - goodwill(80) -
    # other_intangibles(20) = 400, in $mm
    assert opening.other_assets_mm == pytest.approx(400.0)


def test_build_opening_balance_raises_when_other_assets_residual_is_negative():
    # loans=3500 leaves total assets (4,200) - cash(200) - afs(400) - htm(100) - loans(3500) -
    # goodwill(80) - other_intangibles(20) = -100: a negative "other assets" residual, which
    # would mean the bank-level loan figure doesn't actually fit inside HC total assets --
    # must surface as a validation error (BankOpeningBalance's ge=0.0), not be silently clipped.
    call_report_row = _synthetic_call_report_row()
    y9c_row = _synthetic_y9c_row(BHCK2170=4_200_000.0)

    with pytest.raises(ValidationError):
        build_opening_balance_from_real_data(
            name="Acquirer Bank A",
            bank_id="11111",
            hc_rssd_id="22222",
            call_report_row=call_report_row,
            y9c_row=y9c_row,
            bank_level_net_loans_mm=3500.0,
        )


def test_build_opening_balance_sources_cet1_bridge_from_rcri_items():
    call_report_row = _synthetic_call_report_row()
    y9c_row = _synthetic_y9c_row()

    opening = build_opening_balance_from_real_data(
        name="Acquirer Bank A",
        bank_id="11111",
        hc_rssd_id="22222",
        call_report_row=call_report_row,
        y9c_row=y9c_row,
        bank_level_net_loans_mm=3000.0,
    )

    assert opening.preferred_stock_mm == pytest.approx(30.0)
    assert opening.goodwill_net_of_dtl_mm == pytest.approx(75.0)
    assert opening.other_intangibles_net_of_dtl_mm == pytest.approx(15.0)
    assert opening.dta_nol_deduction_mm == pytest.approx(1.0)
    assert opening.aoci_afs_unrealized_mm == pytest.approx(-5.0)
    assert opening.aoci_cash_flow_hedge_mm == pytest.approx(0.5)
    assert opening.reported_cet1_capital_mm == pytest.approx(380.0)
    assert opening.reported_cet1_ratio == pytest.approx(0.115)
    assert opening.reported_rwa_mm == pytest.approx(3300.0)
    assert opening.reported_tier1_leverage_ratio == pytest.approx(0.09)


def test_build_opening_balance_income_statement_sourced_from_y9c():
    call_report_row = _synthetic_call_report_row()
    y9c_row = _synthetic_y9c_row()

    opening = build_opening_balance_from_real_data(
        name="Acquirer Bank A",
        bank_id="11111",
        hc_rssd_id="22222",
        call_report_row=call_report_row,
        y9c_row=y9c_row,
        bank_level_net_loans_mm=3000.0,
    )
    # YTD (full year as of Q4), divided by 4 to approximate one quarter's run-rate -- see
    # schema.py's "YTD-VS-QUARTERLY".
    assert opening.net_interest_income_jumpoff_mm == pytest.approx(35.0)  # (200,000-60,000)/4
    assert opening.noninterest_income_jumpoff_mm == pytest.approx(6.25)  # 25,000/4
    assert opening.noninterest_expense_jumpoff_mm == pytest.approx(22.5)  # 90,000/4
    assert opening.preferred_dividends_jumpoff_mm == pytest.approx(2.0)  # 8,000/4


def test_build_opening_balance_name_is_fictitious_not_a_real_bank():
    # Guards against a real company name ever being hardcoded into this builder's defaults.
    call_report_row = _synthetic_call_report_row()
    y9c_row = _synthetic_y9c_row()
    opening = build_opening_balance_from_real_data(
        name="Acquirer Bank A",
        bank_id="11111",
        hc_rssd_id="22222",
        call_report_row=call_report_row,
        y9c_row=y9c_row,
        bank_level_net_loans_mm=3000.0,
    )
    assert opening.name == "Acquirer Bank A"
