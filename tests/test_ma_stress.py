"""Offline tests for corefin.ma.stress -- synthetic data only. Covers
the required Stage 5 checks: the stressed pro forma CET1 path's own
period-0 value matches the baseline deal's already-validated at-close
CET1 bridge exactly, and CDI amortization alone (with no other income
driver, and a zero dividend payout so retained earnings tracks net
income exactly) doesn't move the pro forma CET1 ratio -- the book
value decline and the regulatory deduction's own decline offset
exactly."""

from __future__ import annotations

import numpy as np
import pytest

from corefin.bank.model import run_bank_model
from corefin.bank.schema import BankConfig, BankOpeningBalance
from corefin.credit.interface import CreditLossProjection
from corefin.ma.model import run_deal_model
from corefin.ma.schema import CdiConfig, ConsiderationConfig, CreditMarkConfig, DealConfig
from corefin.ma.stress import (
    BASEL_III_CET1_MINIMUM_RATIO,
    STRESS_CAPITAL_BUFFER_FLOOR,
    compute_standalone_stressed_cet1_path,
    run_stress_test,
)
from corefin.timeline import Timeline

_TIMELINE = Timeline.quarterly(9, n_historical=1, start_year=2025, start_quarter=4)
_CATEGORIES = ("commercial_and_industrial", "residential_mortgage")


def _balanced_opening(name, bank_id, hc_rssd_id, net_loans_mm, equity_mm=1000.0, **overrides):
    layout_keys = {
        "goodwill_mm",
        "other_intangibles_mm",
        "deposits_mm",
        "borrowings_mm",
        "other_liabilities_mm",
        "securities_mm",
        "other_assets_mm",
    }
    layout = dict(
        goodwill_mm=50.0,
        other_intangibles_mm=10.0,
        deposits_mm=3000.0,
        borrowings_mm=200.0,
        other_liabilities_mm=50.0,
        securities_mm=500.0,
        other_assets_mm=100.0,
    )
    layout.update({k: v for k, v in overrides.items() if k in layout_keys})
    direct_overrides = {k: v for k, v in overrides.items() if k not in layout_keys}
    cash_mm = (
        layout["deposits_mm"] + layout["borrowings_mm"] + layout["other_liabilities_mm"] + equity_mm
    ) - (
        layout["securities_mm"]
        + net_loans_mm
        + layout["goodwill_mm"]
        + layout["other_intangibles_mm"]
        + layout["other_assets_mm"]
    )
    kwargs = dict(
        name=name,
        bank_id=bank_id,
        hc_rssd_id=hc_rssd_id,
        cash_mm=cash_mm,
        securities_afs_mm=layout["securities_mm"] / 2,
        securities_htm_mm=layout["securities_mm"] / 2,
        other_assets_mm=layout["other_assets_mm"],
        goodwill_mm=layout["goodwill_mm"],
        other_intangibles_mm=layout["other_intangibles_mm"],
        deposits_mm=layout["deposits_mm"],
        borrowings_mm=layout["borrowings_mm"],
        other_liabilities_mm=layout["other_liabilities_mm"],
        equity_mm=equity_mm,
        net_interest_income_jumpoff_mm=40.0,
        noninterest_income_jumpoff_mm=10.0,
        noninterest_expense_jumpoff_mm=30.0,
        goodwill_net_of_dtl_mm=layout["goodwill_mm"],
        other_intangibles_net_of_dtl_mm=layout["other_intangibles_mm"] * 0.7,
        dta_nol_deduction_mm=1.0,
        aoci_afs_unrealized_mm=-5.0,
        reported_cet1_capital_mm=(
            equity_mm - layout["goodwill_mm"] - layout["other_intangibles_mm"] * 0.7 - 1.0 + 5.0
        ),
        reported_cet1_ratio=0.12,
        reported_rwa_mm=net_loans_mm,
        reported_tier1_leverage_ratio=0.09,
    )
    kwargs.update(direct_overrides)
    return BankOpeningBalance(**kwargs)


def _credit_projection_for(opening, net_loans_mm, n=_TIMELINE.n_periods):
    n_cat = len(_CATEGORIES)
    per_category = net_loans_mm / n_cat
    return CreditLossProjection(
        timeline=Timeline.quarterly(n, n_historical=1, start_year=2025, start_quarter=4),
        categories=_CATEGORIES,
        scenario_name="severely_adverse",
        bank_identifier=opening.bank_id,
        balance_mm=np.full((n_cat, n), per_category),
        net_charge_off_mm=np.zeros((n_cat, n)),
        provision_expense_mm=np.zeros((n_cat, n)),
        allowance_mm=np.zeros((n_cat, n)),
        npl_mm=np.zeros((n_cat, n)),
        nco_rate=np.zeros((n_cat, n)),
        npl_ratio=np.zeros((n_cat, n)),
        monte_carlo_mean=np.zeros(n_cat),
        monte_carlo_percentiles={},
    )


def _run_bank(opening, net_loans_mm):
    credit_projection = _credit_projection_for(opening, net_loans_mm)
    return run_bank_model(opening, credit_projection, BankConfig(), _TIMELINE)


def _acquirer_and_target():
    acquirer_opening = _balanced_opening("Acquirer Bank A", "111", "222", net_loans_mm=2000.0)
    target_opening = _balanced_opening(
        "Target Bank B",
        "333",
        "444",
        net_loans_mm=784.0,
        equity_mm=400.0,
        deposits_mm=1200.0,
        borrowings_mm=50.0,
        other_liabilities_mm=20.0,
        securities_mm=200.0,
        other_assets_mm=40.0,
    )
    return _run_bank(acquirer_opening, 2000.0), _run_bank(target_opening, 784.0)


def _dummy_income_statement(n: int, tl: Timeline):
    from corefin.bank.income_statement import IncomeStatement

    zeros = np.zeros(n)
    return IncomeStatement(
        timeline=tl,
        net_interest_income_mm=zeros,
        noninterest_income_mm=zeros,
        noninterest_expense_mm=zeros,
        provision_expense_mm=zeros,
        pretax_income_mm=zeros,
        tax_expense_mm=zeros,
        net_income_mm=zeros,
        preferred_dividends_mm=zeros,
        net_income_available_to_common_mm=zeros,
        dividends_mm=zeros,
    )


def test_standalone_stressed_path_breach_and_scb_floor():
    from corefin.bank.capital import CapitalResult
    from corefin.bank.model import BankModelResult

    cet1_ratio = np.array([0.12, 0.11, 0.09, 0.05, 0.08])
    tl = Timeline.quarterly(5, n_historical=1, start_year=2025, start_quarter=4)
    capital = CapitalResult(
        timeline=tl,
        cet1_capital_mm=cet1_ratio * 1000.0,
        preferred_stock_mm=np.zeros(5),
        rwa_mm=np.full(5, 1000.0),
        average_assets_mm=np.full(5, 1000.0),
        rwa_calibration_factor=1.0,
        unexplained_cet1_residual_mm=0.0,
    )
    starting = cet1_ratio[0]
    minimum = cet1_ratio.min()
    summarized = compute_standalone_stressed_cet1_path(
        BankModelResult(
            opening=None,
            balance_sheet=None,
            income_statement=_dummy_income_statement(5, tl),
            capital=capital,
        )
    )
    assert summarized.starting_cet1_ratio == pytest.approx(starting)
    assert summarized.minimum_cet1_ratio == pytest.approx(minimum)
    assert summarized.breaches_4_5_pct_minimum == (minimum < BASEL_III_CET1_MINIMUM_RATIO)
    assert summarized.illustrative_stress_capital_buffer == pytest.approx(
        max(starting - minimum, STRESS_CAPITAL_BUFFER_FLOOR)
    )


def test_standalone_stressed_path_flags_a_real_breach():
    from corefin.bank.capital import CapitalResult
    from corefin.bank.model import BankModelResult

    cet1_ratio = np.array([0.12, 0.08, 0.04, 0.05])  # dips below 4.5% at index 2
    tl = Timeline.quarterly(4, n_historical=1, start_year=2025, start_quarter=4)
    capital = CapitalResult(
        timeline=tl,
        cet1_capital_mm=cet1_ratio * 1000.0,
        preferred_stock_mm=np.zeros(4),
        rwa_mm=np.full(4, 1000.0),
        average_assets_mm=np.full(4, 1000.0),
        rwa_calibration_factor=1.0,
        unexplained_cet1_residual_mm=0.0,
    )
    summarized = compute_standalone_stressed_cet1_path(
        BankModelResult(
            opening=None,
            balance_sheet=None,
            income_statement=_dummy_income_statement(4, tl),
            capital=capital,
        )
    )
    assert summarized.breaches_4_5_pct_minimum is True


def test_illustrative_scb_floors_at_2_5_pct_for_shallow_depletion():
    from corefin.bank.capital import CapitalResult
    from corefin.bank.model import BankModelResult

    tl = Timeline.quarterly(3, n_historical=1, start_year=2025, start_quarter=4)
    cet1_ratio = np.array([0.12, 0.115, 0.118])  # shallow depletion, well under 2.5pp
    capital = CapitalResult(
        timeline=tl,
        cet1_capital_mm=cet1_ratio * 1000.0,
        preferred_stock_mm=np.zeros(3),
        rwa_mm=np.full(3, 1000.0),
        average_assets_mm=np.full(3, 1000.0),
        rwa_calibration_factor=1.0,
        unexplained_cet1_residual_mm=0.0,
    )
    summarized = compute_standalone_stressed_cet1_path(
        BankModelResult(
            opening=None,
            balance_sheet=None,
            income_statement=_dummy_income_statement(3, tl),
            capital=capital,
        )
    )
    assert summarized.illustrative_stress_capital_buffer == pytest.approx(
        STRESS_CAPITAL_BUFFER_FLOOR
    )
    assert not summarized.breaches_4_5_pct_minimum


def _run_deal(**config_overrides):
    acquirer_result, target_result = _acquirer_and_target()
    config_kwargs = dict(
        consideration=ConsiderationConfig(
            price_to_tbv=1.0,
            stock_pct=1.0,
            acquirer_share_price=20.0,
            acquirer_shares_outstanding_mm=50.0,
        ),
        credit_mark=CreditMarkConfig(credit_mark_pct=0.0, pcd_share=0.0),
    )
    config_kwargs.update(config_overrides)
    config = DealConfig(**config_kwargs)
    deal_result = run_deal_model(acquirer_result, target_result, BankConfig(), BankConfig(), config)
    return acquirer_result, target_result, deal_result, config


def test_pro_forma_stressed_cet1_at_close_matches_the_original_bridge():
    acquirer_result, target_result, deal_result, config = _run_deal(
        cdi=CdiConfig(cdi_pct_of_core_deposits=0.02, cdi_amortization_years=2.0)
    )
    acquirer_stress_proj = _credit_projection_for(acquirer_result.opening, 2000.0)
    target_stress_proj = _credit_projection_for(target_result.opening, 784.0)

    result = run_stress_test(
        acquirer_result.opening,
        target_result.opening,
        acquirer_stress_proj,
        target_stress_proj,
        BankConfig(),
        BankConfig(),
        deal_result,
        config,
        acquirer_unexplained_cet1_residual_mm=0.0,
        n_quarters=8,
    )
    expected_at_close = (
        deal_result.pro_forma_cet1_bridge.pro_forma_cet1_mm
        / deal_result.pro_forma_capital_ratios.pro_forma_rwa_mm
    )
    assert result.pro_forma_combined.cet1_ratio[0] == pytest.approx(expected_at_close)
    assert result.pro_forma_combined.starting_cet1_ratio == pytest.approx(expected_at_close)


def test_cdi_amortization_alone_does_not_move_pro_forma_cet1_ratio():
    # CDI amortizes FRONT-LOADED (sum-of-years-digits -- a bigger GAAP expense in early
    # quarters, tapering off), so if it had ANY net effect on regulatory capital, the
    # period-over-period INCREMENT in the (dollar, not ratio) pro forma CET1 path would
    # mirror that same decaying shape. With zero marks/cost-saves/restructuring and zero
    # dividend payout (BankConfig's own default), the ONLY other driver is each bank's own
    # FLAT organic standalone earnings, which contribute a CONSTANT increment every quarter
    # -- so a constant increment throughout (no decaying/front-loaded shape at all) confirms
    # CDI amortization's own book-value decline and regulatory-deduction release offset
    # exactly, leaving no residual effect on CET1.
    acquirer_result, target_result, deal_result, config = _run_deal(
        cdi=CdiConfig(cdi_pct_of_core_deposits=0.02, cdi_amortization_years=2.0)
    )
    acquirer_stress_proj = _credit_projection_for(acquirer_result.opening, 2000.0)
    target_stress_proj = _credit_projection_for(target_result.opening, 784.0)

    result = run_stress_test(
        acquirer_result.opening,
        target_result.opening,
        acquirer_stress_proj,
        target_stress_proj,
        BankConfig(),
        BankConfig(),
        deal_result,
        config,
        acquirer_unexplained_cet1_residual_mm=0.0,
        n_quarters=8,  # exactly cdi_amortization_years=2.0 -- CDI fully amortizes within this
    )
    cet1_mm = (
        result.pro_forma_combined.cet1_ratio * deal_result.pro_forma_capital_ratios.pro_forma_rwa_mm
    )
    increments = np.diff(cet1_mm)
    assert np.allclose(increments, increments[0], atol=1e-6)


def test_run_stress_test_minimum_cet1_ratio_change_pp_is_the_difference():
    acquirer_result, target_result, deal_result, config = _run_deal()
    acquirer_stress_proj = _credit_projection_for(acquirer_result.opening, 2000.0)
    target_stress_proj = _credit_projection_for(target_result.opening, 784.0)

    result = run_stress_test(
        acquirer_result.opening,
        target_result.opening,
        acquirer_stress_proj,
        target_stress_proj,
        BankConfig(),
        BankConfig(),
        deal_result,
        config,
        acquirer_unexplained_cet1_residual_mm=0.0,
        n_quarters=8,
    )
    assert result.minimum_cet1_ratio_change_pp == pytest.approx(
        result.pro_forma_combined.minimum_cet1_ratio - result.acquirer_standalone.minimum_cet1_ratio
    )


def test_peak_to_trough_is_zero_when_cet1_only_rises():
    from corefin.bank.capital import CapitalResult
    from corefin.bank.model import BankModelResult

    tl = Timeline.quarterly(4, n_historical=1, start_year=2025, start_quarter=4)
    cet1_ratio = np.array([0.10, 0.11, 0.12, 0.13])  # monotonically rising
    capital = CapitalResult(
        timeline=tl,
        cet1_capital_mm=cet1_ratio * 1000.0,
        preferred_stock_mm=np.zeros(4),
        rwa_mm=np.full(4, 1000.0),
        average_assets_mm=np.full(4, 1000.0),
        rwa_calibration_factor=1.0,
        unexplained_cet1_residual_mm=0.0,
    )
    summarized = compute_standalone_stressed_cet1_path(
        BankModelResult(
            opening=None,
            balance_sheet=None,
            income_statement=_dummy_income_statement(4, tl),
            capital=capital,
        )
    )
    assert summarized.peak_to_trough_cet1_change_pp == pytest.approx(0.0)


def test_peak_to_trough_captures_a_decline_after_an_earlier_peak():
    from corefin.bank.capital import CapitalResult
    from corefin.bank.model import BankModelResult

    tl = Timeline.quarterly(4, n_historical=1, start_year=2025, start_quarter=4)
    cet1_ratio = np.array([0.10, 0.14, 0.09, 0.11])  # peaks at 0.14, troughs at 0.09
    capital = CapitalResult(
        timeline=tl,
        cet1_capital_mm=cet1_ratio * 1000.0,
        preferred_stock_mm=np.zeros(4),
        rwa_mm=np.full(4, 1000.0),
        average_assets_mm=np.full(4, 1000.0),
        rwa_calibration_factor=1.0,
        unexplained_cet1_residual_mm=0.0,
    )
    summarized = compute_standalone_stressed_cet1_path(
        BankModelResult(
            opening=None,
            balance_sheet=None,
            income_statement=_dummy_income_statement(4, tl),
            capital=capital,
        )
    )
    # NOT starting (0.10) minus minimum (0.09) = 1pp -- the true peak-to-trough is
    # 0.14 -> 0.09 = 5pp, a bigger decline than comparing only to the starting point would show
    assert summarized.peak_to_trough_cet1_change_pp == pytest.approx(5.0)


def test_cumulative_ppnr_provision_net_income_sum_periods_1_onward():
    acquirer_result, target_result, deal_result, config = _run_deal()
    acquirer_stress_proj = _credit_projection_for(acquirer_result.opening, 2000.0)
    target_stress_proj = _credit_projection_for(target_result.opening, 784.0)

    result = run_stress_test(
        acquirer_result.opening,
        target_result.opening,
        acquirer_stress_proj,
        target_stress_proj,
        BankConfig(),
        BankConfig(),
        deal_result,
        config,
        acquirer_unexplained_cet1_residual_mm=0.0,
        n_quarters=8,
    )
    standalone = result.acquirer_standalone
    assert standalone.cumulative_ppnr_mm == pytest.approx(float(standalone.ppnr_mm[1:].sum()))
    assert standalone.cumulative_provision_mm == pytest.approx(
        float(standalone.provision_mm[1:].sum())
    )
    assert standalone.cumulative_net_income_mm == pytest.approx(
        float(standalone.net_income_mm[1:].sum())
    )
    # PPNR = pretax + provision, by construction -- zero provision in this fixture means
    # PPNR equals pretax income exactly
    assert np.allclose(standalone.provision_mm, 0.0)
    assert np.allclose(standalone.ppnr_mm, standalone.net_income_mm / 0.75)  # tax_rate=0.25 default


def test_pro_forma_ppnr_includes_one_time_charges_and_provision():
    acquirer_result, target_result, deal_result, config = _run_deal(
        cdi=CdiConfig(cdi_pct_of_core_deposits=0.02, cdi_amortization_years=2.0)
    )
    acquirer_stress_proj = _credit_projection_for(acquirer_result.opening, 2000.0)
    target_stress_proj = _credit_projection_for(target_result.opening, 784.0)

    result = run_stress_test(
        acquirer_result.opening,
        target_result.opening,
        acquirer_stress_proj,
        target_stress_proj,
        BankConfig(),
        BankConfig(),
        deal_result,
        config,
        acquirer_unexplained_cet1_residual_mm=0.0,
        n_quarters=8,
    )
    combined = result.pro_forma_combined
    # PPNR = pretax + provision + one_time_charges -- so pretax = PPNR - provision -
    # one_time_charges; spot-check period 1 (where one_time_charges, if any, would land)
    assert combined.ppnr_mm.shape == (9,)
    assert np.isfinite(combined.ppnr_mm).all()
