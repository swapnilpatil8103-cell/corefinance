"""Offline tests for corefin.ma.projection -- synthetic data only. Covers
the required Stage 4 check: mark accretion/CDI amortization sum to their
original amounts over their configured lives, within the full pro forma
projection (not just the standalone schedule functions -- see
test_ma_purchase_accounting.py for those)."""

from __future__ import annotations

import numpy as np
import pytest

from corefin.bank.model import run_bank_model
from corefin.bank.schema import BankConfig, BankOpeningBalance
from corefin.credit.interface import CreditLossProjection
from corefin.ma.projection import compute_pro_forma_projection
from corefin.ma.purchase_accounting import compute_fair_value_marks, compute_sources_and_uses
from corefin.ma.schema import (
    CdiConfig,
    ConsiderationConfig,
    CreditMarkConfig,
    DealConfig,
    RateMarkConfig,
    SecuritiesMarkConfig,
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


def _credit_projection_for(opening, net_loans_mm):
    n_cat = len(_CATEGORIES)
    n = _TIMELINE.n_periods
    per_category = net_loans_mm / n_cat
    return CreditLossProjection(
        timeline=_TIMELINE,
        categories=_CATEGORIES,
        scenario_name="baseline",
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


def _run_bank(opening, net_loans_mm, config=None):
    credit_projection = _credit_projection_for(opening, net_loans_mm)
    return run_bank_model(opening, credit_projection, config or BankConfig(), _TIMELINE)


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


def test_mark_accretion_sums_to_the_original_mark_over_its_life():
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
    acquirer_result = _run_bank(acquirer_opening, 2000.0)
    target_result = _run_bank(target_opening, 784.0)

    # life = 2 years = the FULL 8-quarter projection horizon, so the schedule should sum to
    # exactly the original mark amount within this projection.
    config = _deal_config(
        rate_mark=RateMarkConfig(rate_mark_pct=-0.02, rate_mark_life_years=2.0),
        securities_mark=SecuritiesMarkConfig(
            securities_mark_pct=-0.01, securities_mark_life_years=2.0
        ),
    )
    marks = compute_fair_value_marks(target_opening, 800.0, 16.0, config)
    sources_and_uses = compute_sources_and_uses(target_opening, marks, config)

    projection = compute_pro_forma_projection(
        acquirer_result=acquirer_result,
        target_result=target_result,
        marks=marks,
        sources_and_uses=sources_and_uses,
        pro_forma_equity_at_close_mm=1000.0,
        pro_forma_goodwill_at_close_mm=100.0,
        pro_forma_other_intangibles_at_close_mm=50.0,
        acquirer_bank_config=BankConfig(),
        config=config,
    )

    expected_total_accretion = -(marks.rate_mark_mm + marks.securities_mark_mm)
    assert projection.mark_accretion_mm.sum() == pytest.approx(expected_total_accretion)
    assert projection.mark_accretion_mm[0] == pytest.approx(0.0)  # no flow at the close instant


def test_cdi_amortization_sums_to_the_original_cdi_over_its_life():
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
    acquirer_result = _run_bank(acquirer_opening, 2000.0)
    target_result = _run_bank(target_opening, 784.0)

    config = _deal_config(cdi=CdiConfig(cdi_pct_of_core_deposits=0.02, cdi_amortization_years=2.0))
    marks = compute_fair_value_marks(target_opening, 800.0, 16.0, config)
    sources_and_uses = compute_sources_and_uses(target_opening, marks, config)

    projection = compute_pro_forma_projection(
        acquirer_result=acquirer_result,
        target_result=target_result,
        marks=marks,
        sources_and_uses=sources_and_uses,
        pro_forma_equity_at_close_mm=1000.0,
        pro_forma_goodwill_at_close_mm=100.0,
        pro_forma_other_intangibles_at_close_mm=50.0 + marks.cdi_gross_mm,
        acquirer_bank_config=BankConfig(),
        config=config,
    )

    assert projection.cdi_amortization_mm.sum() == pytest.approx(marks.cdi_gross_mm)
    assert projection.cdi_amortization_mm[0] == pytest.approx(0.0)
    # other_intangibles_mm declines by exactly the cumulative CDI amortization
    assert projection.other_intangibles_mm[-1] == pytest.approx(
        50.0 + marks.cdi_gross_mm - projection.cdi_amortization_mm.sum()
    )


def test_one_time_charges_hit_only_period_one():
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
    acquirer_result = _run_bank(acquirer_opening, 2000.0)
    target_result = _run_bank(target_opening, 784.0)

    from corefin.ma.schema import CostSaveConfig

    config = _deal_config(cost_saves=CostSaveConfig(restructuring_charge_mm=5.0))
    marks = compute_fair_value_marks(target_opening, 800.0, 16.0, config)
    sources_and_uses = compute_sources_and_uses(target_opening, marks, config)

    projection = compute_pro_forma_projection(
        acquirer_result=acquirer_result,
        target_result=target_result,
        marks=marks,
        sources_and_uses=sources_and_uses,
        pro_forma_equity_at_close_mm=1000.0,
        pro_forma_goodwill_at_close_mm=100.0,
        pro_forma_other_intangibles_at_close_mm=50.0,
        acquirer_bank_config=BankConfig(),
        config=config,
    )

    expected_one_time = marks.day2_allowance_non_pcd_mm + 5.0
    assert projection.one_time_charges_pretax_mm[1] == pytest.approx(expected_one_time)
    assert np.allclose(np.delete(projection.one_time_charges_pretax_mm, 1), 0.0)


def test_net_income_available_to_common_is_zero_at_close():
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
    acquirer_result = _run_bank(acquirer_opening, 2000.0)
    target_result = _run_bank(target_opening, 784.0)

    config = _deal_config()
    marks = compute_fair_value_marks(target_opening, 800.0, 16.0, config)
    sources_and_uses = compute_sources_and_uses(target_opening, marks, config)

    projection = compute_pro_forma_projection(
        acquirer_result=acquirer_result,
        target_result=target_result,
        marks=marks,
        sources_and_uses=sources_and_uses,
        pro_forma_equity_at_close_mm=1000.0,
        pro_forma_goodwill_at_close_mm=100.0,
        pro_forma_other_intangibles_at_close_mm=50.0,
        acquirer_bank_config=BankConfig(),
        config=config,
    )
    assert projection.net_income_available_to_common_mm[0] == pytest.approx(0.0)
    assert projection.equity_mm[0] == pytest.approx(1000.0)
