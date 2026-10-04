"""Offline tests for corefin.ma.excel_export -- synthetic data only."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from corefin.bank.model import run_bank_model
from corefin.bank.schema import BankConfig, BankOpeningBalance
from corefin.credit.interface import CreditLossProjection
from corefin.ma.accretion import compute_acquirer_irr_sensitivity
from corefin.ma.excel_export import export_deal_to_excel
from corefin.ma.model import run_deal_model
from corefin.ma.schema import ConsiderationConfig, CreditMarkConfig, DealConfig
from corefin.ma.sensitivity import (
    DealContext,
    compute_price_cost_save_grid,
    compute_tornado,
    run_monte_carlo,
)
from corefin.ma.stress import run_stress_test
from corefin.timeline import Timeline

_BASELINE_TIMELINE = Timeline.quarterly(9, n_historical=1, start_year=2025, start_quarter=4)
_STRESS_TIMELINE = Timeline.quarterly(10, n_historical=1, start_year=2025, start_quarter=4)
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


def _credit_projection_for(opening, net_loans_mm, timeline):
    n_cat = len(_CATEGORIES)
    n = timeline.n_periods
    per_category = net_loans_mm / n_cat
    return CreditLossProjection(
        timeline=timeline,
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


@pytest.fixture
def full_pipeline():
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
    acquirer_baseline_proj = _credit_projection_for(acquirer_opening, 2000.0, _BASELINE_TIMELINE)
    target_baseline_proj = _credit_projection_for(target_opening, 784.0, _BASELINE_TIMELINE)
    acquirer_stress_proj = _credit_projection_for(acquirer_opening, 2000.0, _STRESS_TIMELINE)
    target_stress_proj = _credit_projection_for(target_opening, 784.0, _STRESS_TIMELINE)

    acquirer_bank_config = BankConfig()
    target_bank_config = BankConfig()
    acquirer_result = run_bank_model(
        acquirer_opening, acquirer_baseline_proj, acquirer_bank_config, _BASELINE_TIMELINE
    )
    target_result = run_bank_model(
        target_opening, target_baseline_proj, target_bank_config, _BASELINE_TIMELINE
    )

    config = DealConfig(
        consideration=ConsiderationConfig(
            price_to_tbv=1.5,
            stock_pct=0.8,
            acquirer_share_price=25.0,
            acquirer_shares_outstanding_mm=50.0,
        ),
        credit_mark=CreditMarkConfig(credit_mark_pct=0.02, pcd_share=0.3),
        deal_horizon_quarters=8,
    )
    deal_result = run_deal_model(
        acquirer_result, target_result, acquirer_bank_config, target_bank_config, config
    )

    irr_by_multiple = compute_acquirer_irr_sensitivity(
        acquirer_result,
        deal_result.pro_forma_projection,
        deal_result.sources_and_uses,
        deal_result.pro_forma_cet1_bridge.pro_forma_cet1_mm,
        deal_result.pro_forma_capital_ratios.pro_forma_rwa_mm,
        config,
        exit_multiples=[1.0, 1.5, 2.0],
    )

    stress_baseline = run_stress_test(
        acquirer_opening,
        target_opening,
        acquirer_baseline_proj,
        target_baseline_proj,
        acquirer_bank_config,
        target_bank_config,
        deal_result,
        config,
        acquirer_unexplained_cet1_residual_mm=0.0,
        n_quarters=8,
    )
    stress_adverse = run_stress_test(
        acquirer_opening,
        target_opening,
        acquirer_stress_proj,
        target_stress_proj,
        acquirer_bank_config,
        target_bank_config,
        deal_result,
        config,
        acquirer_unexplained_cet1_residual_mm=0.0,
        n_quarters=9,
    )
    stress_results = {"baseline": stress_baseline, "severely_adverse": stress_adverse}

    ctx = DealContext(
        acquirer_opening=acquirer_opening,
        target_opening=target_opening,
        acquirer_bank_config=acquirer_bank_config,
        target_bank_config=target_bank_config,
        acquirer_baseline_result=acquirer_result,
        target_baseline_result=target_result,
        acquirer_baseline_projection=acquirer_baseline_proj,
        target_baseline_projection=target_baseline_proj,
        acquirer_severely_adverse_projection=acquirer_stress_proj,
        target_severely_adverse_projection=target_stress_proj,
        baseline_rate_path_pp=np.full(_BASELINE_TIMELINE.n_periods, 3.7),
        severely_adverse_rate_path_pp=np.full(_STRESS_TIMELINE.n_periods, 3.7),
        base_config=config,
        acquirer_unexplained_cet1_residual_mm=0.0,
    )
    tornado = compute_tornado(ctx)
    grid = compute_price_cost_save_grid(ctx, np.array([1.2, 1.5, 1.8]), np.array([0.1, 0.2]))
    mc = run_monte_carlo(ctx, n_draws=10, seed=0)

    return dict(
        config=config,
        deal_result=deal_result,
        irr_by_multiple=irr_by_multiple,
        stress_results=stress_results,
        tornado=tornado,
        grid=grid,
        mc=mc,
    )


def test_export_deal_to_excel_writes_every_expected_sheet(full_pipeline, tmp_path):
    path = tmp_path / "deal.xlsx"
    p = full_pipeline
    export_deal_to_excel(
        str(path),
        "Acquirer Bank A",
        "Target Bank B",
        p["config"],
        p["deal_result"].pro_forma_capital_ratios,
        p["deal_result"].eps_accretion,
        p["deal_result"].tbv_earnback,
        p["irr_by_multiple"],
        p["stress_results"],
        p["tornado"],
        p["grid"],
        p["mc"],
    )
    assert path.exists()
    workbook = pd.ExcelFile(path)
    expected_sheets = {
        "Summary",
        "EPS Accretion",
        "TBV & IRR",
        "Stress Test",
        "Tornado",
        "Grid - Year 2 Accretion",
        "Grid - Earnback (yrs)",
        "Monte Carlo - Percentiles",
        "Monte Carlo - Draws",
    }
    assert expected_sheets.issubset(set(workbook.sheet_names))


def test_summary_sheet_reports_the_given_labels_not_hardcoded_names(full_pipeline, tmp_path):
    path = tmp_path / "deal.xlsx"
    p = full_pipeline
    export_deal_to_excel(
        str(path),
        "Acquirer Bank A",
        "Target Bank B",
        p["config"],
        p["deal_result"].pro_forma_capital_ratios,
        p["deal_result"].eps_accretion,
        p["deal_result"].tbv_earnback,
        p["irr_by_multiple"],
        p["stress_results"],
        p["tornado"],
        p["grid"],
        p["mc"],
    )
    summary = pd.read_excel(path, sheet_name="Summary", index_col=0)
    assert summary.loc["Acquirer", "Value"] == "Acquirer Bank A"
    assert summary.loc["Target", "Value"] == "Target Bank B"


def test_monte_carlo_draws_sheet_has_one_row_per_draw(full_pipeline, tmp_path):
    path = tmp_path / "deal.xlsx"
    p = full_pipeline
    export_deal_to_excel(
        str(path),
        "Acquirer Bank A",
        "Target Bank B",
        p["config"],
        p["deal_result"].pro_forma_capital_ratios,
        p["deal_result"].eps_accretion,
        p["deal_result"].tbv_earnback,
        p["irr_by_multiple"],
        p["stress_results"],
        p["tornado"],
        p["grid"],
        p["mc"],
    )
    draws = pd.read_excel(path, sheet_name="Monte Carlo - Draws")
    assert len(draws) == p["mc"].n_draws
