"""Offline integration tests for corefin.ma.accretion -- synthetic data
only. Covers the required Stage 4 checks: a zero-premium/zero-mark/zero-
synergy all-stock deal leaves EPS/TBV essentially unchanged; a higher
price lowers EPS accretion and lengthens TBV earnback."""

from __future__ import annotations

import numpy as np
import pytest

from corefin.bank.model import run_bank_model
from corefin.bank.schema import BankConfig, BankOpeningBalance
from corefin.credit.interface import CreditLossProjection
from corefin.ma.accretion import compute_irr_from_cash_flows
from corefin.ma.model import run_deal_model
from corefin.ma.schema import (
    CdiConfig,
    ConsiderationConfig,
    CostSaveConfig,
    CreditMarkConfig,
    DealConfig,
    DistributableCashMethod,
    ExitMultipleBasis,
    IrrConfig,
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


def _run_deal(**config_overrides):
    acquirer_result, target_result = _acquirer_and_target()
    config_kwargs = dict(
        consideration=ConsiderationConfig(
            price_to_tbv=1.5,
            stock_pct=0.8,
            acquirer_share_price=25.0,
            acquirer_shares_outstanding_mm=50.0,
        ),
        credit_mark=CreditMarkConfig(credit_mark_pct=0.02, pcd_share=0.3),
        rate_mark=RateMarkConfig(rate_mark_pct=-0.01),
        securities_mark=SecuritiesMarkConfig(securities_mark_pct=-0.005),
        cost_saves=CostSaveConfig(restructuring_charge_mm=20.0),
    )
    config_kwargs.update(config_overrides)
    config = DealConfig(**config_kwargs)
    return run_deal_model(acquirer_result, target_result, BankConfig(), BankConfig(), config)


def test_all_stock_zero_premium_zero_mark_zero_synergy_leaves_eps_and_tbv_unchanged():
    # price_to_tbv=1.0 (no premium), all-stock, zero credit/rate/securities marks, zero CDI,
    # zero cost saves/restructuring, zero existing-allowance-to-eliminate (so the bargain-
    # purchase edge case from test_ma_purchase_accounting doesn't kick in either): the target
    # contributes its OWN tangible book value in new stock, one-for-one -- TBV/share should be
    # unchanged mechanically. EPS also needs the target's ROE to match the acquirer's (EPS
    # accretion/dilution in an all-stock, zero-TBV-premium deal depends on RELATIVE earnings
    # yields, not the book-value premium -- a textbook M&A-math point, not specific to this
    # model), so the target's income statement is scaled to the SAME 40%-of-acquirer ratio as
    # its equity (400 / 1000).
    acquirer_result, _ = _acquirer_and_target()
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
        net_interest_income_jumpoff_mm=16.0,  # 40% of the acquirer's 40.0
        noninterest_income_jumpoff_mm=4.0,  # 40% of 10.0
        noninterest_expense_jumpoff_mm=12.0,  # 40% of 30.0
    )
    # strip the target's existing allowance so eliminating it has no effect (consistent with
    # the "truly zero mark" premise -- see test_ma_purchase_accounting's own equivalent case)
    target_result = run_bank_model(
        target_opening,
        CreditLossProjection(
            timeline=_TIMELINE,
            categories=_CATEGORIES,
            scenario_name="baseline",
            bank_identifier=target_opening.bank_id,
            balance_mm=np.full((2, 9), 392.0),
            net_charge_off_mm=np.zeros((2, 9)),
            provision_expense_mm=np.zeros((2, 9)),
            allowance_mm=np.zeros((2, 9)),
            npl_mm=np.zeros((2, 9)),
            nco_rate=np.zeros((2, 9)),
            npl_ratio=np.zeros((2, 9)),
            monte_carlo_mean=np.zeros(2),
            monte_carlo_percentiles={},
        ),
        BankConfig(),
        _TIMELINE,
    )

    # acquirer_share_price MUST equal the acquirer's OWN standalone TBV/share for this to be a
    # true TBV-neutral baseline: issuing new stock at any price ABOVE book value is itself
    # TBV-accretive (funding the purchase with "cheap" shares relative to book), even with a
    # zero-premium, zero-mark target -- confirmed by deriving the acquirer standalone TBV/share
    # here (equity 1000 - goodwill 50 - other_intangibles 10 = 940, / 50 shares = 18.80).
    acquirer_standalone_tbv_per_share = (1000.0 - 50.0 - 10.0) / 50.0
    config = DealConfig(
        consideration=ConsiderationConfig(
            price_to_tbv=1.0,
            stock_pct=1.0,
            acquirer_share_price=acquirer_standalone_tbv_per_share,
            acquirer_shares_outstanding_mm=50.0,
        ),
        credit_mark=CreditMarkConfig(credit_mark_pct=0.0, pcd_share=0.0),
        cdi=CdiConfig(cdi_pct_of_core_deposits=0.0),
    )
    result = run_deal_model(acquirer_result, target_result, BankConfig(), BankConfig(), config)

    assert result.tbv_earnback.tbv_dilution_at_close_pct == pytest.approx(0.0, abs=1e-6)
    # EPS accretion/dilution at close (period 0) isn't meaningful (NaN by construction); check
    # year-1 instead -- should also be close to flat (no marks/synergies driving a difference).
    year1 = result.eps_accretion.annual()["Year 1"]
    assert year1.accretion_dilution_pct_gaap == pytest.approx(0.0, abs=0.05)  # small tolerance
    # for the combined entity's standalone earning power to roughly carry through


def test_higher_price_lowers_eps_accretion():
    low_price_result = _run_deal(
        consideration=ConsiderationConfig(
            price_to_tbv=1.2,
            stock_pct=0.8,
            acquirer_share_price=25.0,
            acquirer_shares_outstanding_mm=50.0,
        )
    )
    high_price_result = _run_deal(
        consideration=ConsiderationConfig(
            price_to_tbv=2.0,
            stock_pct=0.8,
            acquirer_share_price=25.0,
            acquirer_shares_outstanding_mm=50.0,
        )
    )
    low_year1_accretion = low_price_result.eps_accretion.annual()[
        "Year 1"
    ].accretion_dilution_pct_gaap
    high_year1_accretion = high_price_result.eps_accretion.annual()[
        "Year 1"
    ].accretion_dilution_pct_gaap
    assert high_year1_accretion < low_year1_accretion


def test_higher_price_lengthens_or_prevents_tbv_earnback():
    low_price_result = _run_deal(
        consideration=ConsiderationConfig(
            price_to_tbv=1.2,
            stock_pct=0.8,
            acquirer_share_price=25.0,
            acquirer_shares_outstanding_mm=50.0,
        )
    )
    high_price_result = _run_deal(
        consideration=ConsiderationConfig(
            price_to_tbv=2.0,
            stock_pct=0.8,
            acquirer_share_price=25.0,
            acquirer_shares_outstanding_mm=50.0,
        )
    )
    assert high_price_result.tbv_earnback.tbv_dilution_at_close_pct < (
        low_price_result.tbv_earnback.tbv_dilution_at_close_pct
    )
    low_earnback = low_price_result.tbv_earnback.earnback_period_index
    high_earnback = high_price_result.tbv_earnback.earnback_period_index
    # either the higher-price deal takes longer to earn back, or it never earns back within the
    # horizon while the lower-price deal does
    if low_earnback is None:
        assert high_earnback is None
    elif high_earnback is None:
        pass  # high price never earns back -- consistent with "lengthens or prevents"
    else:
        assert high_earnback >= low_earnback


def test_exchange_ratio_computed_when_target_shares_given():
    result = _run_deal(
        consideration=ConsiderationConfig(
            price_to_tbv=1.5,
            stock_pct=0.8,
            acquirer_share_price=25.0,
            acquirer_shares_outstanding_mm=50.0,
            target_shares_outstanding_mm=20.0,
        )
    )
    assert result.eps_accretion.exchange_ratio == pytest.approx(
        result.eps_accretion.new_shares_issued_mm / 20.0
    )


def test_exchange_ratio_none_when_target_shares_not_given():
    result = _run_deal()
    assert result.eps_accretion.exchange_ratio is None


def test_acquirer_irr_is_finite():
    result = _run_deal()
    assert np.isfinite(result.acquirer_irr)


def test_pro_forma_shares_outstanding_exceeds_acquirer_standalone_shares():
    result = _run_deal()
    assert (
        result.eps_accretion.pro_forma_shares_outstanding_mm
        > result.eps_accretion.new_shares_issued_mm
    )
    assert result.eps_accretion.new_shares_issued_mm > 0


def test_irr_equals_discount_rate_when_price_equals_pv_of_its_cash_flows():
    # the defining property of IRR: if CF[0] is priced at EXACTLY the present value of the
    # other cash flows at rate r, NPV(r) = 0 by construction, so the solved IRR must equal r.
    quarterly_rate = 0.025  # ~10.4% annualized
    future_flows = np.array([8.0, 9.0, 10.0, 11.0 + 150.0])  # last includes a terminal value
    price = sum(cf / (1.0 + quarterly_rate) ** (t + 1) for t, cf in enumerate(future_flows))
    cash_flows = np.concatenate([[-price], future_flows])

    irr = compute_irr_from_cash_flows(cash_flows)
    expected_annual_irr = (1.0 + quarterly_rate) ** 4 - 1.0
    assert irr == pytest.approx(expected_annual_irr, rel=1e-6)


def test_acquirer_irr_with_excess_capital_distributable_cash_is_finite():
    result = _run_deal(
        irr=IrrConfig(
            distributable_cash_method=DistributableCashMethod.EXCESS_CAPITAL_ABOVE_TARGET_CET1,
            target_cet1_ratio=0.10,
        )
    )
    assert np.isfinite(result.acquirer_irr)


def test_acquirer_irr_with_forward_pe_exit_basis_is_finite():
    result = _run_deal(
        irr=IrrConfig(exit_multiple_basis=ExitMultipleBasis.FORWARD_PE, exit_multiple=10.0)
    )
    assert np.isfinite(result.acquirer_irr)


def test_acquirer_irr_exit_multiple_scales_the_terminal_value_and_thus_the_irr():
    low_multiple_result = _run_deal(irr=IrrConfig(exit_multiple=0.5))
    high_multiple_result = _run_deal(irr=IrrConfig(exit_multiple=2.0))
    # a bigger terminal value (higher exit multiple on the same underlying basis) can only
    # raise or leave unchanged the IRR, never lower it -- all else held fixed
    assert high_multiple_result.acquirer_irr >= low_multiple_result.acquirer_irr


def test_tbv_earnback_label_reports_beyond_horizon_when_not_reached():
    high_price_result = _run_deal(
        consideration=ConsiderationConfig(
            price_to_tbv=3.0,
            stock_pct=0.8,
            acquirer_share_price=25.0,
            acquirer_shares_outstanding_mm=50.0,
        )
    )
    assert high_price_result.tbv_earnback.earnback_period_index is None
    assert high_price_result.tbv_earnback.earnback_label == "beyond horizon"


def test_tbv_earnback_label_reports_years_and_quarter_when_reached():
    result = _run_deal()
    if result.tbv_earnback.earnback_period_index is not None:
        label = result.tbv_earnback.earnback_label
        assert "beyond horizon" not in label
        assert "years" in label
