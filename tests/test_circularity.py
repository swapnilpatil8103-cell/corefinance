import numpy as np
import pytest

from corefin.assumptions.schema import InterestMode, RootConfig
from corefin.debt.circularity import CircularityNotConvergedError, run_debt_schedule
from corefin.scenarios.generator import deterministic_drivers
from corefin.statements.balance_sheet import compute_nwc
from corefin.statements.cash_flow import compute_delta
from corefin.statements.income_statement import compute_revenue
from corefin.timeline import Timeline
from tests.conftest import minimal_config_dict


def _debt_inputs(config: RootConfig, timeline: Timeline):
    drivers = deterministic_drivers(config, timeline)
    revenue = compute_revenue(config.company.revenue_base_mm, drivers.revenue_growth)
    ebitda = revenue * drivers.ebitda_margin
    from corefin.assumptions.loader import expand_series

    da_pct = expand_series(config.company.da_pct_revenue, timeline.n_periods).reshape(1, -1)
    da = revenue * da_pct
    ebit = ebitda - da
    capex = revenue * drivers.capex_pct_revenue
    nwc = compute_nwc(revenue, drivers.nwc_pct_revenue)
    delta_nwc = compute_delta(nwc, config.opening_balance_sheet.nwc_mm)
    return drivers, ebit, da, capex, delta_nwc


def _run(config: RootConfig, timeline: Timeline):
    drivers, ebit, da, capex, delta_nwc = _debt_inputs(config, timeline)
    return run_debt_schedule(
        tranches=config.tranches,
        waterfall_config=config.waterfall,
        timeline=timeline,
        drivers=drivers,
        ebit=ebit,
        da=da,
        capex=capex,
        delta_nwc=delta_nwc,
        tax_rate=config.company.tax_rate,
        nol_beginning_mm=config.company.nol_beginning_balance_mm,
        dividend_pct_of_ni=config.company.dividend_pct_of_ni,
        cash_beginning_mm=config.opening_balance_sheet.cash_mm,
    )


def test_average_balance_mode_converges():
    data = minimal_config_dict(n_periods=5)
    config = RootConfig.model_validate(data)
    timeline = Timeline.annual(n_periods=5)
    result = _run(config, timeline)
    assert np.all(result.iterations_used > 0)
    assert np.all(result.iterations_used <= config.waterfall.circularity_max_iterations)


def test_average_balance_matches_beginning_balance_when_rates_are_zero():
    data = minimal_config_dict(n_periods=5)
    for tranche in data["tranches"]:
        if tranche["rate_type"] == "floating":
            tranche["spread"] = 0.0
    data["scenario"]["base_rate"] = 0.0
    data["tranches"][0]["commitment_fee_pct"] = 0.0

    data_avg = {**data, "waterfall": {**data["waterfall"], "interest_mode": "average_balance"}}
    data_beg = {**data, "waterfall": {**data["waterfall"], "interest_mode": "beginning_balance"}}
    timeline = Timeline.annual(n_periods=5)

    result_avg = _run(RootConfig.model_validate(data_avg), timeline)
    result_beg = _run(RootConfig.model_validate(data_beg), timeline)

    for name in result_avg.ending_balance:
        np.testing.assert_allclose(
            result_avg.ending_balance[name], result_beg.ending_balance[name], atol=1e-9
        )
    np.testing.assert_allclose(result_avg.ending_cash, result_beg.ending_cash, atol=1e-9)


def test_beginning_balance_mode_converges_in_two_iterations():
    data = minimal_config_dict(n_periods=4)
    data["waterfall"]["interest_mode"] = "beginning_balance"
    config = RootConfig.model_validate(data)
    timeline = Timeline.annual(n_periods=4)
    result = _run(config, timeline)
    assert np.all(result.iterations_used <= 2)


def test_non_convergence_raises_clear_error():
    data = minimal_config_dict(n_periods=2)
    data["waterfall"]["circularity_max_iterations"] = 1
    data["waterfall"]["circularity_tolerance"] = 1e-15
    data["tranches"][1]["spread"] = 5.0
    data["scenario"]["base_rate"] = 2.0
    config = RootConfig.model_validate(data)
    timeline = Timeline.annual(n_periods=2)
    with pytest.raises(CircularityNotConvergedError):
        _run(config, timeline)


def test_interest_mode_toggle_is_respected():
    data = minimal_config_dict(n_periods=3)
    data["waterfall"]["interest_mode"] = "average_balance"
    config = RootConfig.model_validate(data)
    assert config.waterfall.interest_mode is InterestMode.AVERAGE_BALANCE
