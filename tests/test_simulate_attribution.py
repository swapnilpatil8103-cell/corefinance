import numpy as np
import pytest

from corefin.assumptions.schema import RootConfig
from corefin.optimize.structure import build_structure
from corefin.simulate.attribution import (
    compute_value_creation_bridge,
    summarize_value_creation_bridge,
)
from corefin.simulate.engine import run_simulation
from corefin.timeline import Timeline
from tests.conftest import minimal_config_dict


def _config(n_periods: int = 6, **scenario_overrides) -> RootConfig:
    data = minimal_config_dict(n_periods=n_periods)
    data["optimizer"] = {
        "decision_variables": [{"tranche_name": "TLB", "min_multiple": 1.0, "max_multiple": 3.0}],
    }
    data["scenario"].update(scenario_overrides)
    return RootConfig.model_validate(data)


def _high_vol_result(n_scenarios: int = 3000, tlb_multiple: float = 2.5, seed: int = 7):
    config = _config(
        driver_vol={
            "revenue_growth_std": 0.10,
            "ebitda_margin_std": 0.05,
            "exit_multiple_std": 1.5,
            "base_rate_std": 0.01,
        }
    )
    timeline = Timeline.annual(n_periods=6)
    candidate = build_structure(config, {"TLB": tlb_multiple})
    result = run_simulation(config, candidate, timeline, n_scenarios=n_scenarios, seed=seed)
    return config, result


def test_bridge_components_sum_exactly_to_total_value_creation():
    """High vol + high leverage, chosen so the sample includes at least one
    wipeout (MOIC ~ 0, realized equity floored at zero) -- the identity
    must still hold exactly there, not just in the common case."""
    config, result = _high_vol_result(tlb_multiple=3.0)
    assert np.any(result.exit_result.moic < 0.05)  # sanity: sample does include a near-wipeout

    bridge = compute_value_creation_bridge(config, result)
    lhs = (
        bridge.ebitda_growth
        + bridge.multiple_change
        + bridge.debt_paydown_and_cash
        + bridge.fees_and_leakage
    )
    np.testing.assert_allclose(lhs, bridge.total_value_creation, atol=1e-6)


def test_total_value_creation_matches_its_definition():
    config, result = _high_vol_result()
    bridge = compute_value_creation_bridge(config, result)
    exit_idx = config.transaction.exit_year_index
    dividends = np.sum(result.model_result.cash_flow_statement.dividends[:, : exit_idx + 1], axis=1)
    expected = (
        result.exit_result.realized_exit_equity_value
        + dividends
        - result.candidate.sources_and_uses.sponsor_equity_mm
    )
    np.testing.assert_allclose(bridge.total_value_creation, expected, atol=1e-9)


def test_fees_bucket_is_constant_and_matches_sources_and_uses():
    config, result = _high_vol_result()
    bridge = compute_value_creation_bridge(config, result)
    sau = result.candidate.sources_and_uses
    expected = -(sau.transaction_fees_mm + sau.financing_fees_and_oid_mm)
    assert np.all(bridge.fees_and_leakage == pytest.approx(expected))


def test_ebitda_growth_and_multiple_change_sum_to_enterprise_value_change():
    config, result = _high_vol_result()
    bridge = compute_value_creation_bridge(config, result)
    entry_ev = config.transaction.entry_multiple * result.candidate.entry_ebitda_mm
    exit_ev = result.exit_result.exit_enterprise_value
    np.testing.assert_allclose(
        bridge.ebitda_growth + bridge.multiple_change, exit_ev - entry_ev, atol=1e-6
    )


def test_summarize_bridge_returns_average_plus_three_percentile_scenarios():
    config, result = _high_vol_result()
    bridge = compute_value_creation_bridge(config, result)
    summaries = summarize_value_creation_bridge(bridge, result.exit_result.irr)
    assert [s.label for s in summaries] == ["Average", "P10", "Median", "P90"]


def test_summarize_bridge_percentile_scenarios_match_their_own_irr_rank():
    config, result = _high_vol_result(n_scenarios=2000)
    bridge = compute_value_creation_bridge(config, result)
    irr = result.exit_result.irr
    summaries = {s.label: s for s in summarize_value_creation_bridge(bridge, irr)[1:]}

    for label, pct in (("P10", 10), ("Median", 50), ("P90", 90)):
        target = np.percentile(irr, pct)
        idx = int(np.argmin(np.abs(irr - target)))
        expected_total = float(bridge.total_value_creation[idx])
        assert summaries[label].total_value_creation == pytest.approx(expected_total)


def test_average_bridge_equals_mean_of_per_scenario_components():
    config, result = _high_vol_result()
    bridge = compute_value_creation_bridge(config, result)
    average = summarize_value_creation_bridge(bridge, result.exit_result.irr)[0]
    assert average.ebitda_growth == pytest.approx(float(np.mean(bridge.ebitda_growth)))
    assert average.total_value_creation == pytest.approx(
        float(np.mean(bridge.total_value_creation))
    )
