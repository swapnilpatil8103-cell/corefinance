import time

import numpy as np
import pytest

from corefin.assumptions.schema import RootConfig
from corefin.checks.corporate_checks import check_balance_sheet_balances
from corefin.checks.framework import run_checks
from corefin.optimize.structure import build_structure, compute_entry_ebitda_mm
from corefin.simulate.compare import (
    bump_decision_values,
    compare_structures,
    input_config_decision_values,
)
from corefin.timeline import Timeline
from tests.conftest import minimal_config_dict


def _config(n_periods: int = 6, **scenario_overrides) -> RootConfig:
    data = minimal_config_dict(n_periods=n_periods)
    data["optimizer"] = {
        "decision_variables": [{"tranche_name": "TLB", "min_multiple": 1.0, "max_multiple": 5.0}],
    }
    data["scenario"].update(scenario_overrides)
    return RootConfig.model_validate(data)


def test_input_config_decision_values_matches_configured_tranche_size():
    config = _config()
    values = input_config_decision_values(config)
    tlb = next(t for t in config.tranches if t.name == "TLB")
    entry_ebitda_mm = compute_entry_ebitda_mm(config)
    assert values == {"TLB": pytest.approx(tlb.size_mm / entry_ebitda_mm)}


def test_input_config_decision_values_requires_optimizer_config():
    data = minimal_config_dict(n_periods=5)
    config = RootConfig.model_validate(data)
    with pytest.raises(ValueError, match="optimizer"):
        input_config_decision_values(config)


def test_bump_decision_values_only_changes_the_named_tranche():
    values = {"TLB": 2.0, "Notes": 1.0}
    bumped = bump_decision_values(values, "TLB", 1.0)
    assert bumped == {"TLB": 3.0, "Notes": 1.0}
    assert values == {"TLB": 2.0, "Notes": 1.0}  # original untouched


def test_bump_decision_values_rejects_unknown_tranche_name():
    with pytest.raises(ValueError, match="Notes"):
        bump_decision_values({"TLB": 2.0}, "Notes", 1.0)


def test_compare_structures_returns_one_entry_per_named_structure_in_order():
    config = _config(driver_vol={"revenue_growth_std": 0.04})
    timeline = Timeline.annual(n_periods=6)
    named = {"A": {"TLB": 1.5}, "B": {"TLB": 2.5}, "C": {"TLB": 3.5}}
    entries = compare_structures(config, timeline, named, n_scenarios=200, seed=1)
    assert [e.name for e in entries] == ["A", "B", "C"]


def test_compare_structures_uses_common_random_numbers():
    """Operating performance (revenue/EBITDA drivers) must be identical
    across differently-levered structures -- financing structure never
    affects it, so any difference in these arrays would mean the random
    draws diverged instead of being shared."""
    config = _config(driver_vol={"revenue_growth_std": 0.05, "ebitda_margin_std": 0.02})
    timeline = Timeline.annual(n_periods=6)
    named = {"Low": {"TLB": 1.5}, "High": {"TLB": 3.5}}
    entries = compare_structures(config, timeline, named, n_scenarios=500, seed=3)
    low, high = entries
    np.testing.assert_array_equal(
        low.simulation.drivers.revenue_growth, high.simulation.drivers.revenue_growth
    )
    np.testing.assert_array_equal(
        low.simulation.drivers.ebitda_margin, high.simulation.drivers.ebitda_margin
    )
    np.testing.assert_array_equal(
        low.simulation.drivers.exit_multiple, high.simulation.drivers.exit_multiple
    )


def test_more_leverage_raises_mean_irr_and_downside_risk_via_compare_structures():
    config = _config(
        driver_vol={"revenue_growth_std": 0.06, "ebitda_margin_std": 0.03, "exit_multiple_std": 0.8}
    )
    timeline = Timeline.annual(n_periods=6)
    named = {"Low": {"TLB": 1.0}, "High": {"TLB": 3.0}}
    entries = {
        e.name: e for e in compare_structures(config, timeline, named, n_scenarios=3000, seed=42)
    }

    low_returns = entries["Low"].downside.returns
    high_returns = entries["High"].downside.returns
    assert high_returns.mean_irr > low_returns.mean_irr
    assert high_returns.expected_shortfall_irr_10pct < low_returns.expected_shortfall_irr_10pct


def test_compare_structures_includes_stress_results_for_each_structure():
    data = minimal_config_dict(n_periods=6)
    data["optimizer"] = {
        "decision_variables": [{"tranche_name": "TLB", "min_multiple": 1.0, "max_multiple": 5.0}]
    }
    data["simulate"] = {
        "stress_scenarios": [
            {
                "name": "Recession",
                "shocks": {"recession": {"start_year_index": 2}},
            }
        ]
    }
    config = RootConfig.model_validate(data)
    timeline = Timeline.annual(n_periods=6)
    named = {"Low": {"TLB": 1.0}, "High": {"TLB": 3.0}}
    entries = compare_structures(config, timeline, named, n_scenarios=200, seed=1)
    for entry in entries:
        assert [s.name for s in entry.stress_results] == ["Recession"]


def test_compare_structures_bumped_variant_is_more_levered_than_the_original():
    config = _config()
    original_values = input_config_decision_values(config)
    bumped_values = bump_decision_values(original_values, "TLB", 1.0)
    original = build_structure(config, original_values)
    bumped = build_structure(config, bumped_values)
    assert bumped.decision_values["TLB"] == pytest.approx(original.decision_values["TLB"] + 1.0)
    assert bumped.leverage.total_leverage > original.leverage.total_leverage


@pytest.mark.slow
def test_ten_thousand_scenarios_with_regimes_and_fat_tails_two_structures_within_time_budget():
    config = _config(
        n_periods=7,
        driver_vol={
            "revenue_growth_std": 0.04,
            "ebitda_margin_std": 0.02,
            "base_rate_std": 0.01,
            "exit_multiple_std": 0.5,
        },
        advanced={
            "regime": {"annual_probability": 0.15, "duration_years": 2},
            "fat_tails": {"degrees_of_freedom": 5.0},
        },
    )
    timeline = Timeline.annual(n_periods=7)
    named = {"Low": {"TLB": 1.5}, "High": {"TLB": 3.5}}

    start = time.perf_counter()
    entries = compare_structures(config, timeline, named, n_scenarios=10_000, seed=1)
    elapsed = time.perf_counter() - start

    assert elapsed < 35.0, (
        f"10k scenarios x 2 structures with regimes/fat tails took {elapsed:.2f}s"
    )
    for entry in entries:
        assert entry.simulation.exit_result.irr.shape == (10_000,)
        run_checks([check_balance_sheet_balances(entry.simulation.model_result.balance_sheet)])
