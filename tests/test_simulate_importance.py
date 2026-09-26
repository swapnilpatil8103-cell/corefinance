import pytest

from corefin.assumptions.schema import RootConfig
from corefin.optimize.structure import build_structure
from corefin.simulate.engine import run_simulation
from corefin.simulate.importance import (
    DRIVER_NAMES,
    compute_driver_importance,
    compute_tornado_chart,
)
from corefin.timeline import Timeline
from tests.conftest import minimal_config_dict


def _config(n_periods: int = 6, **scenario_overrides) -> RootConfig:
    data = minimal_config_dict(n_periods=n_periods)
    data["optimizer"] = {
        "decision_variables": [{"tranche_name": "TLB", "min_multiple": 1.0, "max_multiple": 3.0}],
    }
    data["scenario"].update(scenario_overrides)
    return RootConfig.model_validate(data)


def _result(n_scenarios: int = 3000, seed: int = 11):
    config = _config(
        driver_vol={
            "revenue_growth_std": 0.08,
            "ebitda_margin_std": 0.03,
            "exit_multiple_std": 1.2,
            "base_rate_std": 0.015,
        }
    )
    timeline = Timeline.annual(n_periods=6)
    candidate = build_structure(config, {"TLB": 2.0})
    result = run_simulation(config, candidate, timeline, n_scenarios=n_scenarios, seed=seed)
    return config, candidate, timeline, result


def test_driver_importance_returns_one_entry_per_driver_in_order():
    _, _, _, result = _result()
    importance = compute_driver_importance(result)
    assert [i.name for i in importance] == list(DRIVER_NAMES)


def test_zero_variance_drivers_report_exactly_zero_importance():
    """capex_pct_revenue and nwc_pct_revenue have no configured vol --
    regression coefficient and rank correlation must be exactly 0, not a
    spurious huge value amplified from floating-point-noise-scale std."""
    _, _, _, result = _result()
    importance = {i.name: i for i in compute_driver_importance(result)}
    for name in ("capex_pct_revenue", "nwc_pct_revenue"):
        assert importance[name].standardized_coefficient == pytest.approx(0.0, abs=1e-9)
        assert importance[name].rank_correlation == 0.0


def test_revenue_growth_and_exit_multiple_are_positively_associated_with_irr():
    _, _, _, result = _result()
    importance = {i.name: i for i in compute_driver_importance(result)}
    for name in ("revenue_growth", "exit_multiple"):
        assert importance[name].standardized_coefficient > 0
        assert importance[name].rank_correlation > 0


def test_revenue_growth_is_the_most_important_driver_when_its_vol_dominates():
    config = _config(
        driver_vol={
            "revenue_growth_std": 0.15,
            "ebitda_margin_std": 0.005,
            "exit_multiple_std": 0.1,
            "base_rate_std": 0.001,
        }
    )
    timeline = Timeline.annual(n_periods=6)
    candidate = build_structure(config, {"TLB": 2.0})
    result = run_simulation(config, candidate, timeline, n_scenarios=3000, seed=3)
    importance = compute_driver_importance(result)
    most_important = max(importance, key=lambda i: abs(i.rank_correlation))
    assert most_important.name == "revenue_growth"


def test_rank_correlation_is_bounded():
    _, _, _, result = _result()
    for imp in compute_driver_importance(result):
        assert -1.0 <= imp.rank_correlation <= 1.0


def test_tornado_chart_returns_one_bar_per_driver_with_shared_base_irr():
    config, candidate, timeline, result = _result()
    bars = compute_tornado_chart(config, candidate, timeline, result)
    assert [b.name for b in bars] == list(DRIVER_NAMES)
    base_irrs = {b.base_irr for b in bars}
    assert len(base_irrs) == 1  # same deterministic base case for every driver


def test_tornado_chart_revenue_growth_and_exit_multiple_p90_beats_p10():
    config, candidate, timeline, result = _result()
    bars = {b.name: b for b in compute_tornado_chart(config, candidate, timeline, result)}
    for name in ("revenue_growth", "exit_multiple"):
        assert bars[name].irr_at_p90 > bars[name].irr_at_p10
        assert bars[name].irr_range > 0


def test_tornado_chart_zero_variance_driver_has_zero_range():
    config, candidate, timeline, result = _result()
    bars = {b.name: b for b in compute_tornado_chart(config, candidate, timeline, result)}
    for name in ("capex_pct_revenue", "nwc_pct_revenue"):
        assert bars[name].irr_range == pytest.approx(0.0, abs=1e-9)
        assert bars[name].irr_at_p10 == pytest.approx(bars[name].base_irr, abs=1e-9)
        assert bars[name].irr_at_p90 == pytest.approx(bars[name].base_irr, abs=1e-9)


def test_tornado_chart_base_irr_matches_plain_deterministic_run():
    from corefin.scenarios.generator import deterministic_drivers
    from corefin.simulate.importance import _deterministic_irr

    config, candidate, timeline, result = _result()
    bars = compute_tornado_chart(config, candidate, timeline, result)
    base_drivers = deterministic_drivers(config, timeline)
    expected = _deterministic_irr(config, candidate, timeline, base_drivers)
    assert bars[0].base_irr == pytest.approx(expected)
