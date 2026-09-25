import numpy as np
import pytest

from corefin.assumptions.schema import (
    MultipleCompressionStressConfig,
    NamedStressScenarioConfig,
    RateShockStressConfig,
    RecessionStressConfig,
    RootConfig,
    StressShockConfig,
)
from corefin.checks.corporate_checks import check_balance_sheet_balances, check_cash_ties_to_cfs
from corefin.checks.debt_checks import check_debt_rollforward
from corefin.checks.framework import run_checks
from corefin.optimize.structure import build_structure
from corefin.scenarios.generator import deterministic_drivers
from corefin.simulate.stress import apply_stress_shock, run_stress_scenario, run_stress_scenarios
from corefin.statements.corporate_model import run_corporate_model_with_debt
from corefin.timeline import Timeline
from tests.conftest import minimal_config_dict


def _config(n_periods: int = 6) -> RootConfig:
    data = minimal_config_dict(n_periods=n_periods)
    data["optimizer"] = {
        "decision_variables": [{"tranche_name": "TLB", "min_multiple": 1.0, "max_multiple": 3.0}],
    }
    return RootConfig.model_validate(data)


def _baseline_scenario() -> NamedStressScenarioConfig:
    return NamedStressScenarioConfig(name="Base", shocks=StressShockConfig())


def test_recession_shock_peaks_at_start_year_and_tapers_to_zero():
    config = _config(n_periods=6)
    timeline = Timeline.annual(n_periods=6)
    base = deterministic_drivers(config, timeline)
    shock = StressShockConfig(
        recession=RecessionStressConfig(
            start_year_index=2, revenue_growth_hit=-0.10, ebitda_margin_hit=-0.03, recovery_years=2
        )
    )
    shocked = apply_stress_shock(base, timeline, shock)
    base_growth = base.revenue_growth[0]
    shocked_growth = shocked.revenue_growth[0]

    np.testing.assert_allclose(shocked_growth[:2], base_growth[:2])  # untouched before the shock
    assert shocked_growth[2] == pytest.approx(base_growth[2] - 0.10)  # full hit at start year
    assert shocked_growth[3] == pytest.approx(base_growth[3] - 0.05)  # half hit, one year in
    assert shocked_growth[4] == pytest.approx(base_growth[4])  # fully recovered by year 4
    assert shocked_growth[5] == pytest.approx(base_growth[5])  # stays recovered


def test_rate_shock_is_permanent_and_held():
    config = _config(n_periods=5)
    timeline = Timeline.annual(n_periods=5)
    base = deterministic_drivers(config, timeline)
    shock = StressShockConfig(rate_shock=RateShockStressConfig(start_year_index=1, bps=0.03))
    shocked = apply_stress_shock(base, timeline, shock)

    assert shocked.base_rate[0, 0] == pytest.approx(base.base_rate[0, 0])  # before start: untouched
    for t in range(1, 5):
        assert shocked.base_rate[0, t] == pytest.approx(base.base_rate[0, t] + 0.03)  # held


def test_multiple_compression_only_affects_exit_multiple():
    config = _config(n_periods=5)
    timeline = Timeline.annual(n_periods=5)
    base = deterministic_drivers(config, timeline)
    shock = StressShockConfig(multiple_compression=MultipleCompressionStressConfig(delta=-2.0))
    shocked = apply_stress_shock(base, timeline, shock)

    assert shocked.exit_multiple[0] == pytest.approx(base.exit_multiple[0] - 2.0)
    np.testing.assert_array_equal(shocked.revenue_growth, base.revenue_growth)
    np.testing.assert_array_equal(shocked.ebitda_margin, base.ebitda_margin)
    np.testing.assert_array_equal(shocked.base_rate, base.base_rate)


def test_each_shock_moves_irr_below_baseline():
    config = _config(n_periods=6)
    timeline = Timeline.annual(n_periods=6)
    candidate = build_structure(config, {"TLB": 2.0})

    baseline = run_stress_scenario(config, candidate, timeline, _baseline_scenario())
    recession = run_stress_scenario(
        config,
        candidate,
        timeline,
        NamedStressScenarioConfig(
            name="Recession",
            shocks=StressShockConfig(recession=RecessionStressConfig(start_year_index=2)),
        ),
    )
    rate = run_stress_scenario(
        config,
        candidate,
        timeline,
        NamedStressScenarioConfig(
            name="Rate", shocks=StressShockConfig(rate_shock=RateShockStressConfig())
        ),
    )
    multiple = run_stress_scenario(
        config,
        candidate,
        timeline,
        NamedStressScenarioConfig(
            name="Multiple",
            shocks=StressShockConfig(multiple_compression=MultipleCompressionStressConfig()),
        ),
    )
    for stressed in (recession, rate, multiple):
        assert stressed.irr < baseline.irr


def test_combined_downside_is_worse_than_any_single_shock_for_irr():
    config = _config(n_periods=6)
    timeline = Timeline.annual(n_periods=6)
    candidate = build_structure(config, {"TLB": 2.0})

    individual = [
        NamedStressScenarioConfig(
            name="Recession",
            shocks=StressShockConfig(recession=RecessionStressConfig(start_year_index=2)),
        ),
        NamedStressScenarioConfig(
            name="Rate", shocks=StressShockConfig(rate_shock=RateShockStressConfig())
        ),
        NamedStressScenarioConfig(
            name="Multiple",
            shocks=StressShockConfig(multiple_compression=MultipleCompressionStressConfig()),
        ),
    ]
    combined = NamedStressScenarioConfig(
        name="Combined",
        shocks=StressShockConfig(
            recession=RecessionStressConfig(start_year_index=2),
            rate_shock=RateShockStressConfig(),
            multiple_compression=MultipleCompressionStressConfig(),
        ),
    )

    individual_results = run_stress_scenarios(config, candidate, timeline, individual)
    combined_result = run_stress_scenario(config, candidate, timeline, combined)

    assert combined_result.irr < min(r.irr for r in individual_results)


def test_run_stress_scenarios_returns_one_result_per_config_in_order():
    config = _config(n_periods=5)
    timeline = Timeline.annual(n_periods=5)
    candidate = build_structure(config, {"TLB": 2.0})
    configs = [
        _baseline_scenario(),
        NamedStressScenarioConfig(
            name="Rate", shocks=StressShockConfig(rate_shock=RateShockStressConfig())
        ),
    ]
    results = run_stress_scenarios(config, candidate, timeline, configs)
    assert [r.name for r in results] == ["Base", "Rate"]


def test_stress_scenario_balance_sheet_and_debt_integrity():
    config = _config(n_periods=6)
    timeline = Timeline.annual(n_periods=6)
    candidate = build_structure(config, {"TLB": 2.0})
    shock = StressShockConfig(
        recession=RecessionStressConfig(start_year_index=2),
        rate_shock=RateShockStressConfig(),
        multiple_compression=MultipleCompressionStressConfig(),
    )
    shocked = apply_stress_shock(deterministic_drivers(config, timeline), timeline, shock)
    result = run_corporate_model_with_debt(
        config.company,
        candidate.opening_balance_sheet,
        candidate.tranches,
        config.waterfall,
        timeline,
        shocked,
    )
    run_checks(
        [
            check_balance_sheet_balances(result.balance_sheet),
            check_cash_ties_to_cfs(
                result.balance_sheet,
                result.cash_flow_statement,
                candidate.opening_balance_sheet.cash_mm,
            ),
            check_debt_rollforward(result.debt_schedule),
        ]
    )
