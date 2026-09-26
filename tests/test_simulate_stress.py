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
from corefin.simulate.liquidity import compute_liquidity
from corefin.simulate.stress import (
    apply_stress_shock,
    run_stress_scenario,
    run_stress_scenarios,
    stress_shock_start_year_label,
)
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


def test_liquidity_falls_when_revolver_is_drawn_in_a_stress_scenario():
    data = minimal_config_dict(n_periods=6)
    data["optimizer"] = {
        "decision_variables": [{"tranche_name": "TLB", "min_multiple": 1.0, "max_multiple": 5.0}]
    }
    config = RootConfig.model_validate(data)
    timeline = Timeline.annual(n_periods=6)
    candidate = build_structure(config, {"TLB": 4.5})  # highly levered -- thin cash cushion
    revolver = next(t for t in candidate.tranches if t.is_revolver)

    severe = StressShockConfig(
        recession=RecessionStressConfig(
            start_year_index=0, revenue_growth_hit=-0.6, ebitda_margin_hit=-0.18, recovery_years=0
        )
    )
    shocked = apply_stress_shock(deterministic_drivers(config, timeline), timeline, severe)
    result = run_corporate_model_with_debt(
        config.company,
        candidate.opening_balance_sheet,
        candidate.tranches,
        config.waterfall,
        timeline,
        shocked,
    )

    drawn = result.debt_schedule.ending_balance[revolver.name][0]
    assert np.any(drawn > 0)  # the shock must actually force a draw, or this test proves nothing

    liquidity = compute_liquidity(
        result.balance_sheet.cash, result.debt_schedule, candidate.tranches
    )[0]
    cash = result.balance_sheet.cash[0]
    np.testing.assert_allclose(liquidity, cash + (revolver.size_mm - drawn))
    drawn_periods = drawn > 0
    assert np.all(liquidity[drawn_periods] < cash[drawn_periods] + revolver.size_mm)


def test_stress_scenario_result_reports_liquidity_and_cash_separately():
    config = _config(n_periods=6)
    timeline = Timeline.annual(n_periods=6)
    candidate = build_structure(config, {"TLB": 2.0})
    revolver = next(t for t in candidate.tranches if t.is_revolver)

    baseline = run_stress_scenario(config, candidate, timeline, _baseline_scenario())
    assert baseline.max_revolver_draw_pct == pytest.approx(0.0, abs=1e-9)
    # Undrawn revolver -- liquidity is exactly cash plus the full commitment.
    assert baseline.min_liquidity_mm == pytest.approx(baseline.min_cash_mm + revolver.size_mm)
    # General invariant regardless of draw: liquidity[t] >= cash[t] for every t
    # (undrawn capacity is never negative), so the minimum over time can only rise.
    assert baseline.min_liquidity_mm >= baseline.min_cash_mm


def test_rate_shock_applies_exit_multiple_link_rate_sensitivity_when_configured():
    data = minimal_config_dict(n_periods=5)
    data["scenario"]["advanced"] = {
        "exit_multiple_link": {"beta_rate": -3.0, "reference_rate": 0.03}
    }
    config = RootConfig.model_validate(data)
    timeline = Timeline.annual(n_periods=5)
    base = deterministic_drivers(config, timeline)
    link = config.scenario.advanced.exit_multiple_link

    shock = StressShockConfig(rate_shock=RateShockStressConfig(bps=0.03))
    shocked = apply_stress_shock(base, timeline, shock, exit_multiple_link=link)
    expected = base.exit_multiple[0] + link.beta_rate * 0.03
    assert shocked.exit_multiple[0] == pytest.approx(expected)


def test_rate_shock_without_link_configured_leaves_exit_multiple_unaffected():
    config = _config(n_periods=5)
    timeline = Timeline.annual(n_periods=5)
    base = deterministic_drivers(config, timeline)
    shock = StressShockConfig(rate_shock=RateShockStressConfig(bps=0.03))
    shocked = apply_stress_shock(base, timeline, shock)  # exit_multiple_link defaults to None
    assert shocked.exit_multiple[0] == pytest.approx(base.exit_multiple[0])


def test_run_stress_scenario_rate_shock_exit_multiple_link_lowers_irr_further():
    def _make(beta_rate: float) -> RootConfig:
        data = minimal_config_dict(n_periods=5)
        data["optimizer"] = {
            "decision_variables": [
                {"tranche_name": "TLB", "min_multiple": 1.0, "max_multiple": 3.0}
            ]
        }
        if beta_rate != 0.0:
            data["scenario"]["advanced"] = {"exit_multiple_link": {"beta_rate": beta_rate}}
        return RootConfig.model_validate(data)

    timeline = Timeline.annual(n_periods=5)
    rate_shock_scenario = NamedStressScenarioConfig(
        name="Rate", shocks=StressShockConfig(rate_shock=RateShockStressConfig(bps=0.03))
    )

    config_no_link = _make(0.0)
    config_with_link = _make(-5.0)
    result_no_link = run_stress_scenario(
        config_no_link, build_structure(config_no_link, {"TLB": 2.0}), timeline, rate_shock_scenario
    )
    result_with_link = run_stress_scenario(
        config_with_link,
        build_structure(config_with_link, {"TLB": 2.0}),
        timeline,
        rate_shock_scenario,
    )

    assert result_with_link.irr < result_no_link.irr


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


def test_start_year_label_uses_calendar_year_when_configured():
    timeline = Timeline.annual(n_periods=5, start_year=2025)
    shock = StressShockConfig(recession=RecessionStressConfig(start_year_index=1))
    assert stress_shock_start_year_label(shock, timeline) == "2026"


def test_start_year_label_falls_back_to_period_label_without_start_year():
    timeline = Timeline.annual(n_periods=5)  # no start_year
    shock = StressShockConfig(recession=RecessionStressConfig(start_year_index=1))
    assert stress_shock_start_year_label(shock, timeline) == "Period 2"


def test_start_year_label_is_none_for_multiple_compression_alone():
    timeline = Timeline.annual(n_periods=5, start_year=2025)
    shock = StressShockConfig(multiple_compression=MultipleCompressionStressConfig(delta=-2.0))
    assert stress_shock_start_year_label(shock, timeline) is None


def test_start_year_label_uses_earliest_shock_when_combined():
    timeline = Timeline.annual(n_periods=5, start_year=2025)
    shock = StressShockConfig(
        recession=RecessionStressConfig(start_year_index=2),
        rate_shock=RateShockStressConfig(start_year_index=0),
    )
    assert stress_shock_start_year_label(shock, timeline) == "2025"
