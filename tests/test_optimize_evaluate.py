import numpy as np
import pytest

from corefin.assumptions.schema import ObjectiveConfig, RootConfig
from corefin.optimize.evaluate import compute_objective, evaluate_candidate
from corefin.optimize.pricing import build_priced_structure
from corefin.optimize.structure import compute_entry_ebitda_mm
from corefin.scenarios.generator import deterministic_drivers, generate_stochastic_drivers
from corefin.timeline import Timeline
from tests.test_debt_integration import debt_config_dict


def _optimizer_data(n_periods: int = 6, **optimizer_overrides) -> dict:
    data = debt_config_dict(n_periods=n_periods)
    data["optimizer"] = {
        "decision_variables": [
            {"tranche_name": "TLB", "min_multiple": 0.5, "max_multiple": 6.0},
            {"tranche_name": "Notes", "min_multiple": 0.0, "max_multiple": 3.0},
        ],
        **optimizer_overrides,
    }
    data["scenario"]["n_scenarios"] = 200
    data["scenario"]["driver_vol"] = {"revenue_growth_std": 0.03, "ebitda_margin_std": 0.02}
    return data


def _setup(config: RootConfig, n_periods: int):
    timeline = Timeline.annual(n_periods=n_periods)
    stochastic = generate_stochastic_drivers(config, timeline, seed=1)
    deterministic = deterministic_drivers(config, timeline)
    return timeline, stochastic, deterministic


def test_market_capacity_infeasible_skips_model_run():
    data = _optimizer_data(
        pricing={
            "tranches": [
                {"tranche_name": "TLB", "leverage_threshold": 0.0, "market_capacity_mm": 100.0}
            ]
        }
    )
    config = RootConfig.model_validate(data)
    timeline, stochastic, det = _setup(config, 6)
    candidate = build_priced_structure(config, {"TLB": 4.0, "Notes": 0.5})  # way over 100mm cap
    evaluation = evaluate_candidate(config, candidate, timeline, stochastic, det)
    assert not evaluation.feasible
    assert evaluation.result is None
    assert np.isnan(evaluation.objective_value)
    assert any(v.name == "market_capacity" for v in evaluation.violations)


def test_negative_equity_skips_model_run():
    config = RootConfig.model_validate(_optimizer_data())
    timeline, stochastic, det = _setup(config, 6)
    # Entry EBITDA is 100mm (500mm revenue * 20% margin) at an 8x entry
    # multiple = 800mm purchase price; 6x TLB (600mm) + 3x Notes (300mm) =
    # 900mm of debt sources raises more than the deal needs.
    candidate = build_priced_structure(config, {"TLB": 6.0, "Notes": 3.0})
    assert candidate.sources_and_uses.sponsor_equity_mm < 0
    evaluation = evaluate_candidate(config, candidate, timeline, stochastic, det)
    assert not evaluation.feasible
    assert evaluation.result is None
    assert any(v.name == "sponsor_equity_non_negative" for v in evaluation.violations)


def test_market_capacity_exactly_at_limit_is_feasible():
    data = _optimizer_data(
        pricing={
            "tranches": [
                {"tranche_name": "TLB", "leverage_threshold": 0.0, "market_capacity_mm": 300.0}
            ]
        }
    )
    config = RootConfig.model_validate(data)
    timeline, stochastic, det = _setup(config, 6)
    entry_ebitda_mm = compute_entry_ebitda_mm(config)
    candidate = build_priced_structure(config, {"TLB": 300.0 / entry_ebitda_mm, "Notes": 0.5})
    evaluation = evaluate_candidate(config, candidate, timeline, stochastic, det)
    assert evaluation.result is not None  # ran the model -- not hard-infeasible


def test_max_total_leverage_violation_still_runs_model_and_reports_value():
    data = _optimizer_data()
    data["optimizer"]["deterministic_constraints"] = {"max_total_leverage": 2.0}
    config = RootConfig.model_validate(data)
    timeline, stochastic, det = _setup(config, 6)
    candidate = build_priced_structure(config, {"TLB": 3.0, "Notes": 0.5})  # 3.5x > 2.0x cap
    evaluation = evaluate_candidate(config, candidate, timeline, stochastic, det)
    assert evaluation.result is not None
    assert not evaluation.feasible
    assert not np.isnan(evaluation.objective_value)
    violation = next(v for v in evaluation.violations if v.name == "max_total_leverage")
    assert violation.limit == 2.0
    assert violation.actual == pytest.approx(3.5)


def test_min_equity_pct_of_sources_constraint():
    data = _optimizer_data()
    data["optimizer"]["deterministic_constraints"] = {"min_equity_pct_of_sources": 0.60}
    config = RootConfig.model_validate(data)
    timeline, stochastic, det = _setup(config, 6)
    low_equity = evaluate_candidate(
        config,
        build_priced_structure(config, {"TLB": 5.0, "Notes": 2.0}),
        timeline,
        stochastic,
        det,
    )
    high_equity = evaluate_candidate(
        config,
        build_priced_structure(config, {"TLB": 0.5, "Notes": 0.0}),
        timeline,
        stochastic,
        det,
    )
    assert any(v.name == "min_equity_pct_of_sources" for v in low_equity.violations)
    assert not any(v.name == "min_equity_pct_of_sources" for v in high_equity.violations)


def test_min_interest_coverage_at_close_uses_deterministic_run():
    data = _optimizer_data()
    data["optimizer"]["deterministic_constraints"] = {"min_interest_coverage_at_close": 1000.0}
    config = RootConfig.model_validate(data)
    timeline, stochastic, det = _setup(config, 6)
    candidate = build_priced_structure(config, {"TLB": 2.0, "Notes": 0.5})
    evaluation = evaluate_candidate(config, candidate, timeline, stochastic, det)
    assert any(v.name == "min_interest_coverage_at_close" for v in evaluation.violations)
    assert "interest_coverage_at_close" in evaluation.constraint_values


def test_stochastic_constraints_recorded_and_enforced():
    data = _optimizer_data()
    data["optimizer"]["stochastic_constraints"] = {"max_loss_of_capital_probability": 0.0}
    # Heavy volatility on an already highly levered candidate -- enough for
    # some scenarios to lose money -- without degrading the deterministic
    # base case itself (which would break the structure builder's opening
    # balance sheet derivation; see the negative-goodwill guard elsewhere).
    data["scenario"]["driver_vol"] = {"revenue_growth_std": 0.10, "ebitda_margin_std": 0.06}
    config = RootConfig.model_validate(data)
    timeline, stochastic, det = _setup(config, 6)
    candidate = build_priced_structure(config, {"TLB": 5.0, "Notes": 2.0})
    evaluation = evaluate_candidate(config, candidate, timeline, stochastic, det)
    assert evaluation.constraint_values["loss_of_capital_probability"] > 0.0
    assert any(v.name == "max_loss_of_capital_probability" for v in evaluation.violations)


def test_no_constraints_configured_means_always_feasible_if_model_runs():
    config = RootConfig.model_validate(_optimizer_data())
    timeline, stochastic, det = _setup(config, 6)
    candidate = build_priced_structure(config, {"TLB": 3.0, "Notes": 1.0})
    evaluation = evaluate_candidate(config, candidate, timeline, stochastic, det)
    assert evaluation.feasible
    assert evaluation.violations == []


def test_common_random_numbers_operating_performance_is_structure_independent():
    """Revenue/EBITDA don't depend on financing structure, so reusing the same
    DriverSet across different candidates must give identical operating
    results -- the core of common random numbers working correctly."""
    config = RootConfig.model_validate(_optimizer_data())
    timeline, stochastic, det = _setup(config, 6)
    low = evaluate_candidate(
        config,
        build_priced_structure(config, {"TLB": 1.0, "Notes": 0.5}),
        timeline,
        stochastic,
        det,
    )
    high = evaluate_candidate(
        config,
        build_priced_structure(config, {"TLB": 4.0, "Notes": 1.5}),
        timeline,
        stochastic,
        det,
    )
    np.testing.assert_array_equal(
        low.result.income_statement.revenue, high.result.income_statement.revenue
    )
    np.testing.assert_array_equal(
        low.result.income_statement.ebitda, high.result.income_statement.ebitda
    )


def test_compute_objective_mean_irr():
    irr = np.array([0.1, 0.2, 0.3, -0.1])
    objective = ObjectiveConfig(kind="mean_irr")
    assert compute_objective(objective, irr) == pytest.approx(np.mean(irr))


def test_compute_objective_median_irr():
    irr = np.array([0.1, 0.2, 0.3, -0.1])
    objective = ObjectiveConfig(kind="median_irr")
    assert compute_objective(objective, irr) == pytest.approx(np.median(irr))


def test_compute_objective_percentile_irr():
    irr = np.linspace(-0.2, 0.5, 101)
    objective = ObjectiveConfig(kind="percentile_irr", percentile=10.0)
    assert compute_objective(objective, irr) == pytest.approx(np.percentile(irr, 10.0))


def test_compute_objective_mean_downside_blend_lambda_zero_equals_mean():
    irr = np.array([0.1, 0.2, 0.3, -0.1])
    objective = ObjectiveConfig(kind="mean_downside_blend", percentile=10.0, downside_lambda=0.0)
    assert compute_objective(objective, irr) == pytest.approx(np.mean(irr))


def test_compute_objective_mean_downside_blend_lambda_one_equals_percentile():
    irr = np.linspace(-0.2, 0.5, 101)
    objective = ObjectiveConfig(kind="mean_downside_blend", percentile=10.0, downside_lambda=1.0)
    assert compute_objective(objective, irr) == pytest.approx(np.percentile(irr, 10.0), abs=1e-9)


def test_compute_objective_mean_downside_blend_halfway():
    irr = np.linspace(-0.2, 0.5, 101)
    objective = ObjectiveConfig(kind="mean_downside_blend", percentile=10.0, downside_lambda=0.5)
    mean_irr = np.mean(irr)
    p10 = np.percentile(irr, 10.0)
    expected = mean_irr - 0.5 * (mean_irr - p10)
    assert compute_objective(objective, irr) == pytest.approx(expected)


# Binding-constraint reporting moved to optimize/diagnostics.py (needs
# tolerance config, decision-variable bounds and grid step info a single
# CandidateEvaluation + RootConfig can't provide) -- see
# tests/test_optimize_diagnostics.py.
