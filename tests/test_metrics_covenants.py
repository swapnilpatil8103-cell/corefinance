from pathlib import Path

import numpy as np
import pytest

from corefin.assumptions.loader import load_config
from corefin.assumptions.schema import (
    CovenantConfig,
    RateType,
    RootConfig,
    TrancheConfig,
    TrancheType,
)
from corefin.debt.circularity import DebtScheduleResult
from corefin.metrics.covenants import (
    compute_base_case_headroom,
    evaluate_covenant,
    evaluate_covenants,
    low_headroom_warnings,
)
from corefin.metrics.credit_metrics import (
    CreditMetrics,
    compute_credit_metrics,
    is_senior,
    senior_debt_balance,
)
from corefin.scenarios.generator import deterministic_drivers
from corefin.statements.corporate_model import run_corporate_model_with_debt
from corefin.timeline import Timeline
from tests.test_debt_integration import debt_config_dict

EXAMPLE_CONFIG_PATH = Path(__file__).resolve().parent.parent / "configs" / "example_midmarket.yaml"


def _run(n_periods=6):
    config = RootConfig.model_validate(debt_config_dict(n_periods=n_periods))
    timeline = Timeline.annual(n_periods=n_periods)
    drivers = deterministic_drivers(config, timeline)
    result = run_corporate_model_with_debt(
        config.company,
        config.opening_balance_sheet,
        config.tranches,
        config.waterfall,
        timeline,
        drivers,
    )
    return config, timeline, result


def test_is_senior_classification():
    config, *_ = _run()
    senior_names = {t.name for t in config.tranches if is_senior(t)}
    assert senior_names == {"Revolver", "TLB", "Notes"}


def test_senior_debt_excludes_subordinated_pik():
    data = debt_config_dict(n_periods=3)
    data["tranches"].append(
        {
            "name": "SubPIK",
            "tranche_type": "subordinated_pik",
            "size_mm": 50.0,
            "rate_type": "fixed",
            "fixed_rate": 0.12,
            "pik_fraction": 1.0,
        }
    )
    config = RootConfig.model_validate(data)
    timeline = Timeline.annual(n_periods=3)
    drivers = deterministic_drivers(config, timeline)
    result = run_corporate_model_with_debt(
        config.company,
        config.opening_balance_sheet,
        config.tranches,
        config.waterfall,
        timeline,
        drivers,
    )
    senior_debt = senior_debt_balance(result.debt_schedule, config.tranches)
    total_debt = result.balance_sheet.total_debt
    assert np.all(senior_debt < total_debt)


def test_credit_metrics_hand_check():
    config, timeline, result = _run(n_periods=3)
    metrics = compute_credit_metrics(
        result.debt_schedule,
        config.tranches,
        result.income_statement.ebitda,
        result.cash_flow_statement.capex,
        result.balance_sheet.cash,
        result.balance_sheet.total_debt,
    )
    expected_total_leverage = (result.balance_sheet.total_debt - result.balance_sheet.cash) / (
        result.income_statement.ebitda
    )
    np.testing.assert_allclose(metrics.total_net_leverage, expected_total_leverage)
    initial_total_debt = 250.0 + 100.0  # TLB + Notes at close (revolver starts undrawn)
    np.testing.assert_allclose(
        metrics.cumulative_debt_paydown, initial_total_debt - result.balance_sheet.total_debt
    )


def test_leverage_covenant_breach_is_a_maximum():
    config, timeline, result = _run(n_periods=4)
    metrics = compute_credit_metrics(
        result.debt_schedule,
        config.tranches,
        result.income_statement.ebitda,
        result.cash_flow_statement.capex,
        result.balance_sheet.cash,
        result.balance_sheet.total_debt,
    )
    tight_covenant = CovenantConfig(
        name="max_leverage", metric="total_net_leverage", threshold=0.0, test_from_period=0
    )
    loose_covenant = CovenantConfig(
        name="max_leverage", metric="total_net_leverage", threshold=100.0, test_from_period=0
    )
    tight_result = evaluate_covenant(tight_covenant, metrics, timeline)
    loose_result = evaluate_covenant(loose_covenant, metrics, timeline)
    assert np.all(tight_result.breach)
    assert not np.any(loose_result.breach)
    np.testing.assert_allclose(tight_result.headroom, -metrics.total_net_leverage)


def test_coverage_covenant_breach_is_a_minimum():
    config, timeline, result = _run(n_periods=4)
    metrics = compute_credit_metrics(
        result.debt_schedule,
        config.tranches,
        result.income_statement.ebitda,
        result.cash_flow_statement.capex,
        result.balance_sheet.cash,
        result.balance_sheet.total_debt,
    )
    tight_covenant = CovenantConfig(
        name="min_coverage", metric="interest_coverage", threshold=1000.0, test_from_period=0
    )
    loose_covenant = CovenantConfig(
        name="min_coverage", metric="interest_coverage", threshold=0.0, test_from_period=0
    )
    tight_result = evaluate_covenant(tight_covenant, metrics, timeline)
    loose_result = evaluate_covenant(loose_covenant, metrics, timeline)
    assert np.all(tight_result.breach)
    assert not np.any(loose_result.breach)


def test_covenant_test_from_period_gates_early_periods():
    config, timeline, result = _run(n_periods=4)
    metrics = compute_credit_metrics(
        result.debt_schedule,
        config.tranches,
        result.income_statement.ebitda,
        result.cash_flow_statement.capex,
        result.balance_sheet.cash,
        result.balance_sheet.total_debt,
    )
    covenant = CovenantConfig(
        name="max_leverage", metric="total_net_leverage", threshold=0.0, test_from_period=2
    )
    covenant_result = evaluate_covenant(covenant, metrics, timeline)
    assert not np.any(covenant_result.breach[:, :2])
    assert np.all(covenant_result.breach[:, 2:])


def _revolver_tranche(size_mm: float = 100.0) -> TrancheConfig:
    return TrancheConfig(
        name="Revolver",
        tranche_type=TrancheType.REVOLVER,
        size_mm=size_mm,
        rate_type=RateType.FLOATING,
        spread=0.04,
    )


def _synthetic_debt_schedule(revolver_balance: np.ndarray) -> DebtScheduleResult:
    shape = revolver_balance.shape
    zeros = np.zeros(shape)
    return DebtScheduleResult(
        beginning_balance={"Revolver": zeros},
        ending_balance={"Revolver": revolver_balance},
        cash_interest={"Revolver": zeros},
        pik_interest={"Revolver": zeros},
        fee_amortization={"Revolver": zeros},
        mandatory_amort={},
        sweep_amort={},
        revolver_draw=zeros,
        revolver_paydown=zeros,
        ending_cash=zeros,
        shortfall_flag=np.zeros(shape, dtype=bool),
        iterations_used=np.zeros(shape[1], dtype=int),
    )


def _synthetic_metrics(shape: tuple[int, int], value: float = 3.0) -> CreditMetrics:
    filled = np.full(shape, value)
    return CreditMetrics(
        total_net_leverage=filled,
        senior_net_leverage=filled,
        interest_coverage=filled,
        fccr=filled,
        cumulative_debt_paydown=np.zeros(shape),
    )


def test_springing_covenant_not_tested_at_or_below_trigger():
    shape = (3, 2)
    revolver = _revolver_tranche(size_mm=100.0)
    # 0%, 20%, 35% drawn -- all at or below the 35% trigger.
    balance = np.array([[0.0, 0.0], [20.0, 20.0], [35.0, 35.0]])
    debt_schedule = _synthetic_debt_schedule(balance)
    metrics = _synthetic_metrics(shape)
    timeline = Timeline.annual(n_periods=2)
    covenant = CovenantConfig(
        name="Springing Leverage",
        metric="total_net_leverage",
        threshold=100.0,  # never breaches on value; isolates the tested/not-tested behavior
        springing_revolver_draw_pct=0.35,
    )
    result = evaluate_covenant(covenant, metrics, timeline, debt_schedule, [revolver])
    assert not np.any(result.tested)
    assert not np.any(result.breach)


def test_springing_covenant_tested_above_trigger():
    shape = (2, 2)
    revolver = _revolver_tranche(size_mm=100.0)
    balance = np.array([[36.0, 50.0], [90.0, 36.1]])
    debt_schedule = _synthetic_debt_schedule(balance)
    metrics = _synthetic_metrics(shape)
    timeline = Timeline.annual(n_periods=2)
    covenant = CovenantConfig(
        name="Springing Leverage",
        metric="total_net_leverage",
        threshold=100.0,
        springing_revolver_draw_pct=0.35,
    )
    result = evaluate_covenant(covenant, metrics, timeline, debt_schedule, [revolver])
    assert np.all(result.tested)


def test_springing_covenant_never_triggered_zero_breach_and_zero_tested():
    shape = (50, 4)
    revolver = _revolver_tranche(size_mm=100.0)
    rng = np.random.default_rng(0)
    balance = rng.uniform(0.0, 34.0, size=shape)  # always below the 35% trigger
    debt_schedule = _synthetic_debt_schedule(balance)
    metrics = _synthetic_metrics(shape, value=1000.0)  # would breach any sane threshold if tested
    timeline = Timeline.annual(n_periods=4)
    covenant = CovenantConfig(
        name="Springing Leverage",
        metric="total_net_leverage",
        threshold=5.0,
        springing_revolver_draw_pct=0.35,
    )
    result = evaluate_covenant(covenant, metrics, timeline, debt_schedule, [revolver])
    assert np.mean(result.tested) == 0.0
    assert np.mean(np.any(result.breach, axis=1)) == 0.0


def test_springing_covenant_breaches_only_when_tested_and_over_threshold():
    shape = (1, 3)
    revolver = _revolver_tranche(size_mm=100.0)
    balance = np.array([[10.0, 40.0, 40.0]])  # not tested, tested, tested
    debt_schedule = _synthetic_debt_schedule(balance)
    leverage = np.array([[10.0, 10.0, 2.0]])  # would breach a 5x cap except in period 2
    metrics = CreditMetrics(
        total_net_leverage=leverage,
        senior_net_leverage=leverage,
        interest_coverage=np.full(shape, 3.0),
        fccr=np.full(shape, 3.0),
        cumulative_debt_paydown=np.zeros(shape),
    )
    timeline = Timeline.annual(n_periods=3)
    covenant = CovenantConfig(
        name="Springing Leverage",
        metric="total_net_leverage",
        threshold=5.0,
        springing_revolver_draw_pct=0.35,
    )
    result = evaluate_covenant(covenant, metrics, timeline, debt_schedule, [revolver])
    np.testing.assert_array_equal(result.tested, [[False, True, True]])
    np.testing.assert_array_equal(result.breach, [[False, True, False]])


def test_springing_without_debt_schedule_raises_clear_error():
    shape = (1, 2)
    metrics = _synthetic_metrics(shape)
    timeline = Timeline.annual(n_periods=2)
    covenant = CovenantConfig(
        name="Springing Leverage",
        metric="total_net_leverage",
        threshold=5.0,
        springing_revolver_draw_pct=0.35,
    )
    with pytest.raises(ValueError, match="springing_revolver_draw_pct"):
        evaluate_covenant(covenant, metrics, timeline)


def test_no_springing_field_matches_legacy_test_from_period_behavior():
    """Backward compatibility: a covenant without springing_revolver_draw_pct set
    must produce `tested` identical to the old test_from_period-only mask."""
    config, timeline, result = _run(n_periods=5)
    metrics = compute_credit_metrics(
        result.debt_schedule,
        config.tranches,
        result.income_statement.ebitda,
        result.cash_flow_statement.capex,
        result.balance_sheet.cash,
        result.balance_sheet.total_debt,
    )
    covenant = CovenantConfig(
        name="max_leverage", metric="total_net_leverage", threshold=0.0, test_from_period=2
    )
    assert covenant.springing_revolver_draw_pct is None
    result_cov = evaluate_covenant(covenant, metrics, timeline)
    expected_tested = np.broadcast_to(
        (timeline.period_index >= 2)[np.newaxis, :], result_cov.tested.shape
    )
    np.testing.assert_array_equal(result_cov.tested, expected_tested)
    # Same as the pre-springing behavior asserted in
    # test_covenant_test_from_period_gates_early_periods.
    np.testing.assert_array_equal(result_cov.breach, result_cov.tested & (result_cov.headroom < 0))


def test_headroom_warning_fires_on_tight_config():
    data = debt_config_dict(n_periods=4)
    config = RootConfig.model_validate(data)
    timeline = Timeline.annual(n_periods=4)
    drivers = deterministic_drivers(config, timeline)
    result = run_corporate_model_with_debt(
        config.company,
        config.opening_balance_sheet,
        config.tranches,
        config.waterfall,
        timeline,
        drivers,
    )
    metrics = compute_credit_metrics(
        result.debt_schedule,
        config.tranches,
        result.income_statement.ebitda,
        result.cash_flow_statement.capex,
        result.balance_sheet.cash,
        result.balance_sheet.total_debt,
    )
    # Threshold set just barely under the actual period-0 leverage value, so
    # headroom is small but still compliant (not a breach).
    actual_leverage_t0 = float(metrics.total_net_leverage[0, 0])
    tight_covenant = CovenantConfig(
        name="Tight Leverage",
        metric="total_net_leverage",
        threshold=actual_leverage_t0 * 1.02,
        test_from_period=0,
    )
    covenant_results = evaluate_covenants([tight_covenant], metrics, timeline)
    headrooms = compute_base_case_headroom(covenant_results, timeline)
    warnings = low_headroom_warnings(headrooms)
    assert len(warnings) == 1
    assert warnings[0].name == "Tight Leverage"
    assert warnings[0].cushion_pct < 0.15


def test_headroom_warning_silent_on_recalibrated_example():
    config = load_config(EXAMPLE_CONFIG_PATH)
    timeline = Timeline.annual(
        n_periods=config.timeline.n_periods,
        n_historical=config.timeline.n_historical,
        start_year=config.timeline.start_year,
    )
    drivers = deterministic_drivers(config, timeline)
    result = run_corporate_model_with_debt(
        config.company,
        config.opening_balance_sheet,
        config.tranches,
        config.waterfall,
        timeline,
        drivers,
    )
    metrics = compute_credit_metrics(
        result.debt_schedule,
        config.tranches,
        result.income_statement.ebitda,
        result.cash_flow_statement.capex,
        result.balance_sheet.cash,
        result.balance_sheet.total_debt,
    )
    covenant_results = evaluate_covenants(
        config.covenants, metrics, timeline, result.debt_schedule, config.tranches
    )
    headrooms = compute_base_case_headroom(covenant_results, timeline)
    assert low_headroom_warnings(headrooms) == []
    # The leverage covenant is springing and never triggers in the base case
    # (revolver stays undrawn); interest coverage is a regular covenant with
    # comfortable headroom.
    by_name = {h.name: h for h in headrooms}
    assert by_name["Max Total Net Leverage"].tested is False
    assert by_name["Min Interest Coverage"].tested is True
    assert by_name["Min Interest Coverage"].cushion_pct > 0.15
