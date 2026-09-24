import numpy as np

from corefin.assumptions.schema import CovenantConfig, RootConfig
from corefin.metrics.covenants import evaluate_covenant
from corefin.metrics.credit_metrics import compute_credit_metrics, is_senior, senior_debt_balance
from corefin.scenarios.generator import deterministic_drivers
from corefin.statements.corporate_model import run_corporate_model_with_debt
from corefin.timeline import Timeline
from tests.test_debt_integration import debt_config_dict


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
