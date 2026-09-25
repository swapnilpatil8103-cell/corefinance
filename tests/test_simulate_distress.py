import numpy as np

from corefin.debt.circularity import DebtScheduleResult
from corefin.metrics.credit_metrics import CreditMetrics
from corefin.simulate.distress import compute_distress_flags


def _debt_schedule(shortfall_flag: np.ndarray) -> DebtScheduleResult:
    shape = shortfall_flag.shape
    zeros = np.zeros(shape)
    return DebtScheduleResult(
        beginning_balance={},
        ending_balance={},
        cash_interest={},
        pik_interest={},
        fee_amortization={},
        mandatory_amort={},
        sweep_amort={},
        revolver_draw=zeros,
        revolver_paydown=zeros,
        ending_cash=zeros,
        shortfall_flag=shortfall_flag,
        iterations_used=np.zeros(shape[1], dtype=int),
    )


def _credit_metrics(interest_coverage: np.ndarray) -> CreditMetrics:
    filled = np.full_like(interest_coverage, 5.0)
    return CreditMetrics(
        total_net_leverage=filled,
        secured_net_leverage=filled,
        interest_coverage=interest_coverage,
        fccr=filled,
        cumulative_debt_paydown=np.zeros_like(interest_coverage),
    )


def test_healthy_scenario_is_never_in_distress():
    shape = (3, 4)
    schedule = _debt_schedule(np.zeros(shape, dtype=bool))
    metrics = _credit_metrics(np.full(shape, 3.0))
    assert not np.any(compute_distress_flags(schedule, metrics))


def test_revolver_shortfall_alone_triggers_distress():
    shape = (2, 3)
    shortfall = np.array([[False, True, False], [False, False, False]])
    schedule = _debt_schedule(shortfall)
    metrics = _credit_metrics(np.full(shape, 3.0))  # healthy coverage
    distress = compute_distress_flags(schedule, metrics)
    np.testing.assert_array_equal(distress, shortfall)


def test_thin_interest_coverage_alone_triggers_distress():
    shape = (2, 3)
    schedule = _debt_schedule(np.zeros(shape, dtype=bool))
    coverage = np.array([[3.0, 0.9, 1.0], [1.5, 0.5, 3.0]])
    distress = compute_distress_flags(schedule, _credit_metrics(coverage))
    # coverage < 1.0 is distress; exactly 1.0 is not (it's a floor, not a breach).
    expected = np.array([[False, True, False], [False, True, False]])
    np.testing.assert_array_equal(distress, expected)


def test_either_condition_triggers_distress_via_or():
    shortfall = np.array([[True, False]])
    coverage = np.array([[3.0, 0.5]])
    schedule = _debt_schedule(shortfall)
    metrics = _credit_metrics(coverage)
    distress = compute_distress_flags(schedule, metrics)
    np.testing.assert_array_equal(distress, np.array([[True, True]]))
