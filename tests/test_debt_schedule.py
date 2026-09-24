import numpy as np

from corefin.assumptions.schema import RateType, TrancheConfig, TrancheType
from corefin.debt.schedule import (
    effective_rate,
    fee_amortization_schedule,
    scheduled_mandatory_amortization,
)


def _tranche(**overrides) -> TrancheConfig:
    defaults = dict(
        name="TLB",
        tranche_type=TrancheType.TERM_LOAN_B,
        size_mm=300.0,
        rate_type=RateType.FLOATING,
        spread=0.05,
    )
    defaults.update(overrides)
    return TrancheConfig(**defaults)


def test_effective_rate_fixed():
    tranche = _tranche(rate_type=RateType.FIXED, fixed_rate=0.08, spread=None)
    base_rate = np.array([[0.03, 0.04]])
    rate = effective_rate(tranche, base_rate)
    np.testing.assert_allclose(rate, [[0.08, 0.08]])


def test_effective_rate_floating_with_floor():
    tranche = _tranche(rate_type=RateType.FLOATING, spread=0.05, rate_floor=0.02)
    base_rate = np.array([[0.01, 0.03]])
    rate = effective_rate(tranche, base_rate)
    np.testing.assert_allclose(rate, [[0.07, 0.08]])


def test_effective_rate_floating_without_floor():
    tranche = _tranche(rate_type=RateType.FLOATING, spread=0.05)
    base_rate = np.array([[0.01, 0.03]])
    rate = effective_rate(tranche, base_rate)
    np.testing.assert_allclose(rate, [[0.06, 0.08]])


def test_scheduled_mandatory_amortization_scalar():
    tranche = _tranche(size_mm=300.0, mandatory_amort_pct_of_original=0.01)
    schedule = scheduled_mandatory_amortization(tranche, n_periods=4)
    np.testing.assert_allclose(schedule, [3.0, 3.0, 3.0, 3.0])


def test_scheduled_mandatory_amortization_series():
    tranche = _tranche(size_mm=100.0, mandatory_amort_pct_of_original=[0.0, 0.05, 0.05, 0.10])
    schedule = scheduled_mandatory_amortization(tranche, n_periods=4)
    np.testing.assert_allclose(schedule, [0.0, 5.0, 5.0, 10.0])


def test_fee_amortization_schedule_within_horizon():
    tranche = _tranche(size_mm=200.0, upfront_fee_pct=0.02, oid_pct=0.01, fee_amortization_years=3)
    schedule = fee_amortization_schedule(tranche, n_periods=5)
    total_fee = 0.03 * 200.0
    np.testing.assert_allclose(schedule, [total_fee / 3, total_fee / 3, total_fee / 3, 0.0, 0.0])


def test_fee_amortization_schedule_beyond_horizon():
    tranche = _tranche(size_mm=200.0, upfront_fee_pct=0.02, fee_amortization_years=10)
    schedule = fee_amortization_schedule(tranche, n_periods=4)
    total_fee = 0.02 * 200.0
    assert schedule.shape == (4,)
    np.testing.assert_allclose(schedule, np.full(4, total_fee / 10))


def test_fee_amortization_schedule_zero_when_no_fees():
    tranche = _tranche()
    schedule = fee_amortization_schedule(tranche, n_periods=4)
    np.testing.assert_allclose(schedule, np.zeros(4))
