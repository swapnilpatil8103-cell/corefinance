import numpy as np
import pytest

from corefin.checks.framework import check_bounds, check_close_to_zero, run_checks


def test_check_close_to_zero_passes_within_tolerance():
    residual = np.array([[1e-9, -1e-9], [0.0, 5e-10]])
    result = check_close_to_zero("balance", residual, tolerance=1e-6)
    assert result.passed
    assert bool(result)


def test_check_close_to_zero_reports_failing_indices():
    residual = np.zeros((3, 2))
    residual[1, 0] = 0.01
    residual[2, 1] = 0.02
    result = check_close_to_zero("balance", residual, tolerance=1e-6)
    assert not result.passed
    pairs = zip(result.failing_scenarios.tolist(), result.failing_periods.tolist(), strict=True)
    assert set(pairs) == {(1, 0), (2, 1)}
    assert result.max_abs_error == pytest.approx(0.02)


def test_check_bounds_lower_and_upper():
    values = np.array([[-1.0, 0.5, 2.0]])
    result = check_bounds("revolver_balance", values, lower=0.0, upper=1.0)
    assert not result.passed
    assert set(result.failing_periods.tolist()) == {0, 2}


def test_check_bounds_all_within():
    values = np.array([[0.1, 0.5, 0.9]])
    result = check_bounds("revolver_balance", values, lower=0.0, upper=1.0)
    assert result.passed


def test_run_checks_raises_on_failure():
    residual = np.array([[0.5]])
    failing = check_close_to_zero("bs_balance", residual, tolerance=1e-6)
    with pytest.raises(AssertionError, match="1 integrity check"):
        run_checks([failing])


def test_run_checks_passes_silently():
    residual = np.array([[0.0]])
    passing = check_close_to_zero("bs_balance", residual, tolerance=1e-6)
    run_checks([passing])
