"""Reusable integrity-check framework.

A check is a function `(scenario_index, period_index) -> CheckResult` operating
on full (n_scenarios, n_periods) arrays; it reports *which* scenario/period
pairs failed rather than a single pass/fail bool, since a bug in vectorized
code often only shows up for a handful of scenarios or periods.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

CheckFn = Callable[..., "CheckResult"]


@dataclass(frozen=True)
class CheckResult:
    name: str
    passed: bool
    failing_scenarios: np.ndarray
    failing_periods: np.ndarray
    max_abs_error: float
    message: str = ""

    def __bool__(self) -> bool:
        return self.passed

    def describe(self) -> str:
        if self.passed:
            return f"{self.name}: PASS (max_abs_error={self.max_abs_error:.3g})"
        n_failures = len(self.failing_scenarios)
        first = list(
            zip(
                self.failing_scenarios[:5].tolist(),
                self.failing_periods[:5].tolist(),
                strict=True,
            )
        )
        return (
            f"{self.name}: FAIL ({n_failures} scenario/period failures, "
            f"max_abs_error={self.max_abs_error:.3g}); first offenders "
            f"(scenario, period)={first}{' ...' if n_failures > 5 else ''}. "
            f"{self.message}"
        )


def check_close_to_zero(name: str, residual: np.ndarray, tolerance: float) -> CheckResult:
    """residual: shape (n_scenarios, n_periods); passes where |residual| <= tolerance."""
    abs_residual = np.abs(residual)
    failing = abs_residual > tolerance
    scenario_idx, period_idx = np.nonzero(failing)
    return CheckResult(
        name=name,
        passed=not np.any(failing),
        failing_scenarios=scenario_idx,
        failing_periods=period_idx,
        max_abs_error=float(np.max(abs_residual)) if abs_residual.size else 0.0,
    )


def check_bounds(
    name: str,
    values: np.ndarray,
    lower: np.ndarray | float | None = None,
    upper: np.ndarray | float | None = None,
    tolerance: float = 1e-9,
) -> CheckResult:
    """Passes where lower - tol <= values <= upper + tol (either bound optional)."""
    violation = np.zeros_like(values, dtype=bool)
    excess = np.zeros_like(values, dtype=float)
    if lower is not None:
        shortfall = np.maximum(lower - tolerance - values, 0.0)
        violation |= shortfall > 0
        excess = np.maximum(excess, shortfall)
    if upper is not None:
        overage = np.maximum(values - upper - tolerance, 0.0)
        violation |= overage > 0
        excess = np.maximum(excess, overage)
    scenario_idx, period_idx = np.nonzero(violation)
    return CheckResult(
        name=name,
        passed=not np.any(violation),
        failing_scenarios=scenario_idx,
        failing_periods=period_idx,
        max_abs_error=float(np.max(excess)) if excess.size else 0.0,
    )


def run_checks(results: list[CheckResult]) -> None:
    """Raise with a combined message if any check in the list failed."""
    failures = [r for r in results if not r.passed]
    if failures:
        detail = "\n".join(r.describe() for r in failures)
        raise AssertionError(f"{len(failures)} integrity check(s) failed:\n{detail}")
