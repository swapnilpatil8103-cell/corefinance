"""Driver importance: which assumptions explain IRR variance, kept simple
and explainable -- no black-box ML.

Two views from the Monte Carlo sample, both computed on each driver's
*path-averaged* value per scenario (a single number per scenario per
driver, since a driver is a whole (n_periods,) path):
- Standardized regression coefficients: OLS of IRR on the z-scored driver
  values (`numpy.linalg.lstsq`, no new dependency), so coefficients are
  directly comparable in magnitude -- "how much does IRR move for a
  1-standard-deviation move in this driver, holding the linear fit's
  other terms fixed."
- Spearman rank correlation of each driver against IRR -- robust to
  outliers and nonlinearity, no distributional assumption.

Plus a tornado chart built from actual deterministic re-runs (not a
linear extrapolation): each driver alone is shifted to its Monte Carlo
sample's p10/p90 path-average, every other driver held at the
deterministic base case, and the corporate model is re-run to get the
resulting IRR -- an honest "what if this one assumption moved" answer.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from corefin.assumptions.schema import RootConfig
from corefin.optimize.structure import CandidateStructure
from corefin.scenarios.drivers import DriverSet
from corefin.scenarios.generator import deterministic_drivers
from corefin.simulate.engine import SimulationResult
from corefin.statements.corporate_model import run_corporate_model_with_debt
from corefin.timeline import Timeline
from corefin.transaction.returns import compute_exit_and_returns

DRIVER_NAMES = (
    "revenue_growth",
    "ebitda_margin",
    "capex_pct_revenue",
    "nwc_pct_revenue",
    "base_rate",
    "exit_multiple",
)


def _path_averaged_driver_values(drivers: DriverSet) -> dict[str, np.ndarray]:
    """One scalar per scenario per driver: the mean over periods for the
    five path drivers, and exit_multiple as-is (already one value per
    scenario, since exit happens once)."""
    return {
        "revenue_growth": np.mean(drivers.revenue_growth, axis=1),
        "ebitda_margin": np.mean(drivers.ebitda_margin, axis=1),
        "capex_pct_revenue": np.mean(drivers.capex_pct_revenue, axis=1),
        "nwc_pct_revenue": np.mean(drivers.nwc_pct_revenue, axis=1),
        "base_rate": np.mean(drivers.base_rate, axis=1),
        "exit_multiple": drivers.exit_multiple,
    }


_ZERO_VARIANCE_STD_THRESHOLD = 1e-10
"""A driver with no configured vol still shows ~1e-17-scale float noise in
its cross-scenario std (from repeat/broadcast arithmetic), not exactly
0.0 -- comparing to a small threshold rather than exact equality avoids
dividing by that noise and amplifying it into a spurious huge z-score."""


def _zscore(x: np.ndarray) -> np.ndarray:
    std = np.std(x)
    if std < _ZERO_VARIANCE_STD_THRESHOLD:
        return np.zeros_like(x)
    return (x - np.mean(x)) / std


def _rank(x: np.ndarray) -> np.ndarray:
    return np.argsort(np.argsort(x)).astype(float)


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    # A constant (or effectively constant) array's double-argsort still
    # produces distinct-looking ranks (stable sort breaks ties by original
    # position), which would misreport a spurious nonzero correlation for
    # a zero-variance driver -- guarded against directly on the raw values
    # before ranking.
    if np.std(x) < _ZERO_VARIANCE_STD_THRESHOLD or np.std(y) < _ZERO_VARIANCE_STD_THRESHOLD:
        return 0.0
    rx, ry = _rank(x), _rank(y)
    return float(np.corrcoef(rx, ry)[0, 1])


@dataclass(frozen=True)
class DriverImportance:
    name: str
    standardized_coefficient: float
    rank_correlation: float


def compute_driver_importance(simulation_result: SimulationResult) -> list[DriverImportance]:
    irr = simulation_result.exit_result.irr
    values = _path_averaged_driver_values(simulation_result.drivers)

    z_columns = [_zscore(values[name]) for name in DRIVER_NAMES]
    design = np.column_stack([np.ones(irr.shape[0]), *z_columns])
    coefficients, *_ = np.linalg.lstsq(design, irr, rcond=None)
    standardized_coefficients = coefficients[1:]

    return [
        DriverImportance(
            name=name,
            standardized_coefficient=float(standardized_coefficients[i]),
            rank_correlation=_spearman(values[name], irr),
        )
        for i, name in enumerate(DRIVER_NAMES)
    ]


@dataclass(frozen=True)
class TornadoBar:
    name: str
    base_irr: float
    irr_at_p10: float
    irr_at_p90: float

    @property
    def irr_range(self) -> float:
        return abs(self.irr_at_p90 - self.irr_at_p10)


def _shift_driver(base: DriverSet, name: str, target: float) -> DriverSet:
    """A new DriverSet identical to `base` except driver `name`, whose path
    is shifted by a constant so its mean equals `target` (or, for
    exit_multiple, which has no path, set directly to `target`)."""
    fields = {
        "revenue_growth": base.revenue_growth,
        "ebitda_margin": base.ebitda_margin,
        "capex_pct_revenue": base.capex_pct_revenue,
        "nwc_pct_revenue": base.nwc_pct_revenue,
        "base_rate": base.base_rate,
        "exit_multiple": base.exit_multiple,
    }
    if name == "exit_multiple":
        fields["exit_multiple"] = np.array([target])
    else:
        path = fields[name]
        fields[name] = path + (target - float(np.mean(path)))
    return DriverSet(**fields)


def _deterministic_irr(
    root_config: RootConfig, candidate: CandidateStructure, timeline: Timeline, drivers: DriverSet
) -> float:
    result = run_corporate_model_with_debt(
        root_config.company,
        candidate.opening_balance_sheet,
        candidate.tranches,
        root_config.waterfall,
        timeline,
        drivers,
    )
    exit_result = compute_exit_and_returns(
        candidate.sources_and_uses.sponsor_equity_mm,
        root_config.transaction.exit_year_index,
        drivers.exit_multiple,
        result.income_statement.ebitda,
        result.balance_sheet.total_debt,
        result.balance_sheet.cash,
        result.cash_flow_statement.dividends,
    )
    return float(exit_result.irr[0])


def compute_tornado_chart(
    root_config: RootConfig,
    candidate: CandidateStructure,
    timeline: Timeline,
    simulation_result: SimulationResult,
) -> list[TornadoBar]:
    base_drivers = deterministic_drivers(root_config, timeline)
    base_irr = _deterministic_irr(root_config, candidate, timeline, base_drivers)
    values = _path_averaged_driver_values(simulation_result.drivers)

    bars = []
    for name in DRIVER_NAMES:
        p10 = float(np.percentile(values[name], 10))
        p90 = float(np.percentile(values[name], 90))
        irr_at_p10 = _deterministic_irr(
            root_config, candidate, timeline, _shift_driver(base_drivers, name, p10)
        )
        irr_at_p90 = _deterministic_irr(
            root_config, candidate, timeline, _shift_driver(base_drivers, name, p90)
        )
        bars.append(
            TornadoBar(name=name, base_irr=base_irr, irr_at_p10=irr_at_p10, irr_at_p90=irr_at_p90)
        )
    return bars
