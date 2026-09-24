"""Period grid shared by every statement/scenario/check module.

Units: `period_length_years` is in years (1.0 for an annual model, 0.5 for a
stub half-year). All arrays elsewhere in the library are indexed along axis 1
by the periods defined here (axis 0 is scenarios).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class Timeline:
    n_periods: int
    period_length_years: np.ndarray
    is_projection: np.ndarray
    start_year: int | None = None
    period_index: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        if self.n_periods <= 0:
            raise ValueError("n_periods must be positive")
        if self.period_length_years.shape != (self.n_periods,):
            raise ValueError("period_length_years must have shape (n_periods,)")
        if self.is_projection.shape != (self.n_periods,):
            raise ValueError("is_projection must have shape (n_periods,)")
        if np.any(self.period_length_years <= 0):
            raise ValueError("period_length_years must be strictly positive")
        object.__setattr__(self, "period_index", np.arange(self.n_periods))

    @classmethod
    def annual(
        cls,
        n_periods: int,
        n_historical: int = 0,
        start_year: int | None = None,
    ) -> Timeline:
        if not 0 <= n_historical <= n_periods:
            raise ValueError("n_historical must be between 0 and n_periods")
        is_projection = np.arange(n_periods) >= n_historical
        return cls(
            n_periods=n_periods,
            period_length_years=np.ones(n_periods),
            is_projection=is_projection,
            start_year=start_year,
        )

    @property
    def n_projection_periods(self) -> int:
        return int(np.sum(self.is_projection))

    @property
    def year_labels(self) -> list[str]:
        if self.start_year is None:
            return [f"Period {i + 1}" for i in self.period_index]
        return [str(self.start_year + i) for i in self.period_index]
