"""Period grid shared by every statement/scenario/check module.

Units: `period_length_years` is in years (1.0 for an annual model, 0.25 for a
quarterly model, 0.5 for a stub half-year). All arrays elsewhere in the
library are indexed along axis 1 by the periods defined here (axis 0 is
scenarios).
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
    start_quarter: int | None = None  # 1-4; only meaningful alongside start_year
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
        if self.start_quarter is not None:
            if self.start_year is None:
                raise ValueError("start_quarter requires start_year to also be set")
            if not 1 <= self.start_quarter <= 4:
                raise ValueError("start_quarter must be between 1 and 4")
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

    @classmethod
    def quarterly(
        cls,
        n_periods: int,
        n_historical: int = 0,
        start_year: int | None = None,
        start_quarter: int = 1,
    ) -> Timeline:
        """For the credit-loss forecasting engine's bank-quarter panel and
        9-quarter stress projections. `start_quarter` (1-4) is only stored
        (and only meaningful) when `start_year` is also given."""
        if not 0 <= n_historical <= n_periods:
            raise ValueError("n_historical must be between 0 and n_periods")
        if start_year is not None and not 1 <= start_quarter <= 4:
            raise ValueError("start_quarter must be between 1 and 4")
        is_projection = np.arange(n_periods) >= n_historical
        return cls(
            n_periods=n_periods,
            period_length_years=np.full(n_periods, 0.25),
            is_projection=is_projection,
            start_year=start_year,
            start_quarter=start_quarter if start_year is not None else None,
        )

    @property
    def n_projection_periods(self) -> int:
        return int(np.sum(self.is_projection))

    @property
    def year_labels(self) -> list[str]:
        if self.start_year is None:
            return [f"Period {i + 1}" for i in self.period_index]
        if self.start_quarter is not None:
            labels = []
            year, quarter = self.start_year, self.start_quarter
            for _ in self.period_index:
                labels.append(f"{year}Q{quarter}")
                quarter += 1
                if quarter > 4:
                    quarter = 1
                    year += 1
            return labels
        return [str(self.start_year + i) for i in self.period_index]
