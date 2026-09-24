"""DriverSet: the vectorized inputs consumed by the statement/debt layers.

Every time-series field has shape (n_scenarios, n_periods). `exit_multiple` is
a single per-scenario draw (shape (n_scenarios,)), since exit happens once per
scenario rather than every period. Units follow `assumptions.schema`: rates
and margins are decimal fractions, exit_multiple is a bare float.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class DriverSet:
    revenue_growth: np.ndarray
    ebitda_margin: np.ndarray
    capex_pct_revenue: np.ndarray
    nwc_pct_revenue: np.ndarray
    base_rate: np.ndarray
    exit_multiple: np.ndarray

    def __post_init__(self) -> None:
        n_scenarios, n_periods = self.revenue_growth.shape
        per_period = (
            self.revenue_growth,
            self.ebitda_margin,
            self.capex_pct_revenue,
            self.nwc_pct_revenue,
            self.base_rate,
        )
        names = (
            "revenue_growth",
            "ebitda_margin",
            "capex_pct_revenue",
            "nwc_pct_revenue",
            "base_rate",
        )
        for name, arr in zip(names, per_period, strict=True):
            if arr.shape != (n_scenarios, n_periods):
                raise ValueError(
                    f"DriverSet.{name} has shape {arr.shape}, expected {(n_scenarios, n_periods)}"
                )
        if self.exit_multiple.shape != (n_scenarios,):
            raise ValueError(
                f"DriverSet.exit_multiple has shape {self.exit_multiple.shape}, "
                f"expected {(n_scenarios,)}"
            )

    @property
    def n_scenarios(self) -> int:
        return self.revenue_growth.shape[0]

    @property
    def n_periods(self) -> int:
        return self.revenue_growth.shape[1]
