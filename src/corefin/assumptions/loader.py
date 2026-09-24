"""YAML -> RootConfig loading, and expansion of scalar-or-series fields."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import yaml

from corefin.assumptions.schema import RootConfig, ScalarOrSeries


def load_config(path: str | Path) -> RootConfig:
    with Path(path).open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return RootConfig.model_validate(raw)


def expand_series(value: ScalarOrSeries, n_periods: int, field_name: str = "") -> np.ndarray:
    """A float broadcasts flat across all periods; a list must match n_periods exactly."""
    if isinstance(value, list):
        if len(value) != n_periods:
            raise ValueError(
                f"{field_name or 'series'} has length {len(value)}, expected n_periods={n_periods}"
            )
        return np.asarray(value, dtype=float)
    return np.full(n_periods, float(value))
