"""Out-of-time backtest harness for Stage 4's model families: a hard
calendar split into train/test windows (never a random row split, which
would leak future quarters into training), and forecast-quality metrics
(RMSE/MAE, peak-rate error, peak-timing error) comparing a forecast
series against the real industry rate over the test window. No function
here touches the network or fits anything -- see models.py for the model
families themselves and cli.py for orchestration.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def split_out_of_time(
    dataset: pd.DataFrame,
    train_end: pd.Period,
    test_start: pd.Period,
    test_end: pd.Period,
    quarter_column: str = "quarter",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Returns (train, test): `train` = rows at or before `train_end`,
    `test` = rows in [test_start, test_end] (inclusive). This is a hard
    calendar cutoff, not a random split -- the defining property an
    out-of-time backtest needs (a model must never train on a quarter it
    is later scored on). `test_start` is ordinarily `train_end + 1`, but
    is taken as given rather than enforced here, since a model with its
    own lag structure (e.g. an AR term) legitimately needs `train_end` to
    be strictly before the forecast horizon by at least the lag length."""
    train = dataset[dataset[quarter_column] <= train_end]
    test = dataset[
        (dataset[quarter_column] >= test_start) & (dataset[quarter_column] <= test_end)
    ]
    return train, test


def forecast_error_metrics(actual: pd.Series, predicted: pd.Series) -> dict[str, float]:
    """RMSE and MAE between `actual` and `predicted`, aligned on their
    shared index (an outer join -- a quarter present in only one of the
    two contributes nothing, not a spurious error). Returns NaN metrics
    (n=0) if nothing overlaps, rather than raising."""
    aligned = pd.concat({"actual": actual, "predicted": predicted}, axis=1)
    valid = aligned.dropna()
    if valid.empty:
        return {"rmse": np.nan, "mae": np.nan, "n": 0}
    diff = valid["actual"] - valid["predicted"]
    return {
        "rmse": float(np.sqrt((diff**2).mean())),
        "mae": float(diff.abs().mean()),
        "n": int(len(valid)),
    }


def peak_metrics(actual: pd.Series, predicted: pd.Series) -> dict[str, float]:
    """Peak-rate error (predicted peak minus actual peak, in rate units --
    positive means the model overstates how bad the worst quarter gets)
    and peak-timing error (predicted peak's quarter minus actual peak's
    quarter, in whole quarters -- positive means the model calls the peak
    later than it really happened), both computed only over quarters
    where both `actual` and `predicted` are available (see
    `forecast_error_metrics`)."""
    aligned = pd.concat({"actual": actual, "predicted": predicted}, axis=1).dropna()
    if aligned.empty:
        return {
            "actual_peak_rate": np.nan,
            "predicted_peak_rate": np.nan,
            "peak_rate_error": np.nan,
            "peak_timing_error_quarters": np.nan,
        }
    actual_peak_quarter = aligned["actual"].idxmax()
    predicted_peak_quarter = aligned["predicted"].idxmax()
    actual_peak_rate = float(aligned.loc[actual_peak_quarter, "actual"])
    predicted_peak_rate = float(aligned.loc[predicted_peak_quarter, "predicted"])
    return {
        "actual_peak_rate": actual_peak_rate,
        "predicted_peak_rate": predicted_peak_rate,
        "peak_rate_error": predicted_peak_rate - actual_peak_rate,
        "peak_timing_error_quarters": int(
            predicted_peak_quarter.ordinal - actual_peak_quarter.ordinal
        ),
    }


def backtest_summary(actual: pd.Series, predicted: pd.Series) -> dict[str, float]:
    """Convenience: `forecast_error_metrics` and `peak_metrics` merged
    into one dict, the row shape `cli.py` writes to the backtest table."""
    return {**forecast_error_metrics(actual, predicted), **peak_metrics(actual, predicted)}
