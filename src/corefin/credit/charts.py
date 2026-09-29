"""Charts for the Credit-Loss Forecasting Engine: one chart per macro
variable (history + the Fed's scenario paths), and actual-vs-predicted
backtest charts per category/dependent variable (Stage 4).
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")  # headless: writing PNG files, no display needed

import matplotlib.pyplot as plt
import pandas as pd

from corefin.credit.macro import build_macro_chart_series

_SERIES_STYLE = {
    "actual": {"color": "black", "linestyle": "-", "linewidth": 1.75},
    "baseline": {"color": "tab:blue", "linestyle": "--", "linewidth": 1.5},
    "severely_adverse": {"color": "tab:red", "linestyle": "--", "linewidth": 1.5},
}

_MODEL_FAMILY_STYLE = {
    "aggregate_ar": {"color": "tab:blue", "linestyle": "--", "marker": "o", "markersize": 3},
    "panel_fe": {"color": "tab:orange", "linestyle": "--", "marker": "s", "markersize": 3},
    "gbm": {"color": "tab:green", "linestyle": "--", "marker": "^", "markersize": 3},
}
_MODEL_FAMILY_LABELS = {
    "aggregate_ar": "Aggregate AR",
    "panel_fe": "Bank panel (FE)",
    "gbm": "Gradient boosting",
}


def render_macro_variable_chart(history, scenarios, variable: str, path: str) -> None:
    """history/scenarios: as taken by macro.build_macro_chart_series.
    Writes a single PNG to `path`: the variable's actual history in solid
    black, each scenario continuing from the last actual quarter in a
    dashed color."""
    long_frame = build_macro_chart_series(history, scenarios, variable)

    fig, ax = plt.subplots(figsize=(9, 5))
    for series_name, group in long_frame.groupby("series", sort=False):
        x = group["quarter"].astype(str)
        style = _SERIES_STYLE.get(series_name, {})
        ax.plot(x, group["value"], label=series_name.replace("_", " ").title(), **style)

    tick_step = max(len(long_frame["quarter"].unique()) // 20, 1)
    ax.set_xticks(ax.get_xticks()[::tick_step])
    ax.tick_params(axis="x", rotation=90, labelsize="x-small")
    ax.set_xlabel("Quarter")
    ax.set_ylabel(variable)
    ax.set_title(variable)
    ax.legend(loc="best", fontsize="small")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def render_backtest_chart(
    actual: pd.Series, forecasts: dict[str, pd.Series], title: str, y_label: str, path: str
) -> None:
    """actual: a quarter-indexed (PeriodIndex) Series of the real industry
    rate over (at least) the backtest window. forecasts: {model family
    name -- "aggregate_ar"/"panel_fe"/"gbm" -- -> quarter-indexed Series
    of that family's forecast over the SAME window}. Writes a single PNG
    to `path`: the actual rate in solid black, each model family's
    forecast in its own dashed/marked color, restricted to the union of
    quarters the forecasts actually cover (so the x-axis doesn't stretch
    back over decades of unrelated history)."""
    forecast_quarters = sorted({q for series in forecasts.values() for q in series.index})
    if not forecast_quarters:
        return
    window_start, window_end = forecast_quarters[0], forecast_quarters[-1]
    actual_window = actual[(actual.index >= window_start) & (actual.index <= window_end)]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(
        actual_window.index.astype(str),
        actual_window.to_numpy(),
        label="Actual",
        color="black",
        linewidth=1.75,
    )
    for family, series in forecasts.items():
        style = _MODEL_FAMILY_STYLE.get(family, {})
        label = _MODEL_FAMILY_LABELS.get(family, family)
        ax.plot(series.index.astype(str), series.to_numpy(), label=label, **style)

    ax.tick_params(axis="x", rotation=90, labelsize="x-small")
    ax.set_xlabel("Quarter")
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.legend(loc="best", fontsize="small")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
