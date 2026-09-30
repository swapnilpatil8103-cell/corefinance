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
    "aggregate_long": {"color": "tab:purple", "linestyle": "--", "marker": "d", "markersize": 3},
    "anchored_to_aggregate": {
        "color": "tab:brown",
        "linestyle": "--",
        "marker": "v",
        "markersize": 3,
    },
    # Stage 5's projection charts reuse this same style lookup keyed by
    # SCENARIO name instead of model family -- matching the macro chart's
    # own baseline/severely_adverse colors (_SERIES_STYLE above).
    "baseline": {"color": "tab:blue", "linestyle": "--", "marker": "o", "markersize": 3},
    "severely_adverse": {"color": "tab:red", "linestyle": "--", "marker": "o", "markersize": 3},
}
_MODEL_FAMILY_LABELS = {
    "aggregate_ar": "Aggregate AR",
    "panel_fe": "Bank panel (FE)",
    "gbm": "Gradient boosting",
    "aggregate_long": "Aggregate (long history)",
    "anchored_to_aggregate": "Anchored to aggregate",
    "baseline": "Baseline",
    "severely_adverse": "Severely Adverse",
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


def _plot_backtest_on_axis(
    ax, actual: pd.Series, forecasts: dict[str, pd.Series], label_fontsize: str = "small"
) -> bool:
    """Plots the actual rate (solid black) and each model family's
    forecast (its own dashed/marked color) onto `ax`, restricted to the
    union of quarters the forecasts actually cover. Returns False (and
    plots nothing) if `forecasts` is empty -- the caller decides what to
    do with an empty subplot/chart."""
    forecast_quarters = sorted({q for series in forecasts.values() for q in series.index})
    if not forecast_quarters:
        return False
    window_start, window_end = forecast_quarters[0], forecast_quarters[-1]
    actual_window = actual[(actual.index >= window_start) & (actual.index <= window_end)]

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
    ax.tick_params(axis="x", rotation=90, labelsize="xx-small")
    ax.legend(loc="best", fontsize=label_fontsize)
    return True


def render_backtest_chart(
    actual: pd.Series, forecasts: dict[str, pd.Series], title: str, y_label: str, path: str
) -> None:
    """actual: a quarter-indexed (PeriodIndex) Series of the real industry
    rate over (at least) the backtest window. forecasts: {model family
    name -- "aggregate_ar"/"panel_fe"/"gbm"/"aggregate_long"/
    "anchored_to_aggregate" -- -> quarter-indexed Series of that family's
    forecast over the SAME window}. Writes a single PNG to `path`: the
    actual rate in solid black, each model family's forecast in its own
    dashed/marked color, restricted to the union of quarters the
    forecasts actually cover (so the x-axis doesn't stretch back over
    decades of unrelated history)."""
    fig, ax = plt.subplots(figsize=(9, 5))
    if not _plot_backtest_on_axis(ax, actual, forecasts):
        plt.close(fig)
        return
    ax.set_xlabel("Quarter")
    ax.set_ylabel(y_label)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def render_backtest_chart_grid(
    entries: list[tuple[str, pd.Series, dict[str, pd.Series]]], path: str, n_cols: int = 3
) -> None:
    """entries: [(title, actual, forecasts), ...] -- same shapes
    `render_backtest_chart` takes, one entry per category/dependent.
    Writes ONE PNG to `path` with every entry as its own subplot in an
    `n_cols`-wide grid, for a compact side-by-side comparison across
    categories (the brief's "chart grid" alongside the one-page backtest
    summary CSV). Entries with no forecast data are skipped (their
    subplot is left blank, not filled with a misleading empty chart)."""
    if not entries:
        return
    n_cols = max(1, min(n_cols, len(entries)))
    n_rows = -(-len(entries) // n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.5 * n_cols, 3.5 * n_rows), squeeze=False)
    for index, (title, actual, forecasts) in enumerate(entries):
        ax = axes[index // n_cols][index % n_cols]
        plotted = _plot_backtest_on_axis(ax, actual, forecasts, label_fontsize="xx-small")
        ax.set_title(title, fontsize="small")
        if not plotted:
            ax.axis("off")
    for index in range(len(entries), n_rows * n_cols):
        axes[index // n_cols][index % n_cols].axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
