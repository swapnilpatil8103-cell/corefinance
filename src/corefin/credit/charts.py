"""Charts for the Credit-Loss Forecasting Engine's macro/scenario data:
one chart per macro variable, showing its full FRED history with the
Fed's baseline and severely-adverse scenario paths appended, continuing
from the last actual quarter.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")  # headless: writing PNG files, no display needed

import matplotlib.pyplot as plt

from corefin.credit.macro import build_macro_chart_series

_SERIES_STYLE = {
    "actual": {"color": "black", "linestyle": "-", "linewidth": 1.75},
    "baseline": {"color": "tab:blue", "linestyle": "--", "linewidth": 1.5},
    "severely_adverse": {"color": "tab:red", "linestyle": "--", "linewidth": 1.5},
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
