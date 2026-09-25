"""Two charts for "what's limiting the optimum":

- `render_objective_heatmap`: the objective surface over two decision
  variables. Feasible cells use a continuous colormap; infeasible cells are
  colored by *which* constraint first made them infeasible (a categorical
  colormap with a legend), not plain gray -- so the shape of the infeasible
  region is informative, not just its existence.
- `render_leverage_frontier`: best feasible objective achievable at each
  level of total closing leverage actually reached in the grid/refinement
  pool, with the dominant limiting constraint labeled along the infeasible
  tail -- the direct "what stops us adding more debt" picture.

Matplotlib is the one plotting dependency in this project, added
specifically for this.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")  # headless: writing PNG files, no display needed

import matplotlib.colors
import matplotlib.pyplot as plt
import numpy as np

from corefin.optimize.search import OptimizationResult

_INFEASIBLE_PALETTE = (
    "#d62728",  # red
    "#9467bd",  # purple
    "#8c564b",  # brown
    "#e377c2",  # pink
    "#7f7f7f",  # gray
    "#bcbd22",  # olive
    "#17becf",  # cyan
)


def render_objective_heatmap(
    optimization_result: OptimizationResult,
    path: str,
    x_name: str | None = None,
    y_name: str | None = None,
) -> None:
    """Objective value over the two named decision variables (default: the
    first two, in config order -- "the two main tranche sizes"). Any other
    decision variable is held at the recommended structure's value, so this
    is a 2D profile slice through the full grid whenever there are more than
    two decision variables."""
    grid_result = optimization_result.grid_result
    names = grid_result.decision_variable_names
    if len(names) < 2:
        raise ValueError("heatmap needs at least two decision variables")
    x_name = x_name or names[0]
    y_name = y_name or names[1]

    recommended_values = optimization_result.recommended.candidate.decision_values
    other_names = [n for n in names if n not in (x_name, y_name)]

    x_values = np.sort(np.unique(grid_result.grid_values[x_name]))
    y_values = np.sort(np.unique(grid_result.grid_values[y_name]))
    objective_grid = np.full((len(y_values), len(x_values)), np.nan)
    reason_grid = np.full((len(y_values), len(x_values)), np.nan)
    x_index = {round(v, 9): i for i, v in enumerate(x_values)}
    y_index = {round(v, 9): i for i, v in enumerate(y_values)}

    reason_codes: dict[str, int] = {}
    for evaluation in grid_result.evaluations:
        dv = evaluation.candidate.decision_values
        if any(abs(dv[n] - recommended_values[n]) > 1e-6 for n in other_names):
            continue
        xi = x_index.get(round(dv[x_name], 9))
        yi = y_index.get(round(dv[y_name], 9))
        if xi is None or yi is None:
            continue
        if evaluation.feasible:
            objective_grid[yi, xi] = evaluation.objective_value
        elif evaluation.violations:
            reason = evaluation.violations[0].name
            if reason not in reason_codes:
                reason_codes[reason] = len(reason_codes)
            reason_grid[yi, xi] = reason_codes[reason]

    fig, ax = plt.subplots(figsize=(8, 6))
    feasible_masked = np.ma.masked_invalid(objective_grid)
    mesh = ax.pcolormesh(x_values, y_values, feasible_masked, cmap="viridis", shading="nearest")
    fig.colorbar(mesh, ax=ax, label="Objective (IRR)")

    if reason_codes:
        infeasible_masked = np.ma.masked_invalid(reason_grid)
        colors = [
            _INFEASIBLE_PALETTE[i % len(_INFEASIBLE_PALETTE)] for i in range(len(reason_codes))
        ]
        infeasible_cmap = matplotlib.colors.ListedColormap(colors)
        ax.pcolormesh(
            x_values,
            y_values,
            infeasible_masked,
            cmap=infeasible_cmap,
            vmin=-0.5,
            vmax=len(reason_codes) - 0.5,
            shading="nearest",
        )
        legend_handles = [
            plt.Rectangle((0, 0), 1, 1, color=colors[code]) for code in reason_codes.values()
        ]
        legend_labels = list(reason_codes.keys())
    else:
        legend_handles, legend_labels = [], []

    star = ax.scatter(
        [recommended_values[x_name]],
        [recommended_values[y_name]],
        marker="*",
        s=300,
        color="gold",
        edgecolor="black",
        zorder=5,
    )
    legend_handles.append(star)
    legend_labels.append("Recommended")

    ax.set_xlabel(f"{x_name} (x entry EBITDA)")
    ax.set_ylabel(f"{y_name} (x entry EBITDA)")
    ax.set_title("Objective value by tranche size (colored = infeasible, by limiting constraint)")
    ax.legend(legend_handles, legend_labels, loc="best", fontsize="small")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def render_leverage_frontier(optimization_result: OptimizationResult, path: str) -> None:
    """Best objective at each level of total closing leverage actually
    reached in the grid + refinement pool. Feasible levels trace the
    achievable frontier; beyond it, the best *attempted* (but infeasible)
    objective is shown as a dashed tail, labeled with the constraint that
    blocked the most candidates at each leverage level."""
    grid_result = optimization_result.grid_result
    pool = list(grid_result.evaluations)
    if optimization_result.refinement_result is not None:
        pool += optimization_result.refinement_result.evaluations
    pool = [e for e in pool if e.result is not None or e.feasible]
    # Bin by leverage rounded to 2dp -- fine enough resolution given the
    # grid's own step size, coarse enough to merge near-duplicate candidates.
    by_leverage: dict[float, list] = {}
    for e in pool:
        lev = round(e.candidate.leverage.total_leverage, 2)
        by_leverage.setdefault(lev, []).append(e)

    levels = sorted(by_leverage.keys())
    feasible_x, feasible_y = [], []
    infeasible_x, infeasible_y, infeasible_label = [], [], []
    for lev in levels:
        evaluations = by_leverage[lev]
        feasible = [e for e in evaluations if e.feasible]
        if feasible:
            feasible_x.append(lev)
            feasible_y.append(max(e.objective_value for e in feasible))
        else:
            attempted = [e for e in evaluations if e.result is not None]
            if not attempted:
                continue
            best = max(attempted, key=lambda e: e.objective_value)
            infeasible_x.append(lev)
            infeasible_y.append(best.objective_value)
            reason_counts: dict[str, int] = {}
            for e in evaluations:
                for v in e.violations:
                    reason_counts[v.name] = reason_counts.get(v.name, 0) + 1
            infeasible_label.append(
                max(reason_counts, key=reason_counts.get) if reason_counts else ""
            )

    fig, ax = plt.subplots(figsize=(9, 5.5))
    if feasible_x:
        ax.plot(
            feasible_x, feasible_y, marker="o", color="tab:green", label="Feasible (achievable)"
        )
    if infeasible_x:
        ax.plot(
            infeasible_x,
            infeasible_y,
            marker="x",
            linestyle="--",
            color="tab:red",
            label="Infeasible (best attempted)",
        )
        # Label a handful of points along the infeasible tail, not every one.
        step = max(1, len(infeasible_x) // 5)
        for i in range(0, len(infeasible_x), step):
            ax.annotate(
                infeasible_label[i],
                (infeasible_x[i], infeasible_y[i]),
                textcoords="offset points",
                xytext=(0, 8),
                ha="center",
                fontsize="x-small",
                rotation=20,
            )

    recommended_leverage = optimization_result.recommended.candidate.leverage.total_leverage
    recommended_objective = optimization_result.recommended.objective_value
    ax.scatter(
        [recommended_leverage],
        [recommended_objective],
        marker="*",
        s=300,
        color="gold",
        edgecolor="black",
        zorder=5,
        label="Recommended",
    )

    ax.set_xlabel("Total closing leverage (x EBITDA)")
    ax.set_ylabel("Best objective (IRR)")
    ax.set_title("Best achievable objective by leverage -- what stops us adding more debt")
    ax.legend(loc="best", fontsize="small")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
