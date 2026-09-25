"""Heatmap of the objective surface over two decision variables, with
infeasible regions masked and the optimum marked. Matplotlib is the one
plotting dependency in this project, added specifically for this.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")  # headless: writing a PNG file, no display needed

import matplotlib.pyplot as plt
import numpy as np

from corefin.optimize.search import OptimizationResult


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
    x_index = {round(v, 9): i for i, v in enumerate(x_values)}
    y_index = {round(v, 9): i for i, v in enumerate(y_values)}

    for evaluation in grid_result.evaluations:
        dv = evaluation.candidate.decision_values
        if any(abs(dv[n] - recommended_values[n]) > 1e-6 for n in other_names):
            continue
        xi = x_index.get(round(dv[x_name], 9))
        yi = y_index.get(round(dv[y_name], 9))
        if xi is None or yi is None or not evaluation.feasible:
            continue
        objective_grid[yi, xi] = evaluation.objective_value

    fig, ax = plt.subplots(figsize=(8, 6))
    masked = np.ma.masked_invalid(objective_grid)
    cmap = plt.get_cmap("viridis").with_extremes(bad="lightgray")
    mesh = ax.pcolormesh(x_values, y_values, masked, cmap=cmap, shading="nearest")
    fig.colorbar(mesh, ax=ax, label="Objective (IRR)")

    ax.scatter(
        [recommended_values[x_name]],
        [recommended_values[y_name]],
        marker="*",
        s=300,
        color="red",
        edgecolor="black",
        zorder=5,
        label="Recommended",
    )

    ax.set_xlabel(f"{x_name} (x entry EBITDA)")
    ax.set_ylabel(f"{y_name} (x entry EBITDA)")
    ax.set_title("Objective value by tranche size (gray = infeasible)")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
