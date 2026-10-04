"""Stage 7: five charts for one `corefin ma run` -- EPS accretion by
year, TBV earnback (standalone vs. pro forma TBV/share path), pro forma
CET1 baseline vs. severely adverse, a tornado, and a price x cost-save
heatmap. Matplotlib is the one plotting dependency in this project
(`optimize/heatmap.py`, `simulate/charts.py`); this module follows the
SAME headless-backend/`fig.savefig(path, dpi=150)`/`plt.close(fig)`
convention, not a new one. Every label here is whatever the caller's
own `acquirer_label`/`target_label` strings are -- never a hardcoded
real bank name (see `excel_export.py`'s own module docstring)."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")  # headless: writing PNG files, no display needed

import matplotlib.pyplot as plt
import numpy as np

from corefin.ma.accretion import EpsAccretionResult, TbvEarnbackResult
from corefin.ma.sensitivity import GridResult, TornadoResult
from corefin.ma.stress import StressTestResult

BASEL_III_CET1_MINIMUM_RATIO = 0.045


def render_eps_accretion_chart(eps: EpsAccretionResult, path: str) -> None:
    """Grouped bars per year: standalone EPS vs. GAAP pro forma EPS."""
    annual = eps.annual()
    years = list(annual.keys())
    standalone = [a.standalone_eps for a in annual.values()]
    pro_forma = [a.pro_forma_eps_gaap for a in annual.values()]

    x = np.arange(len(years))
    width = 0.35
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - width / 2, standalone, width, label="Standalone EPS", color="tab:gray")
    ax.bar(x + width / 2, pro_forma, width, label="Pro forma EPS (GAAP)", color="tab:blue")
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(years)
    ax.set_ylabel("EPS ($/share)")
    ax.set_title("EPS Accretion/Dilution by Year")
    ax.legend(loc="best", fontsize="small")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def render_tbv_earnback_chart(tbv: TbvEarnbackResult, path: str) -> None:
    """Standalone vs. pro forma TBV/share path, with a marker at the
    earnback crossover quarter (if reached within the horizon)."""
    periods = np.arange(len(tbv.standalone_tbv_per_share))
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.plot(periods, tbv.standalone_tbv_per_share, label="Standalone TBV/share", color="tab:gray")
    ax.plot(periods, tbv.pro_forma_tbv_per_share, label="Pro forma TBV/share", color="tab:blue")
    if tbv.earnback_period_index is not None:
        ax.axvline(
            tbv.earnback_period_index,
            color="tab:green",
            linestyle="--",
            linewidth=1,
            label=f"Earnback ({tbv.earnback_label})",
        )
    ax.set_xlabel("Quarter (0 = close)")
    ax.set_ylabel("TBV per share ($)")
    ax.set_title(
        f"TBV Dilution & Earnback (at close: {tbv.tbv_dilution_at_close_pct:+.1%}, "
        f"{tbv.earnback_label})"
    )
    ax.legend(loc="best", fontsize="small")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def render_stress_cet1_chart(stress_results: dict[str, StressTestResult], path: str) -> None:
    """Pro forma combined entity's CET1 ratio path under baseline vs.
    severely adverse, with a horizontal line at the Basel III 4.5%
    minimum."""
    fig, ax = plt.subplots(figsize=(9, 5.5))
    colors = {"baseline": "tab:blue", "severely_adverse": "tab:red"}
    for scenario_name, result in stress_results.items():
        path_values = result.pro_forma_combined.cet1_ratio
        ax.plot(
            np.arange(len(path_values)),
            path_values,
            label=f"Pro forma combined ({scenario_name})",
            color=colors.get(scenario_name, "tab:gray"),
        )
    ax.axhline(
        BASEL_III_CET1_MINIMUM_RATIO,
        color="black",
        linestyle="--",
        linewidth=1,
        label="Basel III 4.5% minimum",
    )
    ax.set_xlabel("Quarter (0 = jump-off)")
    ax.set_ylabel("CET1 ratio")
    ax.set_title("Pro Forma CET1: Baseline vs. Severely Adverse")
    ax.legend(loc="best", fontsize="small")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def render_sensitivity_tornado_chart(
    tornado: TornadoResult, metric: str, metric_label: str, path: str
) -> None:
    """One bar per driver, low-to-high range, sorted widest-first (top),
    matching `simulate.charts.render_tornado_chart`'s own convention.
    Rows where either end is `None` (not meaningful) are skipped."""
    rows = [
        row
        for row in tornado.sorted_by(metric)
        if getattr(row.low_metrics, metric) is not None
        and getattr(row.high_metrics, metric) is not None
    ]
    fig, ax = plt.subplots(figsize=(9, 5.5))
    y_pos = np.arange(len(rows))
    for i, row in enumerate(rows):
        low = getattr(row.low_metrics, metric)
        high = getattr(row.high_metrics, metric)
        lo, hi = sorted((low, high))
        ax.barh(len(rows) - 1 - i, hi - lo, left=lo, color="tab:blue", alpha=0.75)
    ax.set_yticks(y_pos)
    ax.set_yticklabels([row.driver_name for row in reversed(rows)])
    base_value = getattr(tornado.base_metrics, metric)
    if base_value is not None:
        ax.axvline(base_value, color="black", linestyle="--", linewidth=1, label="Base case")
    ax.set_xlabel(metric_label)
    ax.set_title(f"Tornado: {metric_label} sensitivity")
    ax.legend(loc="best", fontsize="small")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def render_price_cost_save_heatmap(grid: GridResult, path: str) -> None:
    """Year-2 EPS accretion (excl. one-time) over price x cost saves,
    following `optimize.heatmap.render_objective_heatmap`'s own
    continuous-colormap convention."""
    fig, ax = plt.subplots(figsize=(7, 5.5))
    masked = np.ma.masked_invalid(grid.year2_accretion_pct)
    mesh = ax.pcolormesh(masked, cmap="RdYlGn", shading="auto")
    ax.set_xticks(np.arange(len(grid.price_values)) + 0.5)
    ax.set_xticklabels([f"{v:.2f}x" for v in grid.price_values])
    ax.set_yticks(np.arange(len(grid.cost_save_values)) + 0.5)
    ax.set_yticklabels([f"{v:.0%}" for v in grid.cost_save_values])
    ax.set_xlabel("Price (x TBV)")
    ax.set_ylabel("Cost saves (% of target noninterest expense)")
    ax.set_title("Year-2 EPS Accretion (excl. one-time) by Price x Cost Saves")
    fig.colorbar(mesh, ax=ax, label="Year-2 accretion")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
