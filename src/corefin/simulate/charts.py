"""Charts for the Sponsor LBO Monte Carlo engine: IRR histogram, leverage
fan chart, breach/distress-by-year, value creation bridge waterfall,
tornado chart, and a return-vs-risk comparison across structures.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")  # headless: writing PNG files, no display needed

import matplotlib.pyplot as plt
import numpy as np

from corefin.simulate.analytics import DownsideAnalytics
from corefin.simulate.attribution import BridgeSummary
from corefin.simulate.compare import StructureComparisonEntry
from corefin.simulate.importance import TornadoBar
from corefin.timeline import Timeline


def render_irr_histogram(downside: DownsideAnalytics, irr: np.ndarray, path: str) -> None:
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.hist(irr, bins=50, color="tab:blue", alpha=0.75)
    y_top = ax.get_ylim()[1]
    for p, v in downside.returns.percentiles_irr.items():
        ax.axvline(v, linestyle="--", color="gray", alpha=0.6)
        ax.annotate(
            f"p{p}", (v, y_top * 0.97), rotation=90, fontsize="x-small", ha="right", va="top"
        )
    ax.axvline(
        downside.returns.irr_hurdle,
        color="red",
        linewidth=2,
        label=f"Hurdle ({downside.returns.irr_hurdle:.0%})",
    )
    ax.set_xlabel("IRR")
    ax.set_ylabel("Scenario count")
    ax.set_title("IRR distribution")
    ax.legend(loc="best", fontsize="small")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def render_leverage_fan_chart(downside: DownsideAnalytics, timeline: Timeline, path: str) -> None:
    bands = downside.leverage_bands
    years = timeline.year_labels
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.fill_between(
        years,
        bands.percentiles[5],
        bands.percentiles[95],
        alpha=0.15,
        color="tab:blue",
        label="p5-p95",
    )
    ax.fill_between(
        years,
        bands.percentiles[10],
        bands.percentiles[90],
        alpha=0.25,
        color="tab:blue",
        label="p10-p90",
    )
    ax.fill_between(
        years,
        bands.percentiles[25],
        bands.percentiles[75],
        alpha=0.35,
        color="tab:blue",
        label="p25-p75",
    )
    ax.plot(years, bands.mean, color="tab:blue", linewidth=2, label="Mean")
    ax.plot(
        years, bands.percentiles[50], color="black", linewidth=1.5, linestyle="--", label="Median"
    )
    ax.set_xlabel("Year")
    ax.set_ylabel("Total net leverage (x EBITDA)")
    ax.set_title("Net leverage fan chart")
    ax.legend(loc="best", fontsize="small")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def render_breach_distress_chart(
    downside: DownsideAnalytics, timeline: Timeline, path: str
) -> None:
    years = timeline.year_labels
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for name, arr in downside.covenant_breaches.breach_probability_by_year.items():
        ax.plot(years, arr, marker="o", label=f"{name} breach")
    if downside.distress.probability_by_year.size:
        ax.plot(
            years,
            downside.distress.probability_by_year,
            marker="s",
            color="black",
            linestyle="--",
            label="Distress",
        )
    ax.set_xlabel("Year")
    ax.set_ylabel("Probability")
    ax.set_ylim(bottom=0)
    ax.set_title("Covenant breach and distress probability by year")
    ax.legend(loc="best", fontsize="small")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def render_value_bridge_waterfall(summary: BridgeSummary, path: str) -> None:
    labels = ["EBITDA\nGrowth", "Multiple\nChange", "Debt Paydown\n/ Cash", "Fees &\nLeakage"]
    values = [
        summary.ebitda_growth,
        summary.multiple_change,
        summary.debt_paydown_and_cash,
        summary.fees_and_leakage,
    ]
    cumulative = np.concatenate([[0.0], np.cumsum(values)])

    fig, ax = plt.subplots(figsize=(9, 5.5))
    for i, (label, val) in enumerate(zip(labels, values, strict=True)):
        bottom = min(cumulative[i], cumulative[i] + val)
        color = "tab:green" if val >= 0 else "tab:red"
        ax.bar(label, abs(val), bottom=bottom, color=color)
    ax.bar("Total Value\nCreation", summary.total_value_creation, color="tab:blue")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylabel("$mm")
    ax.set_title(f"Value creation bridge ({summary.label})")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def render_tornado_chart(bars: list[TornadoBar], path: str) -> None:
    sorted_bars = sorted(bars, key=lambda b: b.irr_range)  # smallest at bottom, largest at top
    fig, ax = plt.subplots(figsize=(9, 5.5))
    y_pos = np.arange(len(sorted_bars))
    for i, bar in enumerate(sorted_bars):
        low, high = sorted((bar.irr_at_p10, bar.irr_at_p90))
        ax.barh(i, high - low, left=low, color="tab:blue", alpha=0.75)
    ax.set_yticks(y_pos)
    ax.set_yticklabels([b.name for b in sorted_bars])
    if sorted_bars:
        ax.axvline(
            sorted_bars[0].base_irr, color="black", linestyle="--", linewidth=1, label="Base IRR"
        )
    ax.set_xlabel("IRR")
    ax.set_title("Tornado chart: IRR sensitivity to each driver (p10 to p90)")
    ax.legend(loc="best", fontsize="small")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def render_structure_comparison_chart(entries: list[StructureComparisonEntry], path: str) -> None:
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for entry in entries:
        risk = entry.downside.returns.expected_shortfall_irr_10pct
        ret = entry.downside.returns.mean_irr
        ax.scatter(risk, ret, s=140, zorder=3)
        ax.annotate(
            entry.name, (risk, ret), textcoords="offset points", xytext=(8, 6), fontsize="small"
        )
    ax.set_xlabel("Expected shortfall, worst 10% (IRR) -- risk")
    ax.set_ylabel("Mean IRR -- return")
    ax.set_title("Return vs risk across structures")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
