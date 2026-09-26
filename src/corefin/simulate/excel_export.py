"""Excel export for the Sponsor LBO Monte Carlo engine: summary,
distributions, stress scenarios, breach/distress by year, attribution,
driver importance and structure comparison sheets. Stress scenarios,
breach/distress-by-year, attribution and driver importance are reported
for one "primary" structure (the optimizer's recommendation when
available, else the input structure) -- comparable across every metric,
that structure is what the rest of the sheets analyze in depth.
"""

from __future__ import annotations

import pandas as pd

from corefin.simulate.analytics import DownsideAnalytics
from corefin.simulate.attribution import BridgeSummary
from corefin.simulate.compare import StructureComparisonEntry
from corefin.simulate.importance import DriverImportance, TornadoBar
from corefin.simulate.stress import StressScenarioResult
from corefin.timeline import Timeline


def _summary_frame(entries: list[StructureComparisonEntry]) -> pd.DataFrame:
    rows = {}
    for e in entries:
        r = e.downside.returns
        rows[e.name] = {
            "Mean IRR": r.mean_irr,
            "Median IRR": r.median_irr,
            "Mean MOIC": r.mean_moic,
            "Median MOIC": r.median_moic,
            "Expected Shortfall (worst 5%)": r.expected_shortfall_irr_5pct,
            "Expected Shortfall (worst 10%)": r.expected_shortfall_irr_10pct,
            "P(MOIC < 1.0x)": r.prob_moic_below_1,
            f"P(IRR < {r.irr_hurdle:.0%} hurdle)": r.prob_irr_below_hurdle,
            "Covenant Breach Probability": e.downside.covenant_breaches.overall_breach_probability,
            "Distress Probability": e.downside.distress.overall_probability,
        }
    return pd.DataFrame(rows)


def _distributions_frame(entries: list[StructureComparisonEntry]) -> pd.DataFrame:
    rows = {}
    for e in entries:
        r = e.downside.returns
        row = {f"IRR p{p}": v for p, v in r.percentiles_irr.items()}
        row.update({f"MOIC p{p}": v for p, v in r.percentiles_moic.items()})
        rows[e.name] = row
    return pd.DataFrame(rows)


def _stress_frame(stress_results: list[StressScenarioResult]) -> pd.DataFrame:
    rows = [
        {
            "Scenario": s.name,
            "IRR": s.irr,
            "MOIC": s.moic,
            "Min Liquidity ($mm)": s.min_liquidity_mm,
            "Min Cash ($mm)": s.min_cash_mm,
            "Max Revolver Draw %": s.max_revolver_draw_pct,
            "Distress": s.distress_overall,
        }
        for s in stress_results
    ]
    return pd.DataFrame(rows)


def _breach_distress_frame(timeline: Timeline, downside: DownsideAnalytics) -> pd.DataFrame:
    data = {}
    for name, arr in downside.covenant_breaches.breach_probability_by_year.items():
        data[f"{name} Breach Probability"] = arr
    if downside.covenant_breaches.combined_breach_probability_by_year.size:
        data["Any Covenant Breach Probability"] = (
            downside.covenant_breaches.combined_breach_probability_by_year
        )
    data["Distress Probability"] = downside.distress.probability_by_year
    return pd.DataFrame(data, index=timeline.year_labels)


def _attribution_frame(bridge_summaries: list[BridgeSummary]) -> pd.DataFrame:
    rows = {}
    for s in bridge_summaries:
        rows[s.label] = {
            "EBITDA Growth": s.ebitda_growth,
            "Multiple Change": s.multiple_change,
            "Debt Paydown / Cash Generation": s.debt_paydown_and_cash,
            "Fees & Leakage": s.fees_and_leakage,
            "Total Value Creation": s.total_value_creation,
        }
    return pd.DataFrame(rows)


def _importance_frame(
    driver_importance: list[DriverImportance], tornado: list[TornadoBar]
) -> pd.DataFrame:
    tornado_by_name = {b.name: b for b in tornado}
    rows = []
    for imp in driver_importance:
        bar = tornado_by_name.get(imp.name)
        rows.append(
            {
                "Driver": imp.name,
                "Standardized Coefficient": imp.standardized_coefficient,
                "Rank Correlation": imp.rank_correlation,
                "IRR at P10": bar.irr_at_p10 if bar else None,
                "Base IRR": bar.base_irr if bar else None,
                "IRR at P90": bar.irr_at_p90 if bar else None,
                "IRR Range": bar.irr_range if bar else None,
            }
        )
    return pd.DataFrame(rows)


def _structure_comparison_frame(entries: list[StructureComparisonEntry]) -> pd.DataFrame:
    rows = {}
    for e in entries:
        row = {
            "Total Leverage": e.candidate.leverage.total_leverage,
            "Secured Leverage": e.candidate.leverage.secured_leverage,
        }
        for name, multiple in e.candidate.decision_values.items():
            row[f"{name} (x EBITDA)"] = multiple
        row["Mean IRR"] = e.downside.returns.mean_irr
        row["Expected Shortfall (worst 10%)"] = e.downside.returns.expected_shortfall_irr_10pct
        row["P(MOIC < 1.0x)"] = e.downside.returns.prob_moic_below_1
        rows[e.name] = row
    return pd.DataFrame(rows)


def export_simulation_to_excel(
    path: str,
    timeline: Timeline,
    entries: list[StructureComparisonEntry],
    primary_name: str,
    bridge_summaries: list[BridgeSummary],
    driver_importance: list[DriverImportance],
    tornado: list[TornadoBar],
) -> None:
    primary = next(e for e in entries if e.name == primary_name)
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        _summary_frame(entries).to_excel(writer, sheet_name="Summary")
        _distributions_frame(entries).to_excel(writer, sheet_name="Distributions")
        _stress_frame(primary.stress_results).to_excel(
            writer, sheet_name="Stress Scenarios", index=False
        )
        _breach_distress_frame(timeline, primary.downside).to_excel(
            writer, sheet_name="Breach & Distress by Year"
        )
        _attribution_frame(bridge_summaries).to_excel(writer, sheet_name="Attribution")
        _importance_frame(driver_importance, tornado).to_excel(
            writer, sheet_name="Driver Importance", index=False
        )
        _structure_comparison_frame(entries).to_excel(writer, sheet_name="Structure Comparison")
