"""Stage 7: Excel export for one `corefin ma run`, matching the rest of
corefin's own Excel export convention (`io/excel_export.py`, `credit/
excel_export.py`): one `pd.ExcelWriter(path, engine="openpyxl")`, a
`_<name>_frame(...) -> pd.DataFrame` helper per sheet. Every label here
is whatever the caller's own `acquirer_label`/`target_label` strings
are -- this project's standing rule (never a real bank name/ticker in
committed or displayed output) is the CALLER's responsibility (`corefin
ma run` always passes the YAML's own "Acquirer Bank A"/"Target Bank B"
-- or a `--local-config` override -- never a hardcoded real name)."""

from __future__ import annotations

import pandas as pd

from corefin.ma.accretion import EpsAccretionResult, TbvEarnbackResult
from corefin.ma.capital import ProFormaCapitalRatios
from corefin.ma.schema import DealConfig
from corefin.ma.sensitivity import GridResult, MonteCarloResult, TornadoResult
from corefin.ma.stress import StressTestResult


def _summary_frame(
    acquirer_label: str,
    target_label: str,
    config: DealConfig,
    capital: ProFormaCapitalRatios,
) -> pd.DataFrame:
    rows = {
        "Acquirer": acquirer_label,
        "Target": target_label,
        "Price (x TBV)": config.consideration.price_to_tbv,
        "Stock % of consideration": config.consideration.stock_pct,
        "Credit mark (%)": config.credit_mark.credit_mark_pct,
        "PCD share of credit mark": config.credit_mark.pcd_share,
        "Rate mark (%)": config.rate_mark.rate_mark_pct,
        "Securities mark (%)": config.securities_mark.securities_mark_pct,
        "CDI (% of core deposits)": config.cdi.cdi_pct_of_core_deposits,
        "Cost saves (% of target noninterest expense)": (
            config.cost_saves.cost_save_pct_of_target_noninterest_expense
        ),
        "Restructuring charge ($mm)": config.cost_saves.restructuring_charge_mm,
        "Deal horizon (quarters)": config.deal_horizon_quarters,
        "Exit multiple": config.irr.exit_multiple,
        "Exit multiple basis": config.irr.exit_multiple_basis.value,
        "Pro forma CET1 ratio": capital.pro_forma_cet1_ratio,
        "Pro forma Tier 1 leverage ratio": capital.pro_forma_tier1_leverage_ratio,
        "Pro forma total capital ratio": capital.pro_forma_total_capital_ratio,
    }
    return pd.DataFrame({"Value": rows})


def _eps_accretion_frame(eps: EpsAccretionResult) -> pd.DataFrame:
    rows = {}
    for year, a in eps.annual().items():
        rows[year] = {
            "Standalone EPS": a.standalone_eps,
            "(a) GAAP pro forma EPS": a.pro_forma_eps_gaap,
            "(a) GAAP $ accretion": a.dollar_accretion_gaap,
            "(a) GAAP % accretion": (
                a.accretion_dilution_pct_gaap if a.accretion_pct_meaningful else None
            ),
            "(b) Excl. one-time pro forma EPS": a.pro_forma_eps_excl_one_time,
            "(b) Excl. one-time $ accretion": a.dollar_accretion_excl_one_time,
            "(b) Excl. one-time % accretion": (
                a.accretion_dilution_pct_excl_one_time if a.accretion_pct_meaningful else None
            ),
            "(c) Excl. one-time+marks pro forma EPS": a.pro_forma_eps_excl_one_time_and_marks,
            "(c) Excl. one-time+marks $ accretion": a.dollar_accretion_excl_one_time_and_marks,
            "(c) Excl. one-time+marks % accretion": (
                a.accretion_dilution_pct_excl_one_time_and_marks
                if a.accretion_pct_meaningful
                else None
            ),
        }
    return pd.DataFrame(rows).T


def _tbv_and_irr_frame(tbv: TbvEarnbackResult, irr_by_multiple: dict[float, float]) -> pd.DataFrame:
    tbv_rows = pd.DataFrame(
        {
            "Standalone TBV/share": tbv.standalone_tbv_per_share,
            "Pro forma TBV/share": tbv.pro_forma_tbv_per_share,
        },
        index=[f"Period {i}" for i in range(len(tbv.standalone_tbv_per_share))],
    )
    irr_rows = pd.DataFrame(
        {"IRR": irr_by_multiple}, index=[f"{m:.2f}x exit multiple" for m in irr_by_multiple]
    )
    summary = pd.DataFrame(
        {
            "Value": {
                "TBV dilution at close": tbv.tbv_dilution_at_close_pct,
                "Earnback": tbv.earnback_label,
            }
        }
    )
    return pd.concat([summary, tbv_rows, irr_rows], axis=0)


def _stressed_path_frame(stress_results: dict[str, StressTestResult]) -> pd.DataFrame:
    rows = []
    for scenario_name, result in stress_results.items():
        for entity_name, path in (
            ("Acquirer standalone", result.acquirer_standalone),
            ("Pro forma combined", result.pro_forma_combined),
        ):
            rows.append(
                {
                    "Scenario": scenario_name,
                    "Entity": entity_name,
                    "Starting CET1": path.starting_cet1_ratio,
                    "Minimum CET1": path.minimum_cet1_ratio,
                    "Peak-to-trough (pp)": path.peak_to_trough_cet1_change_pp,
                    "Breaches 4.5% minimum": path.breaches_4_5_pct_minimum,
                    "Illustrative SCB": path.illustrative_stress_capital_buffer,
                    "Cumulative PPNR ($mm)": path.cumulative_ppnr_mm,
                    "Cumulative provisions ($mm)": path.cumulative_provision_mm,
                    "Cumulative net income ($mm)": path.cumulative_net_income_mm,
                }
            )
    return pd.DataFrame(rows).set_index(["Scenario", "Entity"])


def _tornado_frame(tornado: TornadoResult) -> pd.DataFrame:
    rows = []
    for row in tornado.rows:
        rows.append(
            {
                "Driver": row.driver_name,
                "Low value": row.low_value,
                "High value": row.high_value,
                "Year-2 accretion (low)": row.low_metrics.year2_accretion_pct_excl_one_time,
                "Year-2 accretion (high)": row.high_metrics.year2_accretion_pct_excl_one_time,
                "TBV dilution (low)": row.low_metrics.tbv_dilution_at_close_pct,
                "TBV dilution (high)": row.high_metrics.tbv_dilution_at_close_pct,
                "Earnback years (low)": row.low_metrics.earnback_years,
                "Earnback years (high)": row.high_metrics.earnback_years,
                "Min stressed CET1 (low)": row.low_metrics.minimum_stressed_cet1_ratio,
                "Min stressed CET1 (high)": row.high_metrics.minimum_stressed_cet1_ratio,
            }
        )
    return pd.DataFrame(rows).set_index("Driver")


def _grid_frame(grid: GridResult, metric_array, metric_name: str) -> pd.DataFrame:
    return pd.DataFrame(
        metric_array,
        index=[f"{v:.0%} cost save" for v in grid.cost_save_values],
        columns=[f"{v:.2f}x price" for v in grid.price_values],
    )


def _monte_carlo_draws_frame(mc: MonteCarloResult) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Credit mark %": mc.credit_mark_pct,
            "Cost save %": mc.cost_save_pct,
            "Rate mark %": mc.rate_mark_pct,
            "NIM beta": mc.nim_beta,
            "Year-2 accretion (excl. one-time)": mc.year2_accretion_pct,
            "TBV dilution at close": mc.tbv_dilution_at_close_pct,
            "Earnback (years)": mc.earnback_years,
            "Minimum stressed CET1": mc.minimum_stressed_cet1_ratio,
        }
    )


def _monte_carlo_percentiles_frame(mc: MonteCarloResult) -> pd.DataFrame:
    percentiles = (5, 25, 50, 75, 95)
    rows = {}
    for metric in (
        "year2_accretion_pct",
        "tbv_dilution_at_close_pct",
        "earnback_years",
        "minimum_stressed_cet1_ratio",
    ):
        rows[metric] = mc.percentiles(metric, percentiles)
    frame = pd.DataFrame(rows).T
    frame.columns = [f"P{p}" for p in percentiles]
    frame["P(TBV earnback beyond horizon)"] = mc.probability_beyond_horizon()
    return frame


def export_deal_to_excel(
    path: str,
    acquirer_label: str,
    target_label: str,
    config: DealConfig,
    pro_forma_capital: ProFormaCapitalRatios,
    eps_accretion: EpsAccretionResult,
    tbv_earnback: TbvEarnbackResult,
    irr_by_multiple: dict[float, float],
    stress_results: dict[str, StressTestResult],
    tornado: TornadoResult,
    grid: GridResult,
    monte_carlo: MonteCarloResult,
) -> None:
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        _summary_frame(acquirer_label, target_label, config, pro_forma_capital).to_excel(
            writer, sheet_name="Summary"
        )
        _eps_accretion_frame(eps_accretion).to_excel(writer, sheet_name="EPS Accretion")
        _tbv_and_irr_frame(tbv_earnback, irr_by_multiple).to_excel(writer, sheet_name="TBV & IRR")
        _stressed_path_frame(stress_results).to_excel(writer, sheet_name="Stress Test")
        _tornado_frame(tornado).to_excel(writer, sheet_name="Tornado")
        _grid_frame(grid, grid.year2_accretion_pct, "accretion").to_excel(
            writer, sheet_name="Grid - Year 2 Accretion"
        )
        _grid_frame(grid, grid.earnback_years, "earnback").to_excel(
            writer, sheet_name="Grid - Earnback (yrs)"
        )
        _monte_carlo_percentiles_frame(monte_carlo).to_excel(
            writer, sheet_name="Monte Carlo - Percentiles"
        )
        _monte_carlo_draws_frame(monte_carlo).to_excel(
            writer, sheet_name="Monte Carlo - Draws", index=False
        )
