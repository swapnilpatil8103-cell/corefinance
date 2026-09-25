"""Excel export for a single scenario's statements, debt schedule and metrics.

pandas is used here purely for presentation (building tidy per-scenario
DataFrames to hand to openpyxl); no computation happens on these frames.
"""

from __future__ import annotations

import pandas as pd

from corefin.debt.circularity import DebtScheduleResult
from corefin.metrics.credit_metrics import CreditMetrics
from corefin.statements.balance_sheet import BalanceSheet, balance_residual
from corefin.statements.cash_flow import CashFlowStatement
from corefin.statements.income_statement import IncomeStatement
from corefin.timeline import Timeline


def _income_statement_frame(
    income_statement: IncomeStatement, timeline: Timeline, scenario: int
) -> pd.DataFrame:
    s = scenario
    return pd.DataFrame(
        {
            "Revenue": income_statement.revenue[s],
            "EBITDA": income_statement.ebitda[s],
            "D&A": income_statement.da[s],
            "EBIT": income_statement.ebit[s],
            "Cash Interest Expense": income_statement.cash_interest_expense[s],
            "PIK Interest Expense": income_statement.pik_interest_expense[s],
            "EBT": income_statement.ebt[s],
            "Tax": income_statement.tax[s],
            "Net Income": income_statement.net_income[s],
            "NOL Balance": income_statement.nol_balance[s],
        },
        index=timeline.year_labels,
    )


def _balance_sheet_frame(
    balance_sheet: BalanceSheet, timeline: Timeline, scenario: int
) -> pd.DataFrame:
    s = scenario
    return pd.DataFrame(
        {
            "Cash": balance_sheet.cash[s],
            "NWC": balance_sheet.nwc[s],
            "PP&E": balance_sheet.ppe[s],
            "Goodwill": balance_sheet.goodwill[s],
            "Deferred Financing Costs": balance_sheet.deferred_financing_costs[s],
            "Total Assets": balance_sheet.total_assets[s],
            "Total Debt": balance_sheet.total_debt[s],
            "Other Liabilities": balance_sheet.other_liabilities[s],
            "Equity": balance_sheet.equity[s],
            "Total Liabilities + Equity": balance_sheet.total_liabilities_and_equity[s],
            "Balance Check": balance_residual(balance_sheet)[s],
        },
        index=timeline.year_labels,
    )


def _cash_flow_frame(
    cash_flow_statement: CashFlowStatement, timeline: Timeline, scenario: int
) -> pd.DataFrame:
    s = scenario
    return pd.DataFrame(
        {
            "Net Income": cash_flow_statement.net_income[s],
            "D&A": cash_flow_statement.da[s],
            "PIK Interest (non-cash)": cash_flow_statement.pik_interest_expense[s],
            "Fee Amortization (non-cash)": cash_flow_statement.fee_amortization[s],
            "Change in NWC": cash_flow_statement.delta_nwc[s],
            "CFO": cash_flow_statement.cfo[s],
            "Capex": cash_flow_statement.capex[s],
            "CFI": cash_flow_statement.cfi[s],
            "Debt Draws": cash_flow_statement.debt_draws[s],
            "Debt Repayments": cash_flow_statement.debt_repayments[s],
            "Dividends": cash_flow_statement.dividends[s],
            "CFF": cash_flow_statement.cff[s],
            "Change in Cash": cash_flow_statement.change_in_cash[s],
        },
        index=timeline.year_labels,
    )


def _debt_schedule_frame(
    debt_schedule: DebtScheduleResult, timeline: Timeline, scenario: int
) -> pd.DataFrame:
    s = scenario
    columns: dict[str, object] = {}
    for name in debt_schedule.beginning_balance:
        columns[f"{name} Beginning"] = debt_schedule.beginning_balance[name][s]
        columns[f"{name} Ending"] = debt_schedule.ending_balance[name][s]
        columns[f"{name} Cash Interest"] = debt_schedule.cash_interest[name][s]
        columns[f"{name} PIK Interest"] = debt_schedule.pik_interest[name][s]
        if name in debt_schedule.mandatory_amort:
            columns[f"{name} Mandatory Amort"] = debt_schedule.mandatory_amort[name][s]
            columns[f"{name} Sweep Amort"] = debt_schedule.sweep_amort[name][s]
    columns["Revolver Draw"] = debt_schedule.revolver_draw[s]
    columns["Revolver Paydown"] = debt_schedule.revolver_paydown[s]
    columns["Ending Cash"] = debt_schedule.ending_cash[s]
    columns["Liquidity Shortfall"] = debt_schedule.shortfall_flag[s]
    columns["Circularity Iterations"] = debt_schedule.iterations_used
    return pd.DataFrame(columns, index=timeline.year_labels)


def _credit_metrics_frame(
    credit_metrics: CreditMetrics, timeline: Timeline, scenario: int
) -> pd.DataFrame:
    s = scenario
    return pd.DataFrame(
        {
            "Total Net Leverage": credit_metrics.total_net_leverage[s],
            "Senior Net Leverage": credit_metrics.senior_net_leverage[s],
            "Interest Coverage": credit_metrics.interest_coverage[s],
            "FCCR": credit_metrics.fccr[s],
            "Cumulative Debt Paydown": credit_metrics.cumulative_debt_paydown[s],
        },
        index=timeline.year_labels,
    )


def write_scenario_sheets(
    writer: pd.ExcelWriter,
    timeline: Timeline,
    income_statement: IncomeStatement,
    balance_sheet: BalanceSheet,
    cash_flow_statement: CashFlowStatement,
    debt_schedule: DebtScheduleResult,
    credit_metrics: CreditMetrics,
    scenario: int = 0,
) -> None:
    """Writes the five per-statement sheets to an already-open ExcelWriter --
    shared by export_scenario_to_excel below and optimize/excel_export.py,
    so the optimizer's recommended-structure sheets are produced by the exact
    same code as a standalone `corefin run` export."""
    sheets = {
        "Income Statement": _income_statement_frame(income_statement, timeline, scenario),
        "Balance Sheet": _balance_sheet_frame(balance_sheet, timeline, scenario),
        "Cash Flow Statement": _cash_flow_frame(cash_flow_statement, timeline, scenario),
        "Debt Schedule": _debt_schedule_frame(debt_schedule, timeline, scenario),
        "Credit Metrics": _credit_metrics_frame(credit_metrics, timeline, scenario),
    }
    for sheet_name, frame in sheets.items():
        frame.T.to_excel(writer, sheet_name=sheet_name, index_label="Line Item")


def export_scenario_to_excel(
    path: str,
    timeline: Timeline,
    income_statement: IncomeStatement,
    balance_sheet: BalanceSheet,
    cash_flow_statement: CashFlowStatement,
    debt_schedule: DebtScheduleResult,
    credit_metrics: CreditMetrics,
    scenario: int = 0,
) -> None:
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        write_scenario_sheets(
            writer,
            timeline,
            income_statement,
            balance_sheet,
            cash_flow_statement,
            debt_schedule,
            credit_metrics,
            scenario=scenario,
        )
