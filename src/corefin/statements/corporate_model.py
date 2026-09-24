"""Orchestrates IS/BS/CFS for a corporate entity.

`run_corporate_model_no_debt` wires the statements together with zero debt
and zero interest expense -- used to validate the statement plumbing in
isolation before the debt module (stage 3) injects interest expense and
debt draws/repayments into the same functions.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from corefin.assumptions.loader import expand_series
from corefin.assumptions.schema import (
    CompanyAssumptions,
    OpeningBalanceSheet,
    TrancheConfig,
    WaterfallConfig,
)
from corefin.debt.circularity import DebtScheduleResult, run_debt_schedule
from corefin.scenarios.drivers import DriverSet
from corefin.statements.balance_sheet import (
    BalanceSheet,
    compute_nwc,
    roll_forward_cash,
    roll_forward_deferred_financing_costs,
    roll_forward_equity,
    roll_forward_ppe,
)
from corefin.statements.cash_flow import (
    CashFlowStatement,
    compute_cash_flow_statement,
    compute_delta,
)
from corefin.statements.income_statement import (
    IncomeStatement,
    compute_income_statement,
    compute_nol_and_tax,
    compute_revenue,
)
from corefin.timeline import Timeline


@dataclass(frozen=True)
class CorporateModelResult:
    income_statement: IncomeStatement
    balance_sheet: BalanceSheet
    cash_flow_statement: CashFlowStatement


@dataclass(frozen=True)
class CorporateModelWithDebtResult:
    income_statement: IncomeStatement
    balance_sheet: BalanceSheet
    cash_flow_statement: CashFlowStatement
    debt_schedule: DebtScheduleResult


def run_corporate_model_no_debt(
    company: CompanyAssumptions,
    opening_balance_sheet: OpeningBalanceSheet,
    timeline: Timeline,
    drivers: DriverSet,
) -> CorporateModelResult:
    n = timeline.n_periods
    n_scenarios = drivers.n_scenarios
    da_pct_revenue = expand_series(company.da_pct_revenue, n, "da_pct_revenue").reshape(1, n)

    income_statement = compute_income_statement(
        drivers=drivers,
        revenue_base_mm=company.revenue_base_mm,
        da_pct_revenue=da_pct_revenue,
        tax_rate=company.tax_rate,
        nol_beginning_mm=company.nol_beginning_balance_mm,
    )

    capex = income_statement.revenue * drivers.capex_pct_revenue
    nwc = compute_nwc(income_statement.revenue, drivers.nwc_pct_revenue)
    ppe = roll_forward_ppe(opening_balance_sheet.ppe_mm, capex, income_statement.da)
    dividends = income_statement.net_income * company.dividend_pct_of_ni

    cash_flow_statement = compute_cash_flow_statement(
        net_income=income_statement.net_income,
        da=income_statement.da,
        nwc=nwc,
        nwc_beginning_mm=opening_balance_sheet.nwc_mm,
        capex=capex,
        dividends=dividends,
    )

    cash = roll_forward_cash(opening_balance_sheet.cash_mm, cash_flow_statement.change_in_cash)
    equity = roll_forward_equity(
        opening_balance_sheet.equity_mm, income_statement.net_income, dividends
    )
    goodwill = np.full((n_scenarios, n), opening_balance_sheet.goodwill_mm)
    deferred_financing_costs = np.full(
        (n_scenarios, n), opening_balance_sheet.deferred_financing_costs_mm
    )
    other_liabilities = np.full((n_scenarios, n), opening_balance_sheet.other_liabilities_mm)
    total_debt = np.zeros((n_scenarios, n))

    balance_sheet = BalanceSheet(
        cash=cash,
        nwc=nwc,
        ppe=ppe,
        goodwill=goodwill,
        deferred_financing_costs=deferred_financing_costs,
        total_debt=total_debt,
        other_liabilities=other_liabilities,
        equity=equity,
    )
    return CorporateModelResult(income_statement, balance_sheet, cash_flow_statement)


def run_corporate_model_with_debt(
    company: CompanyAssumptions,
    opening_balance_sheet: OpeningBalanceSheet,
    tranches: list[TrancheConfig],
    waterfall_config: WaterfallConfig,
    timeline: Timeline,
    drivers: DriverSet,
) -> CorporateModelWithDebtResult:
    n = timeline.n_periods
    n_scenarios = drivers.n_scenarios
    da_pct_revenue = expand_series(company.da_pct_revenue, n, "da_pct_revenue").reshape(1, n)

    revenue = compute_revenue(company.revenue_base_mm, drivers.revenue_growth)
    ebitda = revenue * drivers.ebitda_margin
    da = revenue * da_pct_revenue
    ebit = ebitda - da
    capex = revenue * drivers.capex_pct_revenue
    nwc = compute_nwc(revenue, drivers.nwc_pct_revenue)
    delta_nwc = compute_delta(nwc, opening_balance_sheet.nwc_mm)

    debt_schedule = run_debt_schedule(
        tranches=tranches,
        waterfall_config=waterfall_config,
        timeline=timeline,
        drivers=drivers,
        ebit=ebit,
        da=da,
        capex=capex,
        delta_nwc=delta_nwc,
        tax_rate=company.tax_rate,
        nol_beginning_mm=company.nol_beginning_balance_mm,
        dividend_pct_of_ni=company.dividend_pct_of_ni,
        cash_beginning_mm=opening_balance_sheet.cash_mm,
    )

    cash_interest_expense = sum(debt_schedule.cash_interest.values())
    pik_interest_expense = sum(debt_schedule.pik_interest.values())
    fee_amortization = sum(debt_schedule.fee_amortization.values())
    total_interest_for_ebt = cash_interest_expense + pik_interest_expense + fee_amortization

    ebt = ebit - total_interest_for_ebt
    tax, nol_balance = compute_nol_and_tax(ebt, company.tax_rate, company.nol_beginning_balance_mm)
    net_income = ebt - tax
    dividends = net_income * company.dividend_pct_of_ni

    income_statement = IncomeStatement(
        revenue=revenue,
        ebitda=ebitda,
        da=da,
        ebit=ebit,
        cash_interest_expense=cash_interest_expense,
        pik_interest_expense=pik_interest_expense,
        ebt=ebt,
        tax=tax,
        net_income=net_income,
        nol_balance=nol_balance,
    )

    debt_draws = debt_schedule.revolver_draw
    debt_repayments = -(
        sum(debt_schedule.mandatory_amort.values())
        + sum(debt_schedule.sweep_amort.values())
        + debt_schedule.revolver_paydown
    )

    cash_flow_statement = compute_cash_flow_statement(
        net_income=net_income,
        da=da,
        nwc=nwc,
        nwc_beginning_mm=opening_balance_sheet.nwc_mm,
        capex=capex,
        pik_interest_expense=pik_interest_expense,
        fee_amortization=fee_amortization,
        debt_draws=debt_draws,
        debt_repayments=debt_repayments,
        dividends=dividends,
    )

    cash = roll_forward_cash(opening_balance_sheet.cash_mm, cash_flow_statement.change_in_cash)
    equity = roll_forward_equity(opening_balance_sheet.equity_mm, net_income, dividends)
    goodwill = np.full((n_scenarios, n), opening_balance_sheet.goodwill_mm)
    deferred_financing_costs = roll_forward_deferred_financing_costs(
        opening_balance_sheet.deferred_financing_costs_mm, fee_amortization
    )
    other_liabilities = np.full((n_scenarios, n), opening_balance_sheet.other_liabilities_mm)
    total_debt = sum(debt_schedule.ending_balance.values())

    balance_sheet = BalanceSheet(
        cash=cash,
        nwc=nwc,
        ppe=roll_forward_ppe(opening_balance_sheet.ppe_mm, capex, da),
        goodwill=goodwill,
        deferred_financing_costs=deferred_financing_costs,
        total_debt=total_debt,
        other_liabilities=other_liabilities,
        equity=equity,
    )
    return CorporateModelWithDebtResult(
        income_statement, balance_sheet, cash_flow_statement, debt_schedule
    )
