"""Integrity checks for the corporate three-statement model."""

from __future__ import annotations

from corefin.checks.framework import CheckResult, check_close_to_zero
from corefin.statements.balance_sheet import BalanceSheet, balance_residual, roll_forward_cash
from corefin.statements.cash_flow import CashFlowStatement

DEFAULT_TOLERANCE = 1e-6


def check_balance_sheet_balances(
    balance_sheet: BalanceSheet, tolerance: float = DEFAULT_TOLERANCE
) -> CheckResult:
    residual = balance_residual(balance_sheet)
    return check_close_to_zero("balance_sheet_balances", residual, tolerance)


def check_cash_ties_to_cfs(
    balance_sheet: BalanceSheet,
    cash_flow_statement: CashFlowStatement,
    cash_beginning_mm: float,
    tolerance: float = DEFAULT_TOLERANCE,
) -> CheckResult:
    recomputed_cash = roll_forward_cash(cash_beginning_mm, cash_flow_statement.change_in_cash)
    residual = balance_sheet.cash - recomputed_cash
    return check_close_to_zero("cash_ties_to_cfs", residual, tolerance)
