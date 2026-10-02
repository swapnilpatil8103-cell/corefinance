"""Orchestrates balance_sheet/income_statement/capital for ONE bank under
ONE scenario, plus the Stage 2 integrity checks (balance sheet balances;
capital roll-forward ties) and the Call-Report-vs-FR-Y-9C level
reconciliation `corefin.bank.schema` documents (loans/deposits/equity)."""

from __future__ import annotations

from dataclasses import dataclass

from corefin.bank.balance_sheet import BalanceSheet, project_balance_sheet
from corefin.bank.capital import CapitalResult, compute_capital
from corefin.bank.income_statement import IncomeStatement, compute_income_statement
from corefin.bank.schema import BankConfig, BankOpeningBalance
from corefin.checks.framework import CheckResult, check_close_to_zero
from corefin.credit.interface import CreditLossProjection
from corefin.timeline import Timeline

BALANCE_SHEET_TOLERANCE_MM = 1e-6
CAPITAL_ROLLFORWARD_TOLERANCE_MM = 1e-6


@dataclass(frozen=True)
class BankModelResult:
    opening: BankOpeningBalance
    balance_sheet: BalanceSheet
    income_statement: IncomeStatement
    capital: CapitalResult


def run_bank_model(
    opening: BankOpeningBalance,
    credit_projection: CreditLossProjection,
    config: BankConfig,
    timeline: Timeline,
) -> BankModelResult:
    """`credit_projection` must already be a SINGLE-bank projection
    (`CreditLossProjection.bank_identifier == opening.bank_id`) aligned to
    `timeline`."""
    if credit_projection.bank_identifier != opening.bank_id:
        raise ValueError(
            f"credit_projection is for bank_identifier={credit_projection.bank_identifier!r}, "
            f"expected {opening.bank_id!r} (opening.bank_id)"
        )

    provision_expense_total_mm = credit_projection.provision_expense_mm.sum(axis=0)
    income_statement = compute_income_statement(
        net_interest_income_jumpoff_mm=opening.net_interest_income_jumpoff_mm,
        noninterest_income_jumpoff_mm=opening.noninterest_income_jumpoff_mm,
        noninterest_expense_jumpoff_mm=opening.noninterest_expense_jumpoff_mm,
        provision_expense_total_mm=provision_expense_total_mm,
        config=config,
        timeline=timeline,
    )
    balance_sheet = project_balance_sheet(
        opening=opening,
        credit_projection=credit_projection,
        net_income_mm=income_statement.net_income_mm,
        dividends_mm=income_statement.dividends_mm,
        config=config,
        timeline=timeline,
    )
    capital = compute_capital(balance_sheet, opening, config, timeline)
    return BankModelResult(
        opening=opening,
        balance_sheet=balance_sheet,
        income_statement=income_statement,
        capital=capital,
    )


def check_balance_sheet_balances(
    balance_sheet: BalanceSheet, tolerance: float = BALANCE_SHEET_TOLERANCE_MM
) -> CheckResult:
    residual = balance_sheet.total_assets_mm - balance_sheet.total_liabilities_and_equity_mm
    return check_close_to_zero("balance_sheet_balances", residual.reshape(1, -1), tolerance)


def check_capital_rollforward(
    result: BankModelResult, tolerance: float = CAPITAL_ROLLFORWARD_TOLERANCE_MM
) -> CheckResult:
    """CET1 capital's period-over-period change must equal that period's
    (net income - dividends) -- true by construction in this standalone
    model (goodwill/other intangibles are static and the
    `other_cet1_adjustments_mm` calibration constant doesn't change), but
    checked explicitly so a wiring bug anywhere in the pipeline would be
    caught, and so the SAME check still means something once `corefin.ma`
    injects a goodwill/intangible change at deal close."""
    cet1 = result.capital.cet1_capital_mm
    goodwill = result.balance_sheet.goodwill_mm
    other_intangibles = result.balance_sheet.other_intangibles_mm
    net_income = result.income_statement.net_income_mm
    dividends = result.income_statement.dividends_mm

    actual_delta = cet1[1:] - cet1[:-1]
    expected_delta = (
        (net_income[1:] - dividends[1:])
        - (goodwill[1:] - goodwill[:-1])
        - (other_intangibles[1:] - other_intangibles[:-1])
    )
    residual = actual_delta - expected_delta
    return check_close_to_zero("capital_rollforward_ties", residual.reshape(1, -1), tolerance)


@dataclass(frozen=True)
class LevelReconciliation:
    """Call Report (bank-level) vs FR Y-9C (holding-company-level) gap, per
    the explicit user direction to reconcile loans/deposits/equity across
    the two consolidation levels and report any gap rather than silently
    picking one. `hc_deposits_mm` is None when unavailable: confirmed via
    live MDRM search that FR Y-9C's consolidated Schedule HC has no single
    clean "total deposits" item under a BHCK/BHCA prefix (unlike the Call
    Report's RCON2200) -- the only Y-9C series under item 2200 are the
    parent-company-only FR Y-9LP ("BHCP2200") and the unrelated FR Y-11
    forms, neither of which is the consolidated HC figure needed here."""

    bank_level_loans_mm: float
    bank_level_deposits_mm: float
    bank_level_equity_mm: float
    hc_level_assets_mm: float
    hc_level_deposits_mm: float | None
    hc_level_equity_mm: float
    deposits_gap_mm: float | None
    equity_gap_mm: float
    equity_gap_pct: float


def reconcile_levels(
    bank_level_loans_mm: float,
    bank_level_deposits_mm: float,
    bank_level_equity_mm: float,
    hc_level_assets_mm: float,
    hc_level_equity_mm: float,
    hc_level_deposits_mm: float | None = None,
) -> LevelReconciliation:
    deposits_gap_mm = (
        None if hc_level_deposits_mm is None else hc_level_deposits_mm - bank_level_deposits_mm
    )
    equity_gap_mm = hc_level_equity_mm - bank_level_equity_mm
    equity_gap_pct = equity_gap_mm / bank_level_equity_mm if bank_level_equity_mm else float("nan")
    return LevelReconciliation(
        bank_level_loans_mm=bank_level_loans_mm,
        bank_level_deposits_mm=bank_level_deposits_mm,
        bank_level_equity_mm=bank_level_equity_mm,
        hc_level_assets_mm=hc_level_assets_mm,
        hc_level_deposits_mm=hc_level_deposits_mm,
        hc_level_equity_mm=hc_level_equity_mm,
        deposits_gap_mm=deposits_gap_mm,
        equity_gap_mm=equity_gap_mm,
        equity_gap_pct=equity_gap_pct,
    )
