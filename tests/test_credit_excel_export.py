"""Offline test for Stage 7's Excel export -- synthetic data only, no
network, writes to a temp file."""

from __future__ import annotations

import numpy as np
import pandas as pd

from corefin.credit import excel_export
from corefin.credit.interface import CreditLossProjection
from corefin.timeline import Timeline


def _synthetic_result(bank_identifier=None) -> CreditLossProjection:
    timeline = Timeline.quarterly(n_periods=5, n_historical=1, start_year=2025, start_quarter=4)
    categories = ("commercial_and_industrial", "credit_card")
    shape = (len(categories), timeline.n_periods)
    rng = np.random.default_rng(0)
    nco = np.full(shape, np.nan)
    nco[:, 1:] = rng.uniform(0.01, 0.05, size=(len(categories), timeline.n_periods - 1))
    n_categories = len(categories)
    return CreditLossProjection(
        timeline=timeline,
        categories=categories,
        scenario_name="severely_adverse",
        bank_identifier=bank_identifier,
        balance_mm=np.full(shape, 1_000.0),
        net_charge_off_mm=nco * 10,
        provision_expense_mm=nco * 11,
        allowance_mm=np.full(shape, 50.0),
        npl_mm=np.full(shape, 20.0),
        nco_rate=nco,
        npl_ratio=np.full(shape, 0.02),
        monte_carlo_mean=np.full(n_categories, np.nan)
        if bank_identifier
        else rng.uniform(0.01, 0.05, size=n_categories),
        monte_carlo_percentiles={
            p: (
                np.full(n_categories, np.nan)
                if bank_identifier
                else rng.uniform(0.01, 0.05, size=n_categories)
            )
            for p in (5, 25, 50, 75, 95)
        },
    )


def test_export_credit_loss_projection_writes_every_expected_sheet(tmp_path):
    result = _synthetic_result()
    path = tmp_path / "projection.xlsx"
    excel_export.export_credit_loss_projection_to_excel(str(path), result)

    assert path.exists()
    sheets = pd.read_excel(path, sheet_name=None)
    expected = {
        "Summary",
        "Balance ($mm)",
        "Net Charge-offs ($mm)",
        "Provision Expense ($mm)",
        "Allowance ($mm)",
        "NPLs ($mm)",
        "NCO Rate",
        "NPL Ratio",
        "Monte Carlo (9-13Q loss rate)",
    }
    assert expected.issubset(set(sheets))

    balance_sheet = sheets["Balance ($mm)"]
    assert "Total" in balance_sheet.columns
    assert "commercial_and_industrial" in balance_sheet.columns


def test_export_credit_loss_projection_single_bank_mode_skips_monte_carlo_sheet(tmp_path):
    result = _synthetic_result(bank_identifier="1234567")
    path = tmp_path / "projection_bank.xlsx"
    excel_export.export_credit_loss_projection_to_excel(str(path), result)

    sheets = pd.read_excel(path, sheet_name=None)
    assert "Monte Carlo (9-13Q loss rate)" not in sheets
    summary = sheets["Summary"]
    assert "1234567" in summary.to_string()
