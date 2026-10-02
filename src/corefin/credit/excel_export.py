"""Excel export for Stage 7's `CreditLossProjection`
(`corefin credit export-projection`): one sheet per metric (balance, net
charge-offs, provision expense, allowance, NPLs, NCO rate, NPL ratio),
columns = categories plus a "Total" column (the rate sheets have no
"Total" column -- a rate isn't additive across categories), rows =
quarters (`timeline.year_labels`), plus a Monte Carlo percentile sheet
and a one-page Summary. Matches the rest of corefin's own Excel export
convention (`io/excel_export.py`, `optimize/excel_export.py`, `simulate/
excel_export.py`): one `pd.ExcelWriter(path, engine="openpyxl")`, a
`_<name>_frame(result) -> pd.DataFrame` helper per sheet.
"""

from __future__ import annotations

import pandas as pd

from corefin.credit.interface import CreditLossProjection


def _per_category_frame(
    result: CreditLossProjection, array_attr: str, total_attr: str
) -> pd.DataFrame:
    array = getattr(result, array_attr)
    data = {category: array[i] for i, category in enumerate(result.categories)}
    data["Total"] = getattr(result, total_attr)
    return pd.DataFrame(data, index=result.timeline.year_labels)


def _rate_frame(result: CreditLossProjection, array_attr: str) -> pd.DataFrame:
    array = getattr(result, array_attr)
    data = {category: array[i] for i, category in enumerate(result.categories)}
    return pd.DataFrame(data, index=result.timeline.year_labels)


def _monte_carlo_frame(result: CreditLossProjection) -> pd.DataFrame:
    rows = {}
    for i, category in enumerate(result.categories):
        row = {"Mean": result.monte_carlo_mean[i]}
        for percentile, array in sorted(result.monte_carlo_percentiles.items()):
            row[f"P{percentile}"] = array[i]
        rows[category] = row
    return pd.DataFrame(rows).T


def _summary_frame(result: CreditLossProjection) -> pd.DataFrame:
    last = -1
    return pd.DataFrame(
        {
            "Value": {
                "Scenario": result.scenario_name,
                "Bank": result.bank_identifier if result.bank_identifier else "Industry-wide",
                "Categories": ", ".join(result.categories),
                "Jump-off Quarter": result.timeline.year_labels[0],
                "Last Projected Quarter": result.timeline.year_labels[last],
                "Jump-off Total Balance ($mm)": result.balance_total_mm[0],
                "Ending Total Allowance ($mm)": result.allowance_total_mm[last],
                "Ending Total NPL ($mm)": result.npl_total_mm[last],
                "Cumulative Total Net Charge-offs ($mm)": pd.Series(
                    result.net_charge_off_total_mm
                ).sum(),
                "Cumulative Total Provision Expense ($mm)": pd.Series(
                    result.provision_expense_total_mm
                ).sum(),
            }
        }
    )


def export_credit_loss_projection_to_excel(path: str, result: CreditLossProjection) -> None:
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        _summary_frame(result).to_excel(writer, sheet_name="Summary")
        _per_category_frame(result, "balance_mm", "balance_total_mm").to_excel(
            writer, sheet_name="Balance ($mm)"
        )
        _per_category_frame(result, "net_charge_off_mm", "net_charge_off_total_mm").to_excel(
            writer, sheet_name="Net Charge-offs ($mm)"
        )
        _per_category_frame(
            result, "provision_expense_mm", "provision_expense_total_mm"
        ).to_excel(writer, sheet_name="Provision Expense ($mm)")
        _per_category_frame(result, "allowance_mm", "allowance_total_mm").to_excel(
            writer, sheet_name="Allowance ($mm)"
        )
        _per_category_frame(result, "npl_mm", "npl_total_mm").to_excel(
            writer, sheet_name="NPLs ($mm)"
        )
        _rate_frame(result, "nco_rate").to_excel(writer, sheet_name="NCO Rate")
        _rate_frame(result, "npl_ratio").to_excel(writer, sheet_name="NPL Ratio")
        if result.bank_identifier is None:
            _monte_carlo_frame(result).to_excel(writer, sheet_name="Monte Carlo (9-13Q loss rate)")
