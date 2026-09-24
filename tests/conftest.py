from __future__ import annotations

import copy
from typing import Any


def minimal_config_dict(n_periods: int = 4) -> dict[str, Any]:
    return copy.deepcopy(
        {
            "timeline": {"n_periods": n_periods, "n_historical": 0, "start_year": 2024},
            "company": {
                "revenue_base_mm": 500.0,
                "revenue_growth": 0.05,
                "ebitda_margin": 0.20,
                "da_pct_revenue": 0.03,
                "capex_pct_revenue": 0.04,
                "nwc_pct_revenue": 0.10,
                "tax_rate": 0.25,
                "nol_beginning_balance_mm": 0.0,
                "dividend_pct_of_ni": 0.0,
            },
            "opening_balance_sheet": {
                "cash_mm": 20.0,
                "nwc_mm": 50.0,
                "ppe_mm": 200.0,
                "goodwill_mm": 0.0,
                "other_liabilities_mm": 30.0,
                "equity_mm": 240.0,
            },
            "transaction": {
                "entry_multiple": 8.0,
                "transaction_fees_pct": 0.02,
                "exit_year_index": n_periods - 1,
            },
            "tranches": [
                {
                    "name": "Revolver",
                    "tranche_type": "revolver",
                    "size_mm": 50.0,
                    "rate_type": "floating",
                    "spread": 0.04,
                    "rate_floor": 0.0,
                    "commitment_fee_pct": 0.005,
                    "cash_sweep_eligible": False,
                },
                {
                    "name": "TLB",
                    "tranche_type": "term_loan_b",
                    "size_mm": 300.0,
                    "rate_type": "floating",
                    "spread": 0.05,
                    "mandatory_amort_pct_of_original": 0.01,
                    "cash_sweep_eligible": True,
                    "sweep_priority": 1,
                },
            ],
            "covenants": [],
            "scenario": {
                "n_scenarios": 1,
                "random_seed": 42,
                "exit_multiple": 8.0,
                "base_rate": 0.045,
            },
            "waterfall": {
                "minimum_cash_mm": 10.0,
                "sweep_pct": 1.0,
                "interest_mode": "average_balance",
            },
        }
    )
