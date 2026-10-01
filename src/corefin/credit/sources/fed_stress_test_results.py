"""The Fed's own PUBLISHED Dodd-Frank Act stress test RESULTS (not
scenario inputs -- see fed_scenarios.py for those) -- a small, fixed set
of numbers published once a year, so (unlike the scenario/macro-history
pulls elsewhere in sources/) this module does NOT fetch anything at
runtime: the figures below were read LIVE from the Fed's own site on
2026-10-01 (not from training-data memory, per this project's standing
rule on external numeric claims) and are hardcoded here with their exact
source cited, so they can be checked/updated by hand when a newer
vintage is published.

SOURCE (verified live, 2026-10-01):
  https://www.federalreserve.gov/supervisionreg/dfa-stress-tests-2026.htm
    -> "Stress Test Results Data" -> "Stress Test Results, 2013-2026 (CSV)"
  https://www.federalreserve.gov/supervisionreg/files/public_results_DFAST_2026.csv
    -> the row with exercise_name="2026 Stress Test", scenario_name=
       "Supervisory Severely Adverse" (scenario_id=3), id_rssd="" (the
       industry-AGGREGATE row, disclosure_legal_name="32 participating
       banks"), dt_exercise_quarter=12/31/2025 -- the SAME jump-off
       quarter (2025Q4) this project's own Stage 5 projection starts
       from, so both are measuring losses over the identically-dated
       2026Q1-2028Q1 window.
  Narrative results PDF (same figures, same date):
  https://www.federalreserve.gov/publications/files/2026-dfast-results-20260624.pdf
    (dated 2026-06-24 -- "2026 Federal Reserve Stress Test Results")

These are the Fed's own published CUMULATIVE loan loss rates over the
STANDARD 9-quarter DFAST planning horizon (not annualized, not this
project's own 13-quarter scenario horizon -- see projection.
FED_COMPARISON_QUARTERS), already in PERCENT (9.0 means 9.0%), for the
severely adverse scenario only (the scenario DFAST headlines).
"""

from __future__ import annotations

from dataclasses import dataclass

DFAST_RESULTS_VINTAGE = 2026
DFAST_PARTICIPATING_BANKS = 32
DFAST_RESULTS_PUBLISHED_DATE = "2026-06-24"
DFAST_RESULTS_JUMP_OFF_QUARTER = "2025Q4"


@dataclass(frozen=True)
class FedLossRate:
    """`fed_results_column`: the raw column name in the Fed's own public
    results CSV (public_results_DFAST_2026.csv). `severely_adverse_9q_
    loss_rate_percent`: that column's value for the industry-aggregate
    row, in PERCENT. `note`: how this maps (or doesn't) to this
    project's own LoanCategory schema."""

    fed_results_column: str
    severely_adverse_9q_loss_rate_percent: float
    note: str = ""


# Keyed by this project's own category (or a combined-category key, for
# the two buckets the Fed doesn't split further -- CRE and auto/other
# consumer -- matching exactly the two places this project's OWN
# FRED-derived long-history series also can't split further; see
# fred.INDUSTRY_CHARGEOFF_DELINQUENCY_SERIES).
DFAST_2026_SEVERELY_ADVERSE_LOSS_RATES: dict[str, FedLossRate] = {
    "total_loans": FedLossRate(
        "loss_total_loan_rate",
        6.9,
        "All loan types the Fed's DFAST covers, including loss_other_loan_rate "
        "(no equivalent in this project's schema) -- not a clean total-to-total match.",
    ),
    "residential_mortgage": FedLossRate(
        "loss_dom_first_mtg_rate",
        1.5,
        "Fed's 'domestic first-lien mortgages' -- matches this project's "
        "residential_mortgage.",
    ),
    "home_equity": FedLossRate(
        "loss_dom_jr_lien_heloc_rate",
        3.2,
        "Fed's 'domestic junior liens and HELOCs' -- matches this project's home_equity.",
    ),
    "commercial_and_industrial": FedLossRate(
        "loss_comml_ind_loan_rate",
        9.0,
        "Direct match to this project's commercial_and_industrial.",
    ),
    "cre_combined": FedLossRate(
        "loss_dom_cre_loan_rate",
        8.8,
        "The Fed ALSO reports one combined domestic CRE figure, not split by "
        "construction/multifamily/nonfarm-nonresidential -- the same limitation this "
        "project's own FRED long-history series has (one shared 'CRE excluding "
        "farmland' series for all three categories, see fred.py). Compared here "
        "against this project's own balance-weighted average of all three CRE "
        "categories' 9-quarter loss rates.",
    ),
    "auto_and_other_consumer_combined": FedLossRate(
        "loss_other_consumer_loan_rate",
        7.3,
        "The Fed's standard DFAST 'other consumer' bucket (ex credit card, ex "
        "first-lien mortgage/HELOC) covers auto loans and other consumer "
        "installment/revolving credit TOGETHER -- matches this project's own "
        "auto_and_other_consumer_combined category, not auto/other_consumer "
        "individually. Compared here against this project's own balance-weighted "
        "average of auto and other_consumer's 9-quarter loss rates.",
    ),
    "credit_card": FedLossRate(
        "loss_credit_card_loan_rate",
        17.1,
        "Direct match to this project's credit_card.",
    ),
    "other_loans": FedLossRate(
        "loss_other_loan_rate",
        3.8,
        "No equivalent in this project's schema (international loans, agricultural "
        "loans, lease financing, and other loan types this project doesn't model).",
    ),
}
