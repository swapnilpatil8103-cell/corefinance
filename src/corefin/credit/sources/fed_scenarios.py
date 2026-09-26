"""Federal Reserve supervisory stress test scenario data (baseline and
severely adverse), fetched directly from federalreserve.gov. No key
needed. Both URLs below were verified live (200 OK, real CSV content
with exactly the domestic variables the scenario methodology describes)
before being committed here, via the Fed's own DFAST disclosure page.
"""

from __future__ import annotations

from io import StringIO

import pandas as pd
import requests

FED_SCENARIO_BASE_URL = "https://www.federalreserve.gov/supervisionreg/files"

# The Fed publishes a new scenario vintage each cycle. 2025 is the most
# recently FINALIZED vintage as of this writing (the 2026 scenarios were
# still in "proposed" form per the November 2025 Federal Register notice).
# Bump this once a newer vintage is finalized and its file names verified.
CURRENT_SCENARIO_VINTAGE = 2025

_SCENARIO_TABLES = {
    "baseline": "2A_Supervisory_Baseline_Domestic",
    "severely_adverse": "3A_Supervisory_Severely_Adverse_Domestic",
}


def fetch_scenario(scenario: str, vintage: int = CURRENT_SCENARIO_VINTAGE) -> pd.DataFrame:
    """scenario: "baseline" or "severely_adverse". Returns a DataFrame with
    columns ["Scenario Name", "Date", <domestic variable columns...>] --
    one row per quarter, matching the Fed's own published CSV exactly
    (column names are the Fed's, not renamed here, so the mapping in
    fred.FED_SCENARIO_VARIABLE_TO_FRED lines up by name)."""
    table = _SCENARIO_TABLES.get(scenario)
    if table is None:
        raise ValueError(f"scenario must be one of {sorted(_SCENARIO_TABLES)}, got {scenario!r}")
    url = f"{FED_SCENARIO_BASE_URL}/{vintage}-Table_{table}.csv"
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    return pd.read_csv(StringIO(response.text))
