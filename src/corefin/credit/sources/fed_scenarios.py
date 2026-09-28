"""Federal Reserve supervisory stress test scenario data (baseline and
severely adverse), fetched directly from federalreserve.gov. No key
needed. All URLs below were verified live (200 OK, real CSV content with
exactly the domestic variables the scenario methodology describes) before
being committed here, via the Fed's own DFAST disclosure page.
"""

from __future__ import annotations

from io import StringIO

import pandas as pd
import requests

FED_SCENARIO_BASE_URL = "https://www.federalreserve.gov/supervisionreg/files"

# The Fed publishes a new scenario vintage each cycle. 2026 is the most
# recently FINALIZED vintage as of this writing -- confirmed live via
# federalreserve.gov/supervisionreg/dfa-stress-tests-2026.htm, which links
# both "2026_Final_..." (finalized) and "2026_Proposed_..." CSVs; only the
# Final ones are used here. Bump this once a newer vintage is finalized
# and its file names verified.
CURRENT_SCENARIO_VINTAGE = 2026

# The 2026 cycle introduced a NEW file-naming scheme with no table-number
# segment ("<vintage>_Final_Supervisory_Baseline_Domestic.csv"), replacing
# the older "<vintage>-Table_2A_Supervisory_Baseline_Domestic.csv" scheme
# used through the 2025 vintage -- confirmed live: the old-style URL 404s
# for 2026, and "<vintage>_Final_..." 404s for 2025. Vintages before this
# threshold use the old scheme; this one and later use the new one.
_NEW_NAMING_FIRST_VINTAGE = 2026

_SCENARIO_NAMES = {
    "baseline": "Supervisory_Baseline_Domestic",
    "severely_adverse": "Supervisory_Severely_Adverse_Domestic",
}
_SCENARIO_TABLE_NUMBERS = {  # only used for vintages before _NEW_NAMING_FIRST_VINTAGE
    "baseline": "2A",
    "severely_adverse": "3A",
}


def _scenario_url(scenario: str, vintage: int) -> str:
    if scenario not in _SCENARIO_NAMES:
        raise ValueError(f"scenario must be one of {sorted(_SCENARIO_NAMES)}, got {scenario!r}")
    name = _SCENARIO_NAMES[scenario]
    if vintage >= _NEW_NAMING_FIRST_VINTAGE:
        return f"{FED_SCENARIO_BASE_URL}/{vintage}_Final_{name}.csv"
    table_number = _SCENARIO_TABLE_NUMBERS[scenario]
    return f"{FED_SCENARIO_BASE_URL}/{vintage}-Table_{table_number}_{name}.csv"


def fetch_scenario(scenario: str, vintage: int = CURRENT_SCENARIO_VINTAGE) -> pd.DataFrame:
    """scenario: "baseline" or "severely_adverse". Returns a DataFrame with
    columns ["Scenario Name", "Date", <domestic variable columns...>] --
    one row per quarter, matching the Fed's own published CSV exactly
    (column names are the Fed's, not renamed here, so the mapping in
    fred.FED_SCENARIO_VARIABLE_TO_FRED lines up by name). "Date" is the
    Fed's own "YYYY QN" string format (e.g. "2026 Q1"), not yet parsed to
    a pandas Period -- see macro.normalize_scenario for that."""
    url = _scenario_url(scenario, vintage)
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    return pd.read_csv(StringIO(response.text))
