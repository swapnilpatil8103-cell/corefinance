"""FDIC BankFind Suite API client: institution identifiers/profile data and
summary financials. No API key required.

The API's live domain is `api.fdic.gov/banks/...` -- the older
`banks.data.fdic.gov` now just 301-redirects there. This client targets
`api.fdic.gov` directly (verified live, not just via the redirect), so a
future removal of that redirect doesn't silently break it.
"""

from __future__ import annotations

import pandas as pd
import requests

FDIC_BASE_URL = "https://api.fdic.gov/banks"

DEFAULT_FINANCIALS_FIELDS = ("CERT", "REPDTE", "ASSET", "DEP", "NETINC")


def fetch_institutions(
    filters: str, fields: list[str] | None = None, limit: int = 100
) -> pd.DataFrame:
    """Institution directory/profile records (name, charter type, location,
    CERT, RSSDHCR, ...) matching `filters` -- FDIC's query-string filter
    syntax, e.g. 'STNAME:"California"' or 'CERT:3510'."""
    params: dict[str, str] = {"filters": filters, "limit": str(limit)}
    if fields:
        params["fields"] = ",".join(fields)
    response = requests.get(f"{FDIC_BASE_URL}/institutions", params=params, timeout=30)
    response.raise_for_status()
    rows = [row["data"] for row in response.json().get("data", [])]
    return pd.DataFrame(rows)


def fetch_financials(cert: int, fields: list[str] | None = None, limit: int = 400) -> pd.DataFrame:
    """Quarterly summary financials (total assets, deposits, net income,
    ...) for one institution by FDIC certificate number, most recent
    first. This is summary-level data only -- loan-category detail (C&I,
    CRE, ...) comes from FFIEC Call Report schedules, not this endpoint."""
    params: dict[str, str] = {
        "filters": f"CERT:{cert}",
        "fields": ",".join(fields or DEFAULT_FINANCIALS_FIELDS),
        "sort_by": "REPDTE",
        "sort_order": "DESC",
        "limit": str(limit),
    }
    response = requests.get(f"{FDIC_BASE_URL}/financials", params=params, timeout=30)
    response.raise_for_status()
    rows = [row["data"] for row in response.json().get("data", [])]
    df = pd.DataFrame(rows)
    if not df.empty and "REPDTE" in df.columns:
        df["REPDTE"] = pd.to_datetime(df["REPDTE"], format="%Y%m%d")
    return df
