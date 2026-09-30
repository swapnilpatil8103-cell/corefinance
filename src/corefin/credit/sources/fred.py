"""FRED (Federal Reserve Economic Data) client and the mapping from the
Fed's own supervisory stress test scenario variable names to FRED series.

Requires FRED_API_KEY as an environment variable -- get a free key at
https://fredaccount.stlouisfed.org/apikey. Never read from a file, never
logged, never written anywhere by this module; `fetch_series` reads it via
`os.environ` at call time and raises `FredApiKeyMissingError` with a clear
message if it's unset.

Every FRED series ID below was verified live against the FRED API (not
guessed from memory) before being committed here -- two initial picks
turned out to be wrong: a "Wilshire 5000" series ID that doesn't exist on
FRED at all, and a BBB corporate yield series that only starts in 2023
(useless for the 2007-2010/2020 backtest windows this project needs).
Where no precise FRED equivalent exists for the Fed's own variable,
`is_exact_match=False` and `note` says so explicitly -- see FED_SCENARIO_
VARIABLE_TO_FRED below.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
import requests

FRED_OBSERVATIONS_URL = "https://api.stlouisfed.org/fred/series/observations"


class FredApiKeyMissingError(RuntimeError):
    pass


def _api_key() -> str:
    key = os.environ.get("FRED_API_KEY")
    if not key:
        raise FredApiKeyMissingError(
            "FRED_API_KEY environment variable is not set. Get a free key at "
            "https://fredaccount.stlouisfed.org/apikey and export it as "
            "FRED_API_KEY -- never write it to a config file or commit it."
        )
    return key


def fetch_series(
    series_id: str,
    start_date: str | None = None,
    end_date: str | None = None,
    api_key: str | None = None,
) -> pd.DataFrame:
    """Observations for one FRED series as a DataFrame with columns
    ["date", "value"] (value is NaN for FRED's "." missing-data marker).
    Native frequency depends on the series (see FED_SCENARIO_VARIABLE_TO_FRED's
    `frequency` field) -- callers needing quarterly data aggregate
    themselves (credit/panel.py), since how to aggregate (average vs.
    quarter-end) is a modeling choice, not something this thin client
    should decide. `api_key` overrides FRED_API_KEY, for tests only;
    production callers should leave it None."""
    key = api_key or _api_key()
    params: dict[str, str] = {"series_id": series_id, "api_key": key, "file_type": "json"}
    if start_date is not None:
        params["observation_start"] = start_date
    if end_date is not None:
        params["observation_end"] = end_date
    response = requests.get(FRED_OBSERVATIONS_URL, params=params, timeout=30)
    response.raise_for_status()
    payload = response.json()
    observations = payload.get("observations", [])
    dates = [o["date"] for o in observations]
    values = [np.nan if o["value"] == "." else float(o["value"]) for o in observations]
    return pd.DataFrame({"date": pd.to_datetime(dates), "value": values})


@dataclass(frozen=True)
class FredSeriesMapping:
    fred_series_id: str
    frequency: str  # FRED's own code: D/W/M/Q
    observation_start: str  # as reported by FRED, for reference
    transform: str  # "level" or "qoq_annualized_pct_change"
    is_exact_match: bool
    note: str = ""


FED_SCENARIO_VARIABLE_TO_FRED: dict[str, FredSeriesMapping] = {
    "Real GDP growth": FredSeriesMapping(
        "GDPC1", "Q", "1947-01-01", "qoq_annualized_pct_change", True
    ),
    "Nominal GDP growth": FredSeriesMapping(
        "GDP", "Q", "1947-01-01", "qoq_annualized_pct_change", True
    ),
    "Real disposable income growth": FredSeriesMapping(
        "DPIC96", "Q", "1947-01-01", "qoq_annualized_pct_change", True
    ),
    "Nominal disposable income growth": FredSeriesMapping(
        "DSPI",
        "M",
        "1959-01-01",
        "qoq_annualized_pct_change",
        True,
        "Monthly -- aggregate to quarterly (e.g. quarter-end) before the growth transform.",
    ),
    "Unemployment rate": FredSeriesMapping(
        "UNRATE",
        "M",
        "1948-01-01",
        "level",
        True,
        "Monthly -- use the quarterly average to match the Fed's quarterly scenario value.",
    ),
    "CPI inflation rate": FredSeriesMapping(
        "CPIAUCSL",
        "M",
        "1947-01-01",
        "qoq_annualized_pct_change",
        True,
        "Monthly index level -- aggregate to quarterly before the growth transform.",
    ),
    "3-month Treasury rate": FredSeriesMapping(
        "TB3MS", "M", "1934-01-01", "level", True, "Monthly -- use the quarterly average."
    ),
    "5-year Treasury yield": FredSeriesMapping(
        "GS5", "M", "1953-04-01", "level", True, "Monthly -- use the quarterly average."
    ),
    "10-year Treasury yield": FredSeriesMapping(
        "GS10", "M", "1953-04-01", "level", True, "Monthly -- use the quarterly average."
    ),
    "BBB corporate yield": FredSeriesMapping(
        "BAA",
        "M",
        "1919-01-01",
        "level",
        False,
        "Moody's Seasoned Baa Corporate Bond Yield -- Moody's rating scale, not S&P/Fitch "
        "BBB, but the only BBB-adjacent series with history back to the 2007-2010 window. "
        "ICE BofA's own BBB Effective Yield (BAMLC0A4CBBBEY) is a closer rating-scale match "
        "but only starts 2023-09-26 -- unusable for the required backtests, kept as a "
        "possible refinement for recent-period precision only.",
    ),
    "Mortgage rate": FredSeriesMapping(
        "MORTGAGE30US", "W", "1971-04-02", "level", True, "Weekly -- use the quarterly average."
    ),
    "Prime rate": FredSeriesMapping(
        "MPRIME", "M", "1949-01-01", "level", True, "Monthly -- use the quarterly average."
    ),
    "Dow Jones Total Stock Market Index": FredSeriesMapping(
        "NASDAQCOM",
        "D",
        "1971-02-05",
        "level",
        False,
        "No FRED series matches the Fed's own Dow Jones Total Stock Market Index (it's a "
        "proprietary S&P Dow Jones Indices product, not freely published); the Wilshire "
        "5000 series once carried on FRED no longer exists under any ID tried. NASDAQ "
        "Composite is used as an imperfect, tech-heavy broad-equity proxy -- a known gap, "
        "flagged for a future non-FRED source (e.g. a market data vendor or manual file).",
    ),
    "House Price Index": FredSeriesMapping(
        "USSTHPI",
        "Q",
        "1975-01-01",
        "level",
        False,
        "FHFA All-Transactions House Price Index; the Fed's own scenario uses a proprietary "
        "CoreLogic-based national HPI, not freely available -- directionally similar, not a "
        "level match.",
    ),
    "Commercial Real Estate Price Index": FredSeriesMapping(
        "COMREPUSQ159N",
        "Q",
        "2005-01-01",
        "level",
        False,
        "MORE SERIOUS than a methodology difference -- confirmed live via FRED's own series "
        "metadata: COMREPUSQ159N's units are 'Percent Change from Year Ago' (NSA), NOT a "
        "level -- real values run roughly -10 to +16, nothing like the Fed's own CRE Price "
        "Index level (roughly 250-350 in the Fed's own scenario files). transform='level' "
        "here means 'pass through unchanged,' NOT 'already on the Fed's index scale' -- this "
        "column is NOT comparable in level terms to the Fed's own CRE index and should not "
        "be charted/aligned against it as if it were (no freely-published CRE price LEVEL "
        "series was found on FRED as of this check; only started 2005 either way, barely "
        "covering the 2007-2010 window with no pre-crisis lead-in, and its own real history "
        "stops at 2025Q2 with no more recent observations).",
    ),
    "Market Volatility Index": FredSeriesMapping("VIXCLS", "D", "1990-01-02", "level", True),
}


@dataclass(frozen=True)
class IndustrySeriesMapping:
    chargeoff_series_id: str
    delinquency_series_id: str | None
    observation_start: str
    is_exact_match: bool
    note: str = ""


# FRED's own "Charge-Off and Delinquency Rates on Loans and Leases at
# Commercial Banks" release (verified live via the FRED API's series-
# search endpoint, not guessed -- an initial guess at these IDs based on
# naming conventions from a similar, differently-scoped series would have
# been wrong: e.g. "CORCACBS" is real but is "Charge-Off Rate on CONSUMER
# Loans," not credit cards specifically -- the real credit-card series is
# CORCCACBS). All values are ALREADY ANNUALIZED, in PERCENT (e.g. 5.80
# means 5.80%, not 0.058) -- divide by 100 before comparing against this
# project's own decimal-fraction rate columns (annualized_nco_rate,
# winsorized_nco_rate, etc.). NSA (not seasonally adjusted) variants are
# used throughout to match this project's own Call-Report aggregation,
# which is likewise raw/unadjusted -- an SA variant of the same series
# exists for every one of these (just drop the trailing "N" for "S").
#
# These are INDUSTRY-WIDE aggregates across ALL commercial banks (not
# bank-level data) -- used only to extend model family 1's (the
# aggregate AR model's) TRAINING HISTORY back before the Call Report
# panel's own 2001Q1 start, especially to 1991Q1 (where CRE/residential
# mortgage data begins), covering the early-1990s CRE bust the
# 2001-2006 Call Report window misses entirely. Bank-level models (family
# 2/3) stay on the Call Report panel -- these series have no bank-level
# breakdown to join against individual banks' characteristics.
INDUSTRY_CHARGEOFF_DELINQUENCY_SERIES: dict[str, IndustrySeriesMapping] = {
    "commercial_and_industrial": IndustrySeriesMapping(
        "CORBLACBN",
        "DRBLACBN",
        "1985-01-01",
        False,
        "FRED's 'Business Loans' series is the closest available match to this project's "
        "commercial_and_industrial category, but 'business loans' is not necessarily an exact "
        "definitional match to Call Report Schedule RC-C C&I loans -- real crisis-era values "
        "(e.g. 2008Q3-2009Q2: 0.98/1.55/1.71/2.31 percent) are in the right ballpark for known "
        "C&I charge-off experience, but this is a proxy, not a confirmed identical series.",
    ),
    "cre_construction": IndustrySeriesMapping(
        "CORCREXFACBN",
        "DRCRELEXFACBN",
        "1991-01-01",
        False,
        "FRED publishes ONE combined 'Commercial Real Estate Loans (Excluding Farmland)' "
        "series -- it does NOT split by construction/multifamily/nonfarm-nonresidential the "
        "way this project's schema does. The SAME series is used for all three CRE "
        "categories below; each one's long-history aggregate model is really being trained "
        "against overall CRE charge-off experience, not its own sub-category's specific "
        "dynamics -- a real, documented limitation, not a precise match.",
    ),
    "cre_multifamily": IndustrySeriesMapping(
        "CORCREXFACBN",
        "DRCRELEXFACBN",
        "1991-01-01",
        False,
        "Same series/caveat as cre_construction.",
    ),
    "cre_nonfarm_nonresidential": IndustrySeriesMapping(
        "CORCREXFACBN",
        "DRCRELEXFACBN",
        "1991-01-01",
        False,
        "Same series/caveat as cre_construction.",
    ),
    "residential_mortgage": IndustrySeriesMapping(
        "CORSFRMACBN",
        "DRSFRMACBN",
        "1991-01-01",
        True,
        "'Single Family Residential Mortgages, Booked in Domestic Offices' is essentially the "
        "same definition as this project's residential_mortgage category.",
    ),
    "credit_card": IndustrySeriesMapping(
        "CORCCACBN",
        "DRCCLACBN",
        "1985-01-01",
        True,
        "'Charge-Off Rate on Credit Card Loans' is an exact definitional match.",
    ),
    "auto": IndustrySeriesMapping(
        "COROCLACBN",
        "DROCLACBN",
        "1985-01-01",
        False,
        "FRED has no series isolating auto loans specifically for all commercial banks -- "
        "confirmed via a live series search (no match for 'automobile'/'auto loan' charge-off "
        "or delinquency scoped to all commercial banks). 'Other Consumer Loans' (FRED's only "
        "non-credit-card consumer bucket) is used as an imperfect proxy -- auto loans are a "
        "large component of it, but so is everything else non-credit-card. The SAME series is "
        "also used for other_consumer below (FRED doesn't split the two either).",
    ),
    "other_consumer": IndustrySeriesMapping(
        "COROCLACBN", "DROCLACBN", "1985-01-01", False, "Same series/caveat as auto."
    ),
    # home_equity: deliberately absent. No FRED series (charge-off,
    # delinquency, or otherwise) isolates home equity loans/lines for all
    # commercial banks -- confirmed via a live series search. This
    # category's aggregate model has no long-history option and stays on
    # the Call Report panel's own 2001Q1-onward history, same as before.
}
