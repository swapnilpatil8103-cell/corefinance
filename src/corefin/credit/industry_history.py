"""Long-history industry-level NCO/NPL series, built from FRED's own
"Charge-Off and Delinquency Rates on Loans and Leases at Commercial
Banks" release (sources/fred.py's INDUSTRY_CHARGEOFF_DELINQUENCY_SERIES)
-- used only to extend model family 1's (the aggregate AR model's)
training window back to 1991Q1, well before the Call Report panel's own
2001Q1 start, so it can see the early-1990s CRE bust that the 2001-2006
Call Report window misses entirely. Bank-level models stay on the Call
Report panel (see models.py) -- these FRED series have no bank-level
breakdown to join against individual banks' characteristics. No function
here touches the network -- fetching is sources/fred.py's job,
orchestration is the CLI's.
"""

from __future__ import annotations

import pandas as pd

from corefin.credit import models


def build_long_industry_rate(raw_observations: pd.DataFrame) -> pd.Series:
    """raw_observations: fred.fetch_series(...) output (columns "date",
    "value") for one category's chargeoff_series_id or
    delinquency_series_id (sources.fred.INDUSTRY_CHARGEOFF_DELINQUENCY_
    SERIES). Both are already quarterly and already annualized where
    relevant, in PERCENT -- this converts to the decimal-fraction scale
    this project's own rate columns use (5.80 -> 0.0580) and indexes by
    quarter (PeriodIndex, freq="Q")."""
    quarters = pd.PeriodIndex(raw_observations["date"], freq="Q")
    rate = raw_observations["value"] / 100.0
    rate.index = quarters
    return rate.rename_axis("quarter").rename("industry_rate")


def build_long_industry_frame(
    long_rate: pd.Series, macro_history: pd.DataFrame
) -> pd.DataFrame:
    """Joins a FRED-derived long-history industry rate (`build_long_
    industry_rate`'s output) with the Fed's own macro history (quarter-
    indexed, as from macro.normalize_fed_historic), at models.LAG, for
    models.CORE_MACRO_FEATURES -- producing exactly the same shape
    models.build_industry_series(...).frame produces (columns "quarter",
    "industry_rate", "industry_rate_lag1", one models.feature_column(...)
    per CORE_MACRO_FEATURES, and "pandemic"), so models.fit_aggregate_
    model/forecast_aggregate_dynamic work UNCHANGED against it."""
    frame = pd.DataFrame({"industry_rate": long_rate.to_numpy()}, index=long_rate.index)
    frame.index = frame.index.rename("quarter")
    frame = frame.reset_index()
    frame = frame.sort_values("quarter").reset_index(drop=True)
    for variable in models.CORE_MACRO_FEATURES:
        lookup = macro_history[variable]
        lagged_quarter = frame["quarter"] - models.LAG
        frame[models.feature_column(variable)] = lagged_quarter.map(lookup)
    frame["pandemic"] = models.add_pandemic_indicator(frame)
    frame["industry_rate_lag1"] = frame["industry_rate"].shift(1)
    return frame


def compare_with_call_report_aggregate(
    long_industry_frame: pd.DataFrame, call_report_industry_frame: pd.DataFrame
) -> pd.DataFrame:
    """Both frames as produced by `build_long_industry_frame` /
    models.build_industry_series(...).frame. Returns a quarter-indexed
    comparison over their OVERLAPPING quarters only: fred_rate,
    call_report_rate, and their difference -- this project's discipline
    is to verify two independently-sourced series actually agree, not
    assume it, before treating a FRED-derived series as a trustworthy
    stand-in for missing Call Report history."""
    fred_series = long_industry_frame.set_index("quarter")["industry_rate"]
    call_report_series = call_report_industry_frame.set_index("quarter")["industry_rate"]
    aligned = pd.concat(
        {"fred_rate": fred_series, "call_report_rate": call_report_series}, axis=1
    ).dropna()
    aligned["difference"] = aligned["fred_rate"] - aligned["call_report_rate"]
    return aligned.reset_index()
