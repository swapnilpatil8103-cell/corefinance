"""PPNR (pre-provision net revenue) stress: makes net interest income
and noninterest income respond to a projection's own macro scenario,
instead of `income_statement.py`'s default "hold flat" convention --
confirmed too optimistic under a real severely-adverse run (Project
#2's Stage 5 stress test, `ma/stress.py`, only ever had CREDIT losses
respond to the scenario; NII/noninterest income/expense didn't move at
all, which can make a bank's stressed CET1 look stronger than it
should).

NII: `NIM_t = NIM_jumpoff + nim_beta x (rate_t - rate_jumpoff)`, both the
NIM change and the rate change in PERCENTAGE POINTS -- the standard,
simple "rate beta" convention banks themselves use for asset-
sensitivity: `nim_beta` > 0 means NIM moves in the SAME direction as
rates (an ASSET-SENSITIVE bank LOSES margin when rates FALL -- more of
its assets reprice down than its liabilities do). `NIM_t` is ANNUALIZED
(the same basis `calibrate_nim_beta` uses: `quarterly_NII x 4 /
total_assets`), so `NII_t = NIM_t x earning_assets_jumpoff_mm / 4` --
held STATIC, the same static-balance-sheet convention this project uses
throughout. SIMPLIFICATION: "earning assets" is approximated as TOTAL
assets (no loan/securities/cash decomposition) -- both when calibrating
beta below and when applying it in `income_statement.py`, for
consistency between the two.

CALIBRATION: `calibrate_nim_beta` derives an IMPLIED beta from a bank's
own REALIZED NIM change over 2020Q1-2021Q4 against the SAME "3-month
Treasury rate" variable's own REALIZED change over that window -- the
one real historical episode in this project's cached Call Report data
where a short-term rate collapsed (~100bp, 2020Q1 to the COVID-era
near-zero floor) closely analogous to the Fed's own severely-adverse
rate-collapse scenario. `DEFAULT_NIM_BETA` is an ILLUSTRATIVE fallback
for when calibration isn't available (e.g. a bank with incomplete
2020-2021 Call Report history) -- flagged as illustrative, not derived,
the same documentation standard this project applies to its other
no-real-data-basis assumptions.

NONINTEREST INCOME: declines by a configurable FLAT percentage under
stress (`noninterest_income_decline_pct`, ILLUSTRATIVE default) --
simpler than a GDP-linked decline (a defensible, documented alternative
this module does NOT implement, to keep the stress model genuinely
simple as requested). NONINTEREST EXPENSE is held flat with NO growth
at all once PPNR stress is active (sticky in a downturn -- a real,
well-documented banking-industry pattern: cost bases don't shrink as
fast as revenue does in a downturn, so holding them flat is more
realistic than letting them decline alongside revenue)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from corefin.bank.schema import NET_INTEREST_INCOME_ITEMS, TOTAL_ASSETS_ITEM

CALIBRATION_START_QUARTER = pd.Period("2020Q1", freq="Q")
CALIBRATION_END_QUARTER = pd.Period("2021Q4", freq="Q")
RATE_VARIABLE = "3-month Treasury rate"


def _quarterly_from_ytd(
    rows_by_quarter: dict[pd.Period, pd.Series], item: str
) -> dict[pd.Period, float]:
    """quarter -> quarterly (not YTD) `item` value, differencing each
    quarter's own YTD figure against the SAME calendar year's prior
    quarter (must also be in `rows_by_quarter`) -- a Q1 row's own YTD
    figure is already quarterly (the calendar year just started)."""
    quarterly: dict[pd.Period, float] = {}
    for quarter, row in rows_by_quarter.items():
        if quarter.quarter == 1:
            quarterly[quarter] = float(row[item])
            continue
        prior_quarter = quarter - 1
        if prior_quarter not in rows_by_quarter:
            continue
        quarterly[quarter] = float(row[item]) - float(rows_by_quarter[prior_quarter][item])
    return quarterly


def _quarterly_nii(rows_by_quarter: dict[pd.Period, pd.Series]) -> dict[pd.Period, float]:
    income_item, expense_item = NET_INTEREST_INCOME_ITEMS
    income = _quarterly_from_ytd(rows_by_quarter, income_item)
    expense = _quarterly_from_ytd(rows_by_quarter, expense_item)
    return {q: income[q] - expense[q] for q in income if q in expense}


@dataclass(frozen=True)
class CalibratedNimBeta:
    bank_id: str
    start_quarter: pd.Period
    end_quarter: pd.Period
    start_nim: float  # annualized decimal
    end_nim: float
    realized_nim_change_pp: float
    realized_rate_change_pp: float
    nim_beta: float | None  # None if the rate barely moved (division by ~0)


def calibrate_nim_beta(
    bank_id: str,
    call_report_rows_by_quarter: dict[pd.Period, pd.Series],
    rate_by_quarter: pd.Series,
    start_quarter: pd.Period = CALIBRATION_START_QUARTER,
    end_quarter: pd.Period = CALIBRATION_END_QUARTER,
) -> CalibratedNimBeta | None:
    """Returns `None` if `call_report_rows_by_quarter` doesn't fully
    cover the calibration window (each bank-quarter needs ITS OWN row,
    plus the prior quarter's within the same calendar year, for the
    YTD-to-quarterly differencing `_quarterly_from_ytd` performs) or if
    `rate_by_quarter` is missing either endpoint."""
    if (
        start_quarter not in call_report_rows_by_quarter
        or end_quarter not in call_report_rows_by_quarter
        or start_quarter not in rate_by_quarter.index
        or end_quarter not in rate_by_quarter.index
    ):
        return None
    quarterly_nii = _quarterly_nii(call_report_rows_by_quarter)
    if start_quarter not in quarterly_nii or end_quarter not in quarterly_nii:
        return None

    def _nim(quarter: pd.Period) -> float | None:
        total_assets = float(call_report_rows_by_quarter[quarter][TOTAL_ASSETS_ITEM])
        if total_assets <= 0:
            return None
        return (quarterly_nii[quarter] * 4.0) / total_assets

    start_nim, end_nim = _nim(start_quarter), _nim(end_quarter)
    if start_nim is None or end_nim is None:
        return None

    realized_nim_change_pp = (end_nim - start_nim) * 100.0
    realized_rate_change_pp = float(
        rate_by_quarter.loc[end_quarter] - rate_by_quarter.loc[start_quarter]
    )
    nim_beta = (
        realized_nim_change_pp / realized_rate_change_pp
        if abs(realized_rate_change_pp) > 1e-9
        else None
    )
    return CalibratedNimBeta(
        bank_id=bank_id,
        start_quarter=start_quarter,
        end_quarter=end_quarter,
        start_nim=start_nim,
        end_nim=end_nim,
        realized_nim_change_pp=realized_nim_change_pp,
        realized_rate_change_pp=realized_rate_change_pp,
        nim_beta=nim_beta,
    )


def align_rate_path_to_timeline(
    jumpoff_rate_pp: float, projected_rate_path_pp: np.ndarray, n_periods: int
) -> np.ndarray:
    """Builds a `(n_periods,)` rate path (percentage points) with period
    0 = `jumpoff_rate_pp` (the real, actual rate at the jump-off quarter
    -- a scenario's own projected path never includes it) followed by
    `projected_rate_path_pp` (the scenario's own quarter-indexed path,
    sorted), padded flat at its own last value if `n_periods` needs more
    quarters than the scenario natively covers (the same flat-
    continuation rule `ma.horizon.extend_credit_loss_projection` uses),
    or truncated if `n_periods` needs fewer."""
    full = np.concatenate([[jumpoff_rate_pp], projected_rate_path_pp])
    if len(full) >= n_periods:
        return full[:n_periods]
    pad = np.full(n_periods - len(full), full[-1])
    return np.concatenate([full, pad])


def compute_stressed_nii_mm(
    nii_jumpoff_mm: float,
    earning_assets_jumpoff_mm: float,
    rate_path_pp: np.ndarray,
    nim_beta: float,
) -> np.ndarray:
    """`rate_path_pp`: a scenario's own rate path (e.g. "3-month
    Treasury rate"), in PERCENTAGE POINTS, aligned to the model's
    timeline (`align_rate_path_to_timeline`) -- `rate_path_pp[0]` is the
    jump-off's own ACTUAL rate. Returns NII at jump-off PLUS the rate-
    driven delta -- period 0 is exactly `nii_jumpoff_mm` by
    construction, not a replacement for the real, reported figure.

    UNITS: `nim_beta` is calibrated (`calibrate_nim_beta`) against NIM
    ANNUALIZED (`quarterly_NII x 4 / total_assets`), so a `nim_beta x
    rate_change_pp` NIM move is itself an ANNUAL margin change --
    dividing by 4 again converts it to the QUARTERLY dollar NII delta
    `nii_jumpoff_mm` (already quarterly) needs."""
    rate_change_pp = rate_path_pp - rate_path_pp[0]
    annual_nim_change_decimal = nim_beta * rate_change_pp / 100.0
    nii_delta_mm = (annual_nim_change_decimal * earning_assets_jumpoff_mm) / 4.0
    return nii_jumpoff_mm + nii_delta_mm


def compute_stressed_noninterest_income_mm(
    noninterest_income_jumpoff_mm: float, n_periods: int, decline_pct: float
) -> np.ndarray:
    """Jump-off (period 0) unchanged; every PROJECTED period held at
    `noninterest_income_jumpoff_mm x (1 - decline_pct)` -- a flat, one-
    time step down, not a gradual decline (see module docstring)."""
    path = np.full(n_periods, noninterest_income_jumpoff_mm * (1.0 - decline_pct))
    path[0] = noninterest_income_jumpoff_mm
    return path
