"""Parses an FFIEC bulk Call Report "All Schedules" ZIP (as returned by
`ffiec.fetch_bulk_call_report_zip`) into a single wide, one-row-per-bank
item frame with every RCON/RCFD/RIAD-prefixed column, ready for
`panel.build_panel`.

Each schedule .txt file inside the ZIP has THREE header-ish rows, not
one -- confirmed by inspecting a real file's raw bytes: row 1 is the
column names ("IDRSSD", "RCON1766", ...), row 2 is a human-readable
description PER COLUMN (not data -- its own "IDRSSD" cell is blank), and
row 3 onward is real data. Reading naively with pandas' default header
handling folds that description row in as a garbage first data row
(IDRSSD becomes NaN) -- silently harmless for an aggregate sum (it just
contributes a NaN "phantom bank"), but wrong, so it's skipped explicitly.

A schedule that exceeds FFIEC's per-file column limit is split into
multiple "(N of M)" files -- confirmed these share the exact same rows
(same IDRSSD values, same order) but different columns, so they're
concatenated along columns (axis=1), not rows.
"""

from __future__ import annotations

import zipfile
from io import BytesIO, TextIOWrapper

import pandas as pd

_ITEM_PREFIXES = ("RCON", "RCFD", "RIAD")


def _read_schedule_part(zf: zipfile.ZipFile, name: str) -> pd.DataFrame:
    with zf.open(name) as fh:
        text = TextIOWrapper(fh, encoding="latin1")
        # on_bad_lines="skip": confirmed against real data (2004Q1's RIE
        # schedule) that a rare row has one extra tab-separated field --
        # a stray literal tab inside one of that schedule's free-text
        # narrative columns (a bank's own description of a misc income/
        # expense line), not a structural problem with the file. Skipping
        # that one bank's one row for that one schedule/quarter is far
        # better than the whole build crashing; an outer merge elsewhere
        # in this module means the bank still appears via its OTHER
        # schedules, just missing the columns unique to this one.
        return pd.read_csv(
            text, sep="\t", dtype=str, skiprows=[1], on_bad_lines="skip", engine="c"
        )


def _read_schedule(zf: zipfile.ZipFile, name_contains: str) -> pd.DataFrame | None:
    """Reads every part of a (possibly multi-part) schedule and
    concatenates them along columns. Returns None if no file in the ZIP
    matches `name_contains`."""
    matches = sorted(n for n in zf.namelist() if name_contains in n)
    if not matches:
        return None
    parts = [_read_schedule_part(zf, name) for name in matches]
    if len(parts) == 1:
        return parts[0]
    id_col = next(c for c in parts[0].columns if "IDRSSD" in c.upper())
    combined = parts[0]
    for part in parts[1:]:
        overlap = [c for c in part.columns if c != id_col and c in combined.columns]
        combined = pd.concat([combined, part.drop(columns=[id_col, *overlap])], axis=1)
    return combined


def parse_bulk_zip(zip_bytes: bytes, quarter: pd.Period) -> pd.DataFrame:
    """Returns one row per bank ("bank_id", "quarter", plus every
    RCON/RCFD/RIAD item column found across the Call Report schedules
    that carry loan balances (RCCI/RCCII), past-due/nonaccrual (RCN) and
    charge-offs/recoveries/allowance/provision (RIBI/RIBII) -- the
    schedules schema.CATEGORY_MDRM_CODES's item codes live in. Item
    columns are coerced to float (Call Report dollar amounts are in
    thousands); non-numeric/blank cells become NaN, not zero."""
    zf = zipfile.ZipFile(BytesIO(zip_bytes))

    schedules = [
        _read_schedule(zf, "Schedule RC "),  # trailing space -- excludes RCCI/RCCII/RCN/RCA/etc
        _read_schedule(zf, "Schedule RCCI "),
        _read_schedule(zf, "Schedule RCCII "),
        _read_schedule(zf, "Schedule RCN "),
        _read_schedule(zf, "Schedule RIBI "),
        _read_schedule(zf, "Schedule RIBII "),
        # RIADJJ26/JJ28 (CECL adoption indicators, 2019Q1-2023Q4 only) live here
        _read_schedule(zf, "Schedule RIE "),
    ]
    present = [s for s in schedules if s is not None]
    if not present:
        raise ValueError(
            "none of the expected schedule files (RC/RCCI/RCCII/RCN/RIBI/RIBII/RIE) "
            "were found in the ZIP"
        )

    id_col = next(c for c in present[0].columns if "IDRSSD" in c.upper())
    merged = present[0].rename(columns={id_col: "bank_id"})
    for schedule in present[1:]:
        this_id_col = next(c for c in schedule.columns if "IDRSSD" in c.upper())
        schedule = schedule.rename(columns={this_id_col: "bank_id"})
        overlap = [c for c in schedule.columns if c != "bank_id" and c in merged.columns]
        merged = merged.merge(schedule.drop(columns=overlap), on="bank_id", how="outer")

    item_columns = [c for c in merged.columns if c.startswith(_ITEM_PREFIXES)]
    numeric_items = merged[item_columns].apply(pd.to_numeric, errors="coerce")
    bank_id_col = merged[["bank_id"]].astype(str)
    quarter_col = pd.DataFrame({"quarter": [quarter] * len(merged)})
    result = pd.concat([bank_id_col, quarter_col, numeric_items], axis=1).copy()
    return result[result["bank_id"].str.strip() != ""].reset_index(drop=True)
