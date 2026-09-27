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
contributes a NaN "phantom bank"), but wrong, so it's dropped explicitly
after parsing (not via `skiprows`, which pandas' `engine="pyarrow"` only
accepts as a plain row count, not a specific row index).

A schedule that exceeds FFIEC's per-file column limit is split into
multiple "(N of M)" files -- confirmed these share the exact same rows
(same IDRSSD values, same order) but different columns, so they're
concatenated along columns (axis=1), not rows.

Uses `engine="pyarrow"` specifically so `on_bad_lines` can be a counting
callback -- confirmed against real data (2004Q1's RIE schedule) that a
rare row has one extra tab-separated field (a stray literal tab inside
one of that schedule's free-text narrative columns), which the default C
engine cannot recover from at all. Skipping that one bank's one row for
that one schedule/quarter is far better than the whole build crashing; an
outer merge later in this module means the bank still appears via its
OTHER schedules, just missing the columns unique to the skipped row.
Every skip is counted per (schedule, quarter) and returned to the caller
-- see `parse_bulk_zip`'s return value and BAD_ROW_WARNING_THRESHOLD.
"""

from __future__ import annotations

import zipfile
from io import BytesIO, TextIOWrapper

import pandas as pd

_ITEM_PREFIXES = ("RCON", "RCFD", "RIAD")

# Short label -> the "Schedule XXX " substring that uniquely matches only
# that schedule's file name (the trailing space excludes similarly-named
# schedules, e.g. "Schedule RC " vs "Schedule RCCI "/"RCA"/"RCR"/etc).
_SCHEDULE_NAME_CONTAINS = {
    "RC": "Schedule RC ",
    "RCCI": "Schedule RCCI ",
    "RCCII": "Schedule RCCII ",
    "RCN": "Schedule RCN ",
    "RIBI": "Schedule RIBI ",
    "RIBII": "Schedule RIBII ",
    # RIADJJ26/JJ28 (CECL adoption indicators, 2019Q1-2023Q4 only) live here
    "RIE": "Schedule RIE ",
}

# "More than a handful" of skipped rows in one (schedule, quarter) is
# treated as worth a warning -- a single stray malformed row is a known,
# harmless data quirk (see the module docstring), but a bigger count could
# mean something structurally wrong with that file.
BAD_ROW_WARNING_THRESHOLD = 5


def _read_schedule_part(zf: zipfile.ZipFile, name: str) -> tuple[pd.DataFrame, int]:
    bad_line_count = 0

    def _count_bad_line(bad_line: list[str]) -> str:
        nonlocal bad_line_count
        bad_line_count += 1
        return "skip"

    with zf.open(name) as fh:
        text = TextIOWrapper(fh, encoding="latin1")
        df = pd.read_csv(
            text, sep="\t", dtype=str, on_bad_lines=_count_bad_line, engine="pyarrow"
        )
    df = df.iloc[1:].reset_index(drop=True)  # drop the per-column description row
    return df, bad_line_count


def _read_schedule(zf: zipfile.ZipFile, name_contains: str) -> tuple[pd.DataFrame | None, int]:
    """Reads every part of a (possibly multi-part) schedule and
    concatenates them along columns. Returns (None, 0) if no file in the
    ZIP matches `name_contains`."""
    matches = sorted(n for n in zf.namelist() if name_contains in n)
    if not matches:
        return None, 0
    parts = []
    total_bad_lines = 0
    for name in matches:
        part, bad_lines = _read_schedule_part(zf, name)
        parts.append(part)
        total_bad_lines += bad_lines
    if len(parts) == 1:
        return parts[0], total_bad_lines
    id_col = next(c for c in parts[0].columns if "IDRSSD" in c.upper())
    combined = parts[0]
    for part in parts[1:]:
        overlap = [c for c in part.columns if c != id_col and c in combined.columns]
        combined = pd.concat([combined, part.drop(columns=[id_col, *overlap])], axis=1)
    return combined, total_bad_lines


def parse_bulk_zip(zip_bytes: bytes, quarter: pd.Period) -> tuple[pd.DataFrame, dict[str, int]]:
    """Returns (item_frame, bad_row_counts).

    item_frame: one row per bank ("bank_id", "quarter", plus every
    RCON/RCFD/RIAD item column found across the Call Report schedules
    that carry loan balances (RCCI/RCCII), past-due/nonaccrual (RCN),
    charge-offs/recoveries/allowance/provision (RIBI/RIBII) and CECL
    adoption indicators (RIE) -- the schedules schema.py's item codes
    live in. Item columns are coerced to float (Call Report dollar
    amounts are in thousands); non-numeric/blank cells become NaN, not
    zero.

    bad_row_counts: {schedule_label: n_rows_skipped} for every schedule
    present in this ZIP (0 for schedules with no skipped rows) -- see
    BAD_ROW_WARNING_THRESHOLD. A schedule absent from the ZIP entirely is
    absent from this dict too (not the same as 0 skipped rows)."""
    zf = zipfile.ZipFile(BytesIO(zip_bytes))

    schedules: dict[str, pd.DataFrame | None] = {}
    bad_row_counts: dict[str, int] = {}
    for label, name_contains in _SCHEDULE_NAME_CONTAINS.items():
        schedule, bad_lines = _read_schedule(zf, name_contains)
        schedules[label] = schedule
        if schedule is not None:
            bad_row_counts[label] = bad_lines

    present = [s for s in schedules.values() if s is not None]
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
    result = result[result["bank_id"].str.strip() != ""].reset_index(drop=True)
    return result, bad_row_counts
