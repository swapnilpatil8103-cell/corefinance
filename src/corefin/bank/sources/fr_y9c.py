"""Parses an FR Y-9C bulk file (bank HOLDING COMPANY consolidated
financial statements -- the Federal Reserve's FFIEC NIC bulk download,
named `BHCF<YYYYMMDD>.ZIP`), used for capital/RWA/earnings/equity per the
level-of-consolidation choice documented in `corefin.bank.schema`.

FORMAT, confirmed against a real downloaded 2025Q4 file
(BHCF20251231.ZIP): a SINGLE flat text file inside the ZIP, CARET-
separated (not tab-separated like the Call Report's per-schedule files --
a real, confirmed difference, not an assumption), with exactly ONE header
row (no 3-row structure to drop, unlike `ffiec_parse`'s Call Report
schedules). `RSSD9001` is the holding company's own RSSD ID; item columns
use BHCK/BHCA/BHCP/BHCW-style prefixes that largely (but not always, per
`corefin.bank.schema`'s module docstring) reuse the same numeric MDRM
suffix as the equivalent Call Report item.

FETCHING: unlike the Call Report bulk ZIPs (`corefin.credit.sources.ffiec`,
a plain `requests` GET against FFIEC's bulk-download host), the FFIEC NIC
site that serves FR Y-9C bulk files
(https://www.ffiec.gov/npw/FinancialReport/FinancialDataDownload) sits
behind a Cloudflare JS challenge -- confirmed via direct inspection of the
real network traffic: both a plain `requests`-style GET and a spoofed-
User-Agent `curl` request both receive Cloudflare's "CAPTCHA Error" page
(not the real ZIP), while a real browser session's own authenticated
`fetch()` succeeds, preceded by a genuine
`cdn-cgi/challenge-platform/...` JS challenge POST. There is deliberately
NO automated fetcher here (unlike `credit.sources.ffiec`) -- downloading
a bulk file is a one-off, infrequent, manual/browser-assisted step; this
module only parses a file already obtained that way."""

from __future__ import annotations

import zipfile
from io import BytesIO, TextIOWrapper

import pandas as pd

RSSD_COLUMN = "RSSD9001"
NAME_COLUMN = "RSSD9017"
_ID_COLUMNS = (RSSD_COLUMN, NAME_COLUMN)
_ITEM_PREFIXES = ("BHCK", "BHCA", "BHCP", "BHCT", "BHCW", "BHDM", "BHFN")


def parse_y9c_bulk_zip(zip_bytes: bytes, quarter: pd.Period) -> pd.DataFrame:
    """Returns one row per holding company: "rssd_id" (string), "name",
    "quarter", plus every BHCK/BHCA/BHCP/BHCT/BHCW/BHDM/BHFN-prefixed item
    column, coerced to float (FR Y-9C dollar amounts are in thousands,
    same convention as the Call Report). Raises ValueError if the ZIP
    doesn't contain exactly one file."""
    zf = zipfile.ZipFile(BytesIO(zip_bytes))
    names = zf.namelist()
    if len(names) != 1:
        raise ValueError(f"expected exactly one file in the FR Y-9C bulk ZIP, found {names}")

    with zf.open(names[0]) as fh:
        text = TextIOWrapper(fh, encoding="latin1")
        raw = pd.read_csv(text, sep="^", dtype=str, engine="python")

    if RSSD_COLUMN not in raw.columns:
        raise ValueError(f"expected an {RSSD_COLUMN} column, found columns: {list(raw.columns)}")

    item_columns = [c for c in raw.columns if c.startswith(_ITEM_PREFIXES)]
    numeric_items = raw[item_columns].apply(pd.to_numeric, errors="coerce")

    id_frame = raw[[RSSD_COLUMN]].rename(columns={RSSD_COLUMN: "rssd_id"}).astype(str)
    name_frame = (
        raw[[NAME_COLUMN]].rename(columns={NAME_COLUMN: "name"})
        if NAME_COLUMN in raw.columns
        else pd.DataFrame({"name": [None] * len(raw)})
    )
    quarter_frame = pd.DataFrame({"quarter": [quarter] * len(raw)})

    result = pd.concat([id_frame, name_frame, quarter_frame, numeric_items], axis=1).copy()
    result = result[result["rssd_id"].str.strip() != ""].reset_index(drop=True)
    return result


def find_holding_company_by_name(item_frame: pd.DataFrame, name_contains: str) -> pd.DataFrame:
    """Case-insensitive substring lookup against the "name" column --
    convenience for finding a holding company's own `rssd_id` ahead of
    time (used once, outside any committed pipeline, to resolve the
    example deal's two acquirer/target holding companies to RSSD IDs for
    the config file -- see module docstring for why this can't be
    automated via a bulk search API)."""
    mask = item_frame["name"].str.contains(name_contains, case=False, na=False)
    return item_frame.loc[mask, ["rssd_id", "name"]]
