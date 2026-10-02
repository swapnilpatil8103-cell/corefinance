"""Parses the Call Report schedules the bank statement model needs beyond
what `corefin.credit.sources.ffiec_parse.parse_bulk_zip` already reads by
default: Schedule RI (interest income/expense, pretax/net income, taxes),
Schedule RC-M (goodwill, other intangibles), and Schedule RC-R Part I
(regulatory capital ratios -- RCOA/RCFA-prefixed, not covered by the
default RCON/RCFD/RIAD item prefixes). Schedule RC itself (total assets,
other assets, total deposits, total liabilities, total equity,
intangible-assets total) is already read by `parse_bulk_zip`'s default
schedule set -- see `corefin.bank.schema`'s item-code constants, all of
which this module's output makes available via the additive
`extra_schedules`/`extra_item_prefixes` parameters, not by duplicating any
of `ffiec_parse`'s own parsing logic."""

from __future__ import annotations

import pandas as pd

from corefin.credit.sources.ffiec_parse import parse_bulk_zip

_EXTRA_SCHEDULES = {
    "RI": "Schedule RI ",
    "RCM": "Schedule RCM ",
    "RCRI": "Schedule RCRI ",
}
_EXTRA_ITEM_PREFIXES = ("RCOA", "RCFA")


def parse_bank_capital_zip(
    zip_bytes: bytes, quarter: pd.Period
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Returns (item_frame, bad_row_counts) with the same shape/convention
    as `ffiec_parse.parse_bulk_zip` (one row per bank_id, item columns
    coerced to float), but also carrying RI/RC-M/RC-R Part I item columns
    on top of `ffiec_parse`'s own default schedule set (RC/RCCI/RCCII/RCN/
    RIBI/RIBII/RIE)."""
    return parse_bulk_zip(
        zip_bytes,
        quarter,
        extra_schedules=_EXTRA_SCHEDULES,
        extra_item_prefixes=_EXTRA_ITEM_PREFIXES,
    )
