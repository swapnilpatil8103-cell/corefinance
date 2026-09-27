"""Offline tests for the bulk-ZIP parser -- builds tiny synthetic ZIPs in
memory reproducing the real file's quirks (a per-column description row,
and multi-part files sharing rows but not columns); no network."""

import zipfile
from io import BytesIO

import pandas as pd
import pytest

from corefin.credit.sources import ffiec_parse


def _make_zip(files: dict[str, str]) -> bytes:
    buf = BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for name, content in files.items():
            zf.writestr(name, content)
    return buf.getvalue()


def test_parse_bulk_zip_skips_the_description_row():
    rcci = (
        '"IDRSSD"\tRCON1766\n'
        "\tCOMMERCIAL AND INDUSTRIAL LOANS\n"  # description row -- must be skipped
        "37\t1000\n"
        "242\t2000\n"
    )
    zip_bytes = _make_zip({"FFIEC CDR Call Schedule RCCI 12312021.txt": rcci})
    result = ffiec_parse.parse_bulk_zip(zip_bytes, pd.Period("2021Q4", freq="Q"))

    assert len(result) == 2  # the description row must not appear as a phantom bank
    assert set(result["bank_id"]) == {"37", "242"}
    assert result.loc[result["bank_id"] == "37", "RCON1766"].iloc[0] == pytest.approx(1000.0)


def test_parse_bulk_zip_concatenates_multi_part_files_by_column_not_row():
    part1 = '"IDRSSD"\tRCON1606\n\tdesc\n37\t10\n242\t20\n'
    part2 = '"IDRSSD"\tRCON1607\n\tdesc\n37\t5\n242\t8\n'
    zip_bytes = _make_zip(
        {
            "FFIEC CDR Call Schedule RCN 12312021(1 of 2).txt": part1,
            "FFIEC CDR Call Schedule RCN 12312021(2 of 2).txt": part2,
        }
    )
    result = ffiec_parse.parse_bulk_zip(zip_bytes, pd.Period("2021Q4", freq="Q"))

    assert len(result) == 2
    row = result[result["bank_id"] == "37"].iloc[0]
    assert row["RCON1606"] == pytest.approx(10.0)
    assert row["RCON1607"] == pytest.approx(5.0)


def test_parse_bulk_zip_merges_across_schedules_on_bank_id():
    rcci = '"IDRSSD"\tRCON1766\n\tdesc\n37\t1000\n'
    ribi = '"IDRSSD"\tRIAD4638\n\tdesc\n37\t50\n'
    zip_bytes = _make_zip(
        {
            "FFIEC CDR Call Schedule RCCI 12312021.txt": rcci,
            "FFIEC CDR Call Schedule RIBI 12312021.txt": ribi,
        }
    )
    result = ffiec_parse.parse_bulk_zip(zip_bytes, pd.Period("2021Q4", freq="Q"))

    assert len(result) == 1
    assert result["RCON1766"].iloc[0] == pytest.approx(1000.0)
    assert result["RIAD4638"].iloc[0] == pytest.approx(50.0)
    assert result["quarter"].iloc[0] == pd.Period("2021Q4", freq="Q")


def test_parse_bulk_zip_raises_when_no_expected_schedule_is_present():
    rca = '"IDRSSD"\tRCON1\n\tdesc\n37\t1\n'
    zip_bytes = _make_zip({"FFIEC CDR Call Schedule RCA 12312021.txt": rca})
    with pytest.raises(ValueError, match="RCCI"):
        ffiec_parse.parse_bulk_zip(zip_bytes, pd.Period("2021Q4", freq="Q"))


def test_parse_bulk_zip_coerces_non_numeric_cells_to_nan():
    rcci = '"IDRSSD"\tRCON1766\n\tdesc\n37\t\n242\t2000\n'  # blank cell for bank 37
    zip_bytes = _make_zip({"FFIEC CDR Call Schedule RCCI 12312021.txt": rcci})
    result = ffiec_parse.parse_bulk_zip(zip_bytes, pd.Period("2021Q4", freq="Q"))
    assert result.loc[result["bank_id"] == "37", "RCON1766"].isna().iloc[0]
