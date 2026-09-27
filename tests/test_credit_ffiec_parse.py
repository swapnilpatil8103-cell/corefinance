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


def test_parse_bulk_zip_reads_the_plain_rc_schedule_for_allowance_items():
    # "Schedule RC " (with the trailing space) must match only the plain
    # RC schedule, not RCCI/RCCII/RCA/RCR/etc -- confirmed against a real
    # bulk ZIP, RCON3123/RCFD3123 (allowance) live only in this schedule.
    rc = '"IDRSSD"\tRCON3123\tRCFD3123\n\tdesc\tdesc\n37\t100\t\n'
    rca = '"IDRSSD"\tRCONXXXX\n\tdesc\n37\t999\n'  # decoy -- must not be picked up as "RC"
    zip_bytes = _make_zip(
        {
            "FFIEC CDR Call Schedule RC 12312021.txt": rc,
            "FFIEC CDR Call Schedule RCA 12312021.txt": rca,
        }
    )
    result = ffiec_parse.parse_bulk_zip(zip_bytes, pd.Period("2021Q4", freq="Q"))
    assert result["RCON3123"].iloc[0] == pytest.approx(100.0)
    assert "RCONXXXX" not in result.columns


def test_parse_bulk_zip_reads_the_rie_schedule_for_cecl_adoption_items():
    rie = '"IDRSSD"\tRIADJJ26\tRIADJJ28\n\tdesc\tdesc\n37\t1500\t1500\n'
    zip_bytes = _make_zip(
        {
            "FFIEC CDR Call Schedule RCCI 12312021.txt": '"IDRSSD"\tRCON1766\n\tdesc\n37\t100\n',
            "FFIEC CDR Call Schedule RIE 12312021.txt": rie,
        }
    )
    result = ffiec_parse.parse_bulk_zip(zip_bytes, pd.Period("2021Q4", freq="Q"))
    assert result["RIADJJ26"].iloc[0] == pytest.approx(1500.0)
    assert result["RIADJJ28"].iloc[0] == pytest.approx(1500.0)


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


def test_parse_bulk_zip_skips_a_malformed_row_instead_of_crashing():
    # Confirmed against real data (2004Q1's RIE schedule): a rare row has
    # one extra tab-separated field (a stray literal tab inside a
    # free-text narrative column) -- must be skipped, not crash the parse.
    rcci = '"IDRSSD"\tRCON1766\n\tdesc\n37\t100\n'
    malformed_rie = (
        '"IDRSSD"\tRIADJJ26\n'
        "\tdesc\n"
        "37\t1500\n"
        "242\t1\t2\n"  # malformed: one extra field
        "555\t900\n"
    )
    zip_bytes = _make_zip(
        {
            "FFIEC CDR Call Schedule RCCI 12312021.txt": rcci,
            "FFIEC CDR Call Schedule RIE 12312021.txt": malformed_rie,
        }
    )
    result = ffiec_parse.parse_bulk_zip(zip_bytes, pd.Period("2021Q4", freq="Q"))
    # bank 242's malformed row is skipped -- bank 242 itself never appears
    # in RCCI here, so it's simply absent from the result entirely.
    assert "242" not in set(result["bank_id"])
    assert result.loc[result["bank_id"] == "37", "RIADJJ26"].iloc[0] == pytest.approx(1500.0)
