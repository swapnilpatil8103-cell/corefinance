"""Offline tests for corefin.bank.sources.capital_parse -- confirms the
RC-M/RC-R-Part-I/RI schedules are read on top of ffiec_parse's own
default set, using the same tiny-synthetic-ZIP pattern as
test_credit_ffiec_parse.py."""

import zipfile
from io import BytesIO

import pandas as pd
import pytest

from corefin.bank.sources import capital_parse


def _make_zip(files: dict[str, str]) -> bytes:
    buf = BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for name, content in files.items():
            zf.writestr(name, content)
    return buf.getvalue()


def test_parse_bank_capital_zip_reads_rc_schedule_balance_sheet_items():
    rc = '"IDRSSD"\tRCON2170\tRCON2200\n\tdesc\tdesc\n37\t5000\t3000\n'
    rcci = '"IDRSSD"\tRCON1766\n\tdesc\n37\t100\n'
    zip_bytes = _make_zip(
        {
            "FFIEC CDR Call Schedule RC 12312021.txt": rc,
            "FFIEC CDR Call Schedule RCCI 12312021.txt": rcci,
        }
    )
    result, _ = capital_parse.parse_bank_capital_zip(zip_bytes, pd.Period("2021Q4", freq="Q"))
    assert result["RCON2170"].iloc[0] == pytest.approx(5000.0)
    assert result["RCON2200"].iloc[0] == pytest.approx(3000.0)


def test_parse_bank_capital_zip_reads_rcm_schedule_goodwill_and_intangibles():
    rcci = '"IDRSSD"\tRCON1766\n\tdesc\n37\t100\n'
    rcm = '"IDRSSD"\tRCON3163\tRCONJF76\n\tdesc\tdesc\n37\t200\t50\n'
    zip_bytes = _make_zip(
        {
            "FFIEC CDR Call Schedule RCCI 12312021.txt": rcci,
            "FFIEC CDR Call Schedule RCM 12312021.txt": rcm,
        }
    )
    result, _ = capital_parse.parse_bank_capital_zip(zip_bytes, pd.Period("2021Q4", freq="Q"))
    assert result["RCON3163"].iloc[0] == pytest.approx(200.0)
    assert result["RCONJF76"].iloc[0] == pytest.approx(50.0)


def test_parse_bank_capital_zip_reads_ri_schedule_income_statement_items():
    rcci = '"IDRSSD"\tRCON1766\n\tdesc\n37\t100\n'
    ri = (
        '"IDRSSD"\tRIAD4107\tRIAD4073\tRIAD4301\tRIAD4302\tRIAD4340\n'
        "\tdesc\tdesc\tdesc\tdesc\tdesc\n"
        "37\t800\t250\t190\t40\t150\n"
    )
    zip_bytes = _make_zip(
        {
            "FFIEC CDR Call Schedule RCCI 12312021.txt": rcci,
            "FFIEC CDR Call Schedule RI 12312021.txt": ri,
        }
    )
    result, _ = capital_parse.parse_bank_capital_zip(zip_bytes, pd.Period("2021Q4", freq="Q"))
    assert result["RIAD4107"].iloc[0] == pytest.approx(800.0)
    assert result["RIAD4073"].iloc[0] == pytest.approx(250.0)
    assert result["RIAD4301"].iloc[0] == pytest.approx(190.0)
    assert result["RIAD4302"].iloc[0] == pytest.approx(40.0)
    assert result["RIAD4340"].iloc[0] == pytest.approx(150.0)


def test_parse_bank_capital_zip_reads_rcri_schedule_capital_ratio_items_with_percent_sign():
    rcci = '"IDRSSD"\tRCON1766\n\tdesc\n37\t100\n'
    rcri = (
        '"IDRSSD"\tRCOAP859\tRCOAP793\tRCOAA223\tRCOA7204\n'
        "\tdesc\tdesc\tdesc\tdesc\n"
        "37\t900\t13.82%\t6500\t9.69%\n"
    )
    zip_bytes = _make_zip(
        {
            "FFIEC CDR Call Schedule RCCI 12312021.txt": rcci,
            "FFIEC CDR Call Schedule RCRI 12312021.txt": rcri,
        }
    )
    result, _ = capital_parse.parse_bank_capital_zip(zip_bytes, pd.Period("2021Q4", freq="Q"))
    assert result["RCOAP859"].iloc[0] == pytest.approx(900.0)
    assert result["RCOAP793"].iloc[0] == pytest.approx(13.82)
    assert result["RCOAA223"].iloc[0] == pytest.approx(6500.0)
    assert result["RCOA7204"].iloc[0] == pytest.approx(9.69)


def test_parse_bank_capital_zip_default_ffiec_parse_behavior_is_unaffected():
    # The default (non-bank) parse_bulk_zip call should NOT pick up RCOA-prefixed RCRI items --
    # confirms the extension stays additive/opt-in for every other caller.
    from corefin.credit.sources.ffiec_parse import parse_bulk_zip

    rcci = '"IDRSSD"\tRCON1766\n\tdesc\n37\t100\n'
    rcri = '"IDRSSD"\tRCOAP793\n\tdesc\n37\t13.82%\n'
    zip_bytes = _make_zip(
        {
            "FFIEC CDR Call Schedule RCCI 12312021.txt": rcci,
            "FFIEC CDR Call Schedule RCRI 12312021.txt": rcri,
        }
    )
    result, _ = parse_bulk_zip(zip_bytes, pd.Period("2021Q4", freq="Q"))
    assert "RCOAP793" not in result.columns
