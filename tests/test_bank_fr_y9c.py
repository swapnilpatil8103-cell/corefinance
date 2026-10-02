"""Offline tests for corefin.bank.sources.fr_y9c -- builds a tiny
synthetic caret-separated FR Y-9C bulk file in memory (confirmed against
a real downloaded BHCF20251231.ZIP: single header row, no per-column
description row unlike the Call Report's schedule files); no network."""

import zipfile
from io import BytesIO

import pandas as pd
import pytest

from corefin.bank.sources import fr_y9c


def _make_zip(name: str, content: str) -> bytes:
    buf = BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr(name, content)
    return buf.getvalue()


def test_parse_y9c_bulk_zip_reads_caret_separated_items():
    content = (
        "RSSD9001^RSSD9017^BHCK2170^BHCAP793\n"
        "9000001^FIRST FICTIONAL BANCORP, INC.^20842331^11.5390\n"
        "9000002^SAMPLE COMMUNITY BANKSHARES, INC.^4385765^13.2000\n"
    )
    zip_bytes = _make_zip("BHCF20251231.txt", content)
    result = fr_y9c.parse_y9c_bulk_zip(zip_bytes, pd.Period("2025Q4", freq="Q"))

    assert len(result) == 2
    assert set(result["rssd_id"]) == {"9000001", "9000002"}
    row = result[result["rssd_id"] == "9000001"].iloc[0]
    assert row["BHCK2170"] == pytest.approx(20842331.0)
    assert row["BHCAP793"] == pytest.approx(11.5390)
    assert row["quarter"] == pd.Period("2025Q4", freq="Q")


def test_parse_y9c_bulk_zip_coerces_non_numeric_cells_to_nan():
    content = "RSSD9001^BHCK2170\n9000001^\n9000002^4385765\n"
    zip_bytes = _make_zip("BHCF20251231.txt", content)
    result = fr_y9c.parse_y9c_bulk_zip(zip_bytes, pd.Period("2025Q4", freq="Q"))
    assert result.loc[result["rssd_id"] == "9000001", "BHCK2170"].isna().iloc[0]


def test_parse_y9c_bulk_zip_raises_when_zip_has_more_than_one_file():
    buf = BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("a.txt", "RSSD9001\n1\n")
        zf.writestr("b.txt", "RSSD9001\n2\n")
    with pytest.raises(ValueError, match="exactly one file"):
        fr_y9c.parse_y9c_bulk_zip(buf.getvalue(), pd.Period("2025Q4", freq="Q"))


def test_parse_y9c_bulk_zip_raises_when_rssd_column_missing():
    zip_bytes = _make_zip("a.txt", "SOME_OTHER_COL\n1\n")
    with pytest.raises(ValueError, match="RSSD9001"):
        fr_y9c.parse_y9c_bulk_zip(zip_bytes, pd.Period("2025Q4", freq="Q"))


def test_find_holding_company_by_name_is_case_insensitive():
    content = (
        "RSSD9001^RSSD9017^BHCK2170\n"
        "9000001^FIRST FICTIONAL BANCORP, INC.^20842331\n"
        "9000002^SAMPLE COMMUNITY BANKSHARES, INC.^4385765\n"
    )
    zip_bytes = _make_zip("BHCF20251231.txt", content)
    result = fr_y9c.parse_y9c_bulk_zip(zip_bytes, pd.Period("2025Q4", freq="Q"))
    matches = fr_y9c.find_holding_company_by_name(result, "fictional")
    assert matches["rssd_id"].tolist() == ["9000001"]
