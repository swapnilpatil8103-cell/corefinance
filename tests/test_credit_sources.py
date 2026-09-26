"""Offline tests for the credit data source clients -- every network call
is mocked; no test here ever touches the network."""

import sys
from unittest import mock

import pandas as pd
import pytest

from corefin.credit.sources import fdic, fed_scenarios, ffiec, fred


def _fake_response(json_payload=None, text=None, status: int = 200):
    resp = mock.Mock()
    resp.status_code = status
    if json_payload is not None:
        resp.json.return_value = json_payload
    if text is not None:
        resp.text = text

    def _raise_for_status():
        if status >= 400:
            raise Exception(f"HTTP {status}")

    resp.raise_for_status.side_effect = _raise_for_status
    return resp


# ---------------------------------------------------------------- FRED ----


def test_fred_fetch_series_requires_api_key(monkeypatch):
    monkeypatch.delenv("FRED_API_KEY", raising=False)
    with pytest.raises(fred.FredApiKeyMissingError, match="FRED_API_KEY"):
        fred.fetch_series("UNRATE")


def test_fred_fetch_series_parses_observations_and_missing_marker():
    payload = {
        "observations": [
            {"date": "2020-01-01", "value": "3.5"},
            {"date": "2020-02-01", "value": "."},  # FRED's missing-data marker
            {"date": "2020-03-01", "value": "4.4"},
        ]
    }
    with mock.patch.object(fred.requests, "get", return_value=_fake_response(payload)) as get:
        df = fred.fetch_series("UNRATE", api_key="test-key")

    assert df["value"].iloc[0] == 3.5
    assert pd.isna(df["value"].iloc[1])
    assert df["value"].iloc[2] == 4.4
    assert df["date"].iloc[0] == pd.Timestamp("2020-01-01")
    call_kwargs = get.call_args.kwargs
    assert call_kwargs["params"]["series_id"] == "UNRATE"
    assert call_kwargs["params"]["api_key"] == "test-key"


def test_fred_fetch_series_uses_env_var_when_no_explicit_key(monkeypatch):
    monkeypatch.setenv("FRED_API_KEY", "env-key")
    payload = {"observations": []}
    with mock.patch.object(fred.requests, "get", return_value=_fake_response(payload)) as get:
        fred.fetch_series("UNRATE")
    assert get.call_args.kwargs["params"]["api_key"] == "env-key"


def test_fred_series_mapping_covers_every_documented_fed_variable():
    expected_variables = {
        "Real GDP growth",
        "Nominal GDP growth",
        "Real disposable income growth",
        "Nominal disposable income growth",
        "Unemployment rate",
        "CPI inflation rate",
        "3-month Treasury rate",
        "5-year Treasury yield",
        "10-year Treasury yield",
        "BBB corporate yield",
        "Mortgage rate",
        "Prime rate",
        "Dow Jones Total Stock Market Index",
        "House Price Index",
        "Commercial Real Estate Price Index",
        "Market Volatility Index",
    }
    assert expected_variables == set(fred.FED_SCENARIO_VARIABLE_TO_FRED)


def test_fred_series_mapping_flags_imperfect_matches_with_a_note():
    for variable, mapping in fred.FED_SCENARIO_VARIABLE_TO_FRED.items():
        if not mapping.is_exact_match:
            assert mapping.note, f"{variable} is flagged as inexact but has no explanatory note"


# ---------------------------------------------------------------- FDIC ----


def test_fdic_fetch_institutions_flattens_data_envelope():
    payload = {"data": [{"data": {"CERT": 3510, "NAME": "Test Bank"}, "score": 0}]}
    with mock.patch.object(fdic.requests, "get", return_value=_fake_response(payload)) as get:
        df = fdic.fetch_institutions("CERT:3510")
    assert list(df["CERT"]) == [3510]
    assert get.call_args.args[0] == f"{fdic.FDIC_BASE_URL}/institutions"


def test_fdic_fetch_financials_parses_report_date():
    payload = {
        "data": [
            {"data": {"CERT": 3510, "REPDTE": "20260630", "ASSET": 100, "DEP": 80, "NETINC": 5}}
        ]
    }
    with mock.patch.object(fdic.requests, "get", return_value=_fake_response(payload)) as get:
        df = fdic.fetch_financials(3510)
    assert df["REPDTE"].iloc[0] == pd.Timestamp("2026-06-30")
    assert get.call_args.kwargs["params"]["filters"] == "CERT:3510"


def test_fdic_fetch_financials_empty_result_does_not_crash():
    with mock.patch.object(fdic.requests, "get", return_value=_fake_response({"data": []})):
        df = fdic.fetch_financials(999999999)
    assert df.empty


# --------------------------------------------------------- Fed scenarios --


def test_fed_scenario_rejects_unknown_scenario_name():
    with pytest.raises(ValueError, match="baseline"):
        fed_scenarios.fetch_scenario("bogus")


def test_fed_scenario_builds_the_correct_url_and_parses_csv():
    csv_text = "Scenario Name,Date,Unemployment rate\nSupervisory Baseline,2025 Q1,4.3\n"
    with mock.patch.object(
        fed_scenarios.requests, "get", return_value=_fake_response(text=csv_text)
    ) as get:
        df = fed_scenarios.fetch_scenario("baseline", vintage=2025)
    assert get.call_args.args[0] == (
        "https://www.federalreserve.gov/supervisionreg/files/"
        "2025-Table_2A_Supervisory_Baseline_Domestic.csv"
    )
    assert df["Unemployment rate"].iloc[0] == 4.3


def test_fed_scenario_severely_adverse_uses_table_3a():
    csv_text = "Scenario Name,Date\nSupervisory Severely Adverse,2025 Q1\n"
    with mock.patch.object(
        fed_scenarios.requests, "get", return_value=_fake_response(text=csv_text)
    ) as get:
        fed_scenarios.fetch_scenario("severely_adverse", vintage=2025)
    assert "3A_Supervisory_Severely_Adverse_Domestic" in get.call_args.args[0]


# --------------------------------------------------------------- FFIEC ----

_FFIEC_INITIAL_HTML = """
<input type="hidden" id="__VIEWSTATE" value="initial-vs" />
<input type="hidden" id="__VIEWSTATEGENERATOR" value="initial-vg" />
"""

_FFIEC_PRODUCT_SELECTED_HTML = """
<input type="hidden" id="__VIEWSTATE" value="selected-vs" />
<input type="hidden" id="__VIEWSTATEGENERATOR" value="selected-vg" />
<select name="ctl00$MainContentHolder$DatesDropDownList" id="DatesDropDownList" class="valuelabel">
\t<option selected="selected" value="152">06/30/2026</option>
\t<option value="151">03/31/2026</option>
\t<option value="130">12/31/2021</option>
</select>
"""


def _ffiec_fake_html_response(html: str) -> mock.Mock:
    resp = mock.Mock()
    resp.text = html
    resp.raise_for_status.side_effect = lambda: None
    return resp


def _ffiec_fake_zip_response(content: bytes = b"PK\x03\x04fakezipbytes") -> mock.Mock:
    resp = mock.Mock()
    resp.content = content
    disposition = 'attachment; filename="FFIEC CDR Call Bulk All Schedules 12312021.zip"'
    resp.headers = {"Content-Type": "application/octet-stream", "Content-Disposition": disposition}
    resp.raise_for_status.side_effect = lambda: None
    return resp


def _mock_ffiec_session(post_side_effect):
    session = mock.Mock()
    session.get.return_value = _ffiec_fake_html_response(_FFIEC_INITIAL_HTML)
    session.post.side_effect = post_side_effect
    return session


def test_fetch_bulk_call_report_zip_full_flow():
    zip_response = _ffiec_fake_zip_response()
    session = _mock_ffiec_session(
        [_ffiec_fake_html_response(_FFIEC_PRODUCT_SELECTED_HTML), zip_response]
    )
    with mock.patch.object(ffiec.requests, "Session", return_value=session):
        result = ffiec.fetch_bulk_call_report_zip("12/31/2021")

    assert result == zip_response.content
    select_call, download_call = session.post.call_args_list
    assert select_call.kwargs["data"]["__EVENTTARGET"] == "ctl00$MainContentHolder$ListBox1"
    assert select_call.kwargs["data"]["__VIEWSTATE"] == "initial-vs"
    download_data = download_call.kwargs["data"]
    assert download_data["__VIEWSTATE"] == "selected-vs"
    assert download_data["ctl00$MainContentHolder$DatesDropDownList"] == "130"
    assert download_data["ctl00$MainContentHolder$FormatType"] == "TSVRadioButton"


def test_fetch_bulk_call_report_zip_raises_for_unavailable_period():
    session = _mock_ffiec_session([_ffiec_fake_html_response(_FFIEC_PRODUCT_SELECTED_HTML)])
    with mock.patch.object(ffiec.requests, "Session", return_value=session):
        with pytest.raises(ffiec.FfiecPeriodNotFoundError, match="12/31/1999"):
            ffiec.fetch_bulk_call_report_zip("12/31/1999")


def test_fetch_bulk_call_report_zip_raises_when_response_is_not_a_zip():
    bad_response = mock.Mock()
    bad_response.headers = {"Content-Type": "text/html; charset=utf-8", "Content-Disposition": ""}
    bad_response.raise_for_status.side_effect = lambda: None
    session = _mock_ffiec_session(
        [_ffiec_fake_html_response(_FFIEC_PRODUCT_SELECTED_HTML), bad_response]
    )
    with mock.patch.object(ffiec.requests, "Session", return_value=session):
        with pytest.raises(ffiec.FfiecBulkDownloadError):
            ffiec.fetch_bulk_call_report_zip("12/31/2021")


def test_list_available_periods_returns_sorted_oldest_to_newest():
    session = _mock_ffiec_session([_ffiec_fake_html_response(_FFIEC_PRODUCT_SELECTED_HTML)])
    with mock.patch.object(ffiec.requests, "Session", return_value=session):
        periods = ffiec.list_available_periods()
    assert periods == ["12/31/2021", "03/31/2026", "06/30/2026"]


# ------------------------------------------------- FFIEC REST adapter -----


def test_ffiec_rest_requires_both_credentials(monkeypatch):
    monkeypatch.delenv("FFIEC_USERNAME", raising=False)
    monkeypatch.delenv("FFIEC_BEARER_TOKEN", raising=False)
    with pytest.raises(ffiec.FfiecRestCredentialsMissingError, match="FFIEC_USERNAME"):
        ffiec.fetch_single_bank_via_rest("480228", "12/31/2023")


def test_ffiec_rest_requires_both_not_just_one_credential(monkeypatch):
    monkeypatch.setenv("FFIEC_USERNAME", "someuser")
    monkeypatch.delenv("FFIEC_BEARER_TOKEN", raising=False)
    with pytest.raises(ffiec.FfiecRestCredentialsMissingError):
        ffiec.fetch_single_bank_via_rest("480228", "12/31/2023")


def test_ffiec_rest_raises_import_error_when_package_not_installed(monkeypatch):
    monkeypatch.setenv("FFIEC_USERNAME", "someuser")
    monkeypatch.setenv("FFIEC_BEARER_TOKEN", "sometoken")
    with pytest.raises(ImportError, match="ffiec-data-connect"):
        ffiec.fetch_single_bank_via_rest("480228", "12/31/2023")


def test_ffiec_rest_calls_collect_data_with_expected_arguments(monkeypatch):
    monkeypatch.setenv("FFIEC_USERNAME", "someuser")
    monkeypatch.setenv("FFIEC_BEARER_TOKEN", "sometoken")

    fake_module = mock.Mock()
    fake_module.OAuth2Credentials = mock.Mock(return_value="creds-object")
    fake_module.collect_data = mock.Mock(return_value="fake-dataframe")
    monkeypatch.setitem(sys.modules, "ffiec_data_connect", fake_module)

    result = ffiec.fetch_single_bank_via_rest("480228", "12/31/2023")

    assert result == "fake-dataframe"
    fake_module.OAuth2Credentials.assert_called_once_with(
        username="someuser", bearer_token="sometoken"
    )
    fake_module.collect_data.assert_called_once_with(
        "creds-object",
        reporting_period="12/31/2023",
        rssd_id="480228",
        series="call",
        output_type="pandas",
    )
