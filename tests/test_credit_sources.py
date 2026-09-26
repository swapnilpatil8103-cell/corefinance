"""Offline tests for the credit data source clients -- every network call
is mocked; no test here ever touches the network."""

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


def test_ffiec_requires_both_credentials(monkeypatch):
    monkeypatch.delenv("FFIEC_PWS_USERNAME", raising=False)
    monkeypatch.delenv("FFIEC_PWS_PASSWORD", raising=False)
    with pytest.raises(ffiec.FfiecCredentialsMissingError, match="FFIEC_PWS"):
        ffiec.fetch_call_report_bulk_data("ReportingSeriesSinglePeriod", "06/30/2026")


def test_ffiec_requires_both_not_just_one_credential(monkeypatch):
    monkeypatch.setenv("FFIEC_PWS_USERNAME", "someuser")
    monkeypatch.delenv("FFIEC_PWS_PASSWORD", raising=False)
    with pytest.raises(ffiec.FfiecCredentialsMissingError):
        ffiec.fetch_call_report_bulk_data("ReportingSeriesSinglePeriod", "06/30/2026")


def test_ffiec_raises_not_implemented_once_credentials_are_present(monkeypatch):
    monkeypatch.setenv("FFIEC_PWS_USERNAME", "someuser")
    monkeypatch.setenv("FFIEC_PWS_PASSWORD", "somepass")
    with pytest.raises(NotImplementedError):
        ffiec.fetch_call_report_bulk_data("ReportingSeriesSinglePeriod", "06/30/2026")
