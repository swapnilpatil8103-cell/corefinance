"""FFIEC Call Report data clients.

STATUS: FFIEC shut down its SOAP-based Public Data Distribution Web
Service (PWS) on 2026-02-28; only its REST API (OAuth2 bearer tokens)
remains. This module's PRIMARY path is therefore the no-account bulk
Call Report download instead of any FFIEC account-gated API:
https://cdr.ffiec.gov/public/PWS/DownloadBulkData.aspx, a stateful
ASP.NET WebForms page. It was reverse-engineered and verified end-to-end
against live data (downloading real 2021Q4 and 2026Q2 "All Schedules" TSV
bulk ZIPs, both landing with FFIEC-issued filenames matching the
requested period and containing every MDRM item code schema.py depends
on) -- no account, no credentials, and it covers quarterly Call Report
data back to 2001Q1 (102 quarters as of this writing).

Verified flow:
1. GET the page in a `requests.Session` (the session cookies --
   ASP.NET_SessionId, ApplicationGatewayAffinity* -- and the initial
   __VIEWSTATE/__VIEWSTATEGENERATOR are required by every later step;
   the whole flow must reuse one session, not independent requests).
2. POST __EVENTTARGET=ctl00$MainContentHolder$ListBox1 (selecting the
   product) with that __VIEWSTATE/__VIEWSTATEGENERATOR. The response's
   HTML now has ctl00$MainContentHolder$DatesDropDownList populated with
   <option value="NNN">MM/DD/YYYY</option> entries -- an internal numeric
   ID per reporting period that is NOT predictable from the date (IDs
   are not chronological -- e.g. 2001's four quarters are 20/21/22/23
   while 2002's are 2/3/4/5/6/7), so they must always be parsed from this
   response, never guessed or hardcoded.
3. POST again in the SAME session, with this response's (new)
   __VIEWSTATE/__VIEWSTATEGENERATOR, the product, the chosen numeric
   DatesDropDownList value, FormatType=TSVRadioButton, and the Download
   button's own name=value pair. The response IS the ZIP file
   (Content-Type: application/octet-stream, Content-Disposition:
   attachment; filename="FFIEC CDR Call Bulk All Schedules MMDDYYYY.zip").

Caching the raw ZIPs (a gitignored data/raw/ffiec/ directory) is the
caller's job -- see the CLI, not this module.

An OPTIONAL REST adapter is also provided via the `ffiec-data-connect`
PyPI package (FFIEC's own recommended replacement for the retired SOAP
PWS, confirmed against its published README), useful for a single-bank/
single-period pull rather than the full industry bulk file. It needs
FFIEC_USERNAME and FFIEC_BEARER_TOKEN environment variables (a 90-day
JWT bearer token, never a password; never written to a file, never
logged) and lazily imports `ffiec_data_connect`, which is NOT a hard
dependency of corefin[fig] -- install it yourself
(`pip install ffiec-data-connect`) to use this path. The bulk-download
client above is this project's primary source and needs neither an
account nor this package.
"""

from __future__ import annotations

import os
import re

import requests

BULK_DOWNLOAD_URL = "https://cdr.ffiec.gov/public/PWS/DownloadBulkData.aspx"

PRODUCT_SINGLE_PERIOD = "ReportingSeriesSinglePeriod"

_VIEWSTATE_RE = re.compile(r'id="__VIEWSTATE" value="([^"]*)"')
_VIEWSTATEGENERATOR_RE = re.compile(r'id="__VIEWSTATEGENERATOR" value="([^"]*)"')
_DATES_SELECT_RE = re.compile(
    r'name="ctl00\$MainContentHolder\$DatesDropDownList".*?</select>', re.S
)
_DATES_OPTION_RE = re.compile(
    r'<option\s+(?:selected="selected"\s+)?value="(\d+)">([^<]+)</option>'
)

# cdr.ffiec.gov's WAF (Application Gateway) 403s the default python-requests
# User-Agent outright -- verified live: identical requests succeed with a
# browser-like User-Agent and fail with requests' default. Every session
# must set one.
_BROWSER_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)


def _new_session() -> requests.Session:
    session = requests.Session()
    session.headers.update({"User-Agent": _BROWSER_USER_AGENT})
    return session


class FfiecPeriodNotFoundError(ValueError):
    pass


class FfiecBulkDownloadError(RuntimeError):
    pass


def _extract_viewstate(html: str) -> tuple[str, str]:
    viewstate = _VIEWSTATE_RE.search(html)
    generator = _VIEWSTATEGENERATOR_RE.search(html)
    if not viewstate or not generator:
        raise FfiecBulkDownloadError(
            "could not find __VIEWSTATE/__VIEWSTATEGENERATOR in the FFIEC bulk-download "
            "page response -- the site's form may have changed."
        )
    return viewstate.group(1), generator.group(1)


def _extract_period_options(html: str) -> dict[str, str]:
    """Returns {"MM/DD/YYYY" quarter-end date: numeric dropdown value}."""
    block = _DATES_SELECT_RE.search(html)
    if not block:
        raise FfiecBulkDownloadError(
            "DatesDropDownList not found in the FFIEC bulk-download page response -- "
            "the site's form may have changed."
        )
    return {label.strip(): value for value, label in _DATES_OPTION_RE.findall(block.group(0))}


def _select_product(session: requests.Session, product: str) -> tuple[str, str, dict[str, str]]:
    """Runs steps 1-2 of the verified flow. Returns (viewstate, viewstategenerator,
    {period_date: dropdown_value})."""
    initial = session.get(BULK_DOWNLOAD_URL, timeout=60)
    initial.raise_for_status()
    viewstate, generator = _extract_viewstate(initial.text)

    selected = session.post(
        BULK_DOWNLOAD_URL,
        data={
            "__EVENTTARGET": "ctl00$MainContentHolder$ListBox1",
            "__EVENTARGUMENT": "",
            "__VIEWSTATE": viewstate,
            "__VIEWSTATEGENERATOR": generator,
            "ctl00$MainContentHolder$ListBox1": product,
        },
        timeout=60,
    )
    selected.raise_for_status()
    viewstate, generator = _extract_viewstate(selected.text)
    return viewstate, generator, _extract_period_options(selected.text)


def list_available_periods(product: str = PRODUCT_SINGLE_PERIOD) -> list[str]:
    """Every available "MM/DD/YYYY" reporting-period end date for `product`,
    as currently offered by the FFIEC bulk-download page (sorted oldest
    to newest)."""
    session = _new_session()
    _, _, period_options = _select_product(session, product)

    def _sort_key(date_str: str) -> tuple[int, int, int]:
        month, day, year = date_str.split("/")
        return int(year), int(month), int(day)

    return sorted(period_options, key=_sort_key)


def fetch_bulk_call_report_zip(
    reporting_period_end_date: str, product: str = PRODUCT_SINGLE_PERIOD
) -> bytes:
    """reporting_period_end_date: "MM/DD/YYYY" quarter-end date (e.g.
    "12/31/2021"), matching the FFIEC site's own display format -- see
    `list_available_periods` for the exact set of valid values. Returns
    the raw ZIP bytes of the bulk "All Schedules" Call Report download for
    that period. Raises FfiecPeriodNotFoundError if the period isn't
    currently offered, FfiecBulkDownloadError if the site's response
    doesn't look like the expected ZIP (the form having changed)."""
    session = _new_session()
    viewstate, generator, period_options = _select_product(session, product)

    period_value = period_options.get(reporting_period_end_date)
    if period_value is None:
        raise FfiecPeriodNotFoundError(
            f"{reporting_period_end_date!r} is not an available reporting period for "
            f"product {product!r}. Available: {sorted(period_options)}"
        )

    download = session.post(
        BULK_DOWNLOAD_URL,
        data={
            "__VIEWSTATE": viewstate,
            "__VIEWSTATEGENERATOR": generator,
            "ctl00$MainContentHolder$ListBox1": product,
            "ctl00$MainContentHolder$DatesDropDownList": period_value,
            "ctl00$MainContentHolder$FormatType": "TSVRadioButton",
            "ctl00$MainContentHolder$TabStrip1$Download_0": "Download",
        },
        timeout=120,
    )
    download.raise_for_status()
    content_type = download.headers.get("Content-Type", "")
    disposition = download.headers.get("Content-Disposition", "")
    if "octet-stream" not in content_type or "attachment" not in disposition:
        raise FfiecBulkDownloadError(
            "FFIEC bulk-download POST did not return a file attachment -- the site's "
            f"form may have changed. Content-Type: {content_type!r}, "
            f"Content-Disposition: {disposition!r}"
        )
    return download.content


# ------------------------------------------------ optional REST adapter ----


class FfiecRestCredentialsMissingError(RuntimeError):
    pass


def _rest_credentials() -> tuple[str, str]:
    username = os.environ.get("FFIEC_USERNAME")
    bearer_token = os.environ.get("FFIEC_BEARER_TOKEN")
    if not username or not bearer_token:
        raise FfiecRestCredentialsMissingError(
            "FFIEC_USERNAME and FFIEC_BEARER_TOKEN environment variables must both be "
            "set to use the REST adapter (FFIEC_BEARER_TOKEN is a 90-day JWT token "
            "from your FFIEC CDR account, not a password) -- never write them to a "
            "config file or commit them. This adapter is optional; "
            "fetch_bulk_call_report_zip above needs no account or extra package."
        )
    return username, bearer_token


def fetch_single_bank_via_rest(
    rssd_id: str, reporting_period_end_date: str, output_type: str = "pandas"
):
    """Single-bank/single-period pull via the optional `ffiec-data-connect`
    package's REST adapter. `reporting_period_end_date`: "MM/DD/YYYY".
    Lazily imports `ffiec_data_connect` (not a hard dependency of
    corefin[fig]) -- raises ImportError with install instructions if it
    isn't installed."""
    username, bearer_token = _rest_credentials()
    try:
        from ffiec_data_connect import OAuth2Credentials, collect_data
    except ImportError as exc:
        raise ImportError(
            "the optional 'ffiec-data-connect' package is not installed -- run "
            "`pip install ffiec-data-connect` to use fetch_single_bank_via_rest, or use "
            "fetch_bulk_call_report_zip instead (no account or extra package needed)."
        ) from exc

    credentials = OAuth2Credentials(username=username, bearer_token=bearer_token)
    return collect_data(
        credentials,
        reporting_period=reporting_period_end_date,
        rssd_id=rssd_id,
        series="call",
        output_type=output_type,
    )
