"""FFIEC Central Data Repository Public Data Distribution Web Service (PWS)
client -- Call Report bulk data (schedules RC-C loans, RI-B charge-offs
and recoveries, RC-N past due and nonaccrual).

STATUS: stubbed, pending FFIEC PWS account credentials. The PWS requires a
free account (https://cdr.ffiec.gov/public/PWS/PWSPage.aspx); FFIEC emails
a username plus setup instructions (SOAP endpoint, WSDL, sample code)
after registration, and none of that is visible without an account -- so
the actual call is implemented once those instructions are in hand, not
guessed at from documentation that couldn't be reached.

Reads FFIEC_PWS_USERNAME / FFIEC_PWS_PASSWORD from the environment, same
convention as FRED_API_KEY: never read from a file, never logged.

A no-account fallback does exist and was verified working: FFIEC's public
bulk-download page (https://cdr.ffiec.gov/public/PWS/DownloadBulkData.aspx)
is an ASP.NET postback form covering quarterly Call Report data back to
2001 Q1 through the current quarter, scriptable with a plain HTTP session
(GET the page, parse the hidden __VIEWSTATE/__EVENTVALIDATION fields, POST
the product/period/format selection). That path was deliberately not
built here, since the PWS API was the explicit choice over it.
"""

from __future__ import annotations

import os


class FfiecCredentialsMissingError(RuntimeError):
    pass


def _credentials() -> tuple[str, str]:
    username = os.environ.get("FFIEC_PWS_USERNAME")
    password = os.environ.get("FFIEC_PWS_PASSWORD")
    if not username or not password:
        raise FfiecCredentialsMissingError(
            "FFIEC_PWS_USERNAME and FFIEC_PWS_PASSWORD environment variables must both be "
            "set. Create a free account at https://cdr.ffiec.gov/public/PWS/PWSPage.aspx, "
            "then export both -- never write them to a config file or commit them."
        )
    return username, password


def fetch_call_report_bulk_data(
    product: str, reporting_period_end_date: str, file_format: str = "TabDelimited"
) -> bytes:
    """Not yet implemented. Checks credentials first (raises
    FfiecCredentialsMissingError if either env var is unset), then raises
    NotImplementedError -- the SOAP call itself needs the WSDL/endpoint
    details from the PWS account setup email before it can be built
    against the real API rather than guessed."""
    _credentials()
    raise NotImplementedError(
        "FFIEC PWS SOAP client not yet implemented -- provide the WSDL/endpoint details "
        "from your PWS account setup email so this can be built against the real API."
    )
