"""
Unit tests for SEC EDGAR client.
Uses mocked HTTP responses — never hits the live API.
"""

import json
import pytest
import responses as rsps_lib
from unittest.mock import patch, MagicMock

from tools.edgar_client import EdgarClient, Filing

# ── Fixtures ──────────────────────────────────────────────────────────────────

MOCK_TICKERS = {
    "0": {
        "cik_str": 320193,
        "ticker": "AAPL",
        "title": "Apple Inc."
    },
    "1": {
        "cik_str": 789019,
        "ticker": "MSFT",
        "title": "Microsoft Corp"
    },
}

MOCK_SUBMISSIONS = {
    "name": "Apple Inc.",
    "filings": {
        "recent": {
            "form": ["10-K", "10-Q", "8-K", "10-Q"],
            "filingDate": ["2024-11-01", "2024-08-02", "2024-07-15", "2024-05-03"],
            "accessionNumber": [
                "0000320193-24-000123",
                "0000320193-24-000089",
                "0000320193-24-000056",
                "0000320193-24-000034",
            ],
        }
    },
}


# ── Tests ─────────────────────────────────────────────────────────────────────

class TestEdgarClientCIK:
    """Test CIK resolution logic."""

    @rsps_lib.activate
    def test_get_cik_known_ticker(self):
        rsps_lib.add(
            rsps_lib.GET,
            "https://www.sec.gov/files/company_tickers.json",
            json=MOCK_TICKERS,
            status=200,
        )
        client = EdgarClient(user_agent="TestApp/1.0 test@example.com")
        cik = client.get_cik("AAPL")
        assert cik == "0000320193"

    @rsps_lib.activate
    def test_get_cik_zero_padded_to_10_digits(self):
        rsps_lib.add(
            rsps_lib.GET,
            "https://www.sec.gov/files/company_tickers.json",
            json=MOCK_TICKERS,
            status=200,
        )
        client = EdgarClient(user_agent="TestApp/1.0 test@example.com")
        cik = client.get_cik("MSFT")
        assert len(cik) == 10
        assert cik == "0000789019"

    @rsps_lib.activate
    def test_get_cik_case_insensitive(self):
        rsps_lib.add(
            rsps_lib.GET,
            "https://www.sec.gov/files/company_tickers.json",
            json=MOCK_TICKERS,
            status=200,
        )
        client = EdgarClient(user_agent="TestApp/1.0 test@example.com")
        cik = client.get_cik("aapl")
        assert cik == "0000320193"

    @rsps_lib.activate
    def test_get_cik_unknown_ticker_raises(self):
        rsps_lib.add(
            rsps_lib.GET,
            "https://www.sec.gov/files/company_tickers.json",
            json=MOCK_TICKERS,
            status=200,
        )
        client = EdgarClient(user_agent="TestApp/1.0 test@example.com")
        with pytest.raises(ValueError, match="not found in EDGAR"):
            client.get_cik("INVALID_XYZ")

    def test_missing_user_agent_raises(self):
        with pytest.raises(ValueError, match="descriptive User-Agent"):
            EdgarClient(user_agent="")

    def test_user_agent_without_email_raises(self):
        with pytest.raises(ValueError, match="contact email"):
            EdgarClient(user_agent="MyApp/1.0")


class TestEdgarClientFilings:
    """Test filing discovery."""

    @rsps_lib.activate
    def test_get_recent_filings_filters_form_type(self):
        rsps_lib.add(
            rsps_lib.GET,
            "https://www.sec.gov/files/company_tickers.json",
            json=MOCK_TICKERS,
            status=200,
        )
        rsps_lib.add(
            rsps_lib.GET,
            "https://data.sec.gov/submissions/CIK0000320193.json",
            json=MOCK_SUBMISSIONS,
            status=200,
        )
        client = EdgarClient(user_agent="TestApp/1.0 test@example.com")
        filings = client.get_recent_filings("AAPL", form_types=["10-K"], limit=5)

        assert len(filings) == 1
        assert all(f.form_type == "10-K" for f in filings)

    @rsps_lib.activate
    def test_get_recent_filings_respects_limit(self):
        rsps_lib.add(
            rsps_lib.GET,
            "https://www.sec.gov/files/company_tickers.json",
            json=MOCK_TICKERS,
            status=200,
        )
        rsps_lib.add(
            rsps_lib.GET,
            "https://data.sec.gov/submissions/CIK0000320193.json",
            json=MOCK_SUBMISSIONS,
            status=200,
        )
        client = EdgarClient(user_agent="TestApp/1.0 test@example.com")
        filings = client.get_recent_filings("AAPL", form_types=["10-Q"], limit=1)
        assert len(filings) == 1

    @rsps_lib.activate
    def test_filing_has_correct_cik_in_url(self):
        rsps_lib.add(
            rsps_lib.GET,
            "https://www.sec.gov/files/company_tickers.json",
            json=MOCK_TICKERS,
            status=200,
        )
        rsps_lib.add(
            rsps_lib.GET,
            "https://data.sec.gov/submissions/CIK0000320193.json",
            json=MOCK_SUBMISSIONS,
            status=200,
        )
        client = EdgarClient(user_agent="TestApp/1.0 test@example.com")
        filings = client.get_recent_filings("AAPL", form_types=["10-K"])
        assert len(filings) > 0
        assert "320193" in filings[0].document_url


class TestEdgarClientRateLimit:
    """Test rate limiting behavior."""

    def test_rate_limit_enforced(self):
        import time
        client = EdgarClient(user_agent="TestApp/1.0 test@example.com", request_delay=0.05)
        client._last_request_time = time.monotonic()
        start = time.monotonic()
        client._rate_limit()
        elapsed = time.monotonic() - start
        assert elapsed >= 0.04  # slight tolerance
