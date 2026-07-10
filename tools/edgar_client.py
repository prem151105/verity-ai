"""
SEC EDGAR REST API Client
- No API key required.
- Respects SEC fair-access policy: descriptive User-Agent + request delays.
- Docs: https://www.sec.gov/edgar/sec-api-documentation
"""

import time
import json
import logging
from typing import Optional
from dataclasses import dataclass

import requests

logger = logging.getLogger(__name__)

# Base URLs
EDGAR_BASE = "https://data.sec.gov"
COMPANY_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"   # moved endpoint
SUBMISSIONS_URL = f"{EDGAR_BASE}/submissions/CIK{{cik}}.json"
COMPANY_FACTS_URL = f"{EDGAR_BASE}/api/xbrl/companyfacts/CIK{{cik}}.json"
EFTS_SEARCH_URL = "https://efts.sec.gov/LATEST/search-index?q={query}&dateRange=custom&startdt={start}&enddt={end}&forms={forms}"


@dataclass
class Filing:
    cik: str
    company_name: str
    form_type: str
    filed_date: str
    accession_number: str
    document_url: str
    text_content: Optional[str] = None


class EdgarClient:
    """
    Client for SEC EDGAR REST API.
    Thread-safe for sequential use; add asyncio wrapper for concurrent requests.
    """

    def __init__(self, user_agent: str, request_delay: float = 0.1):
        if not user_agent or "@" not in user_agent:
            raise ValueError(
                "SEC requires a descriptive User-Agent with contact email. "
                "Example: 'MyApp/1.0 myname@example.com'"
            )
        self.user_agent = user_agent
        self.request_delay = request_delay
        self._session = requests.Session()
        self._session.headers.update(
            {
                "User-Agent": self.user_agent,
                "Accept": "application/json",
            }
        )
        self._last_request_time = 0.0
        # Cache the ticker → CIK mapping (loaded once per session)
        self._ticker_to_cik: dict[str, str] = {}

    def _rate_limit(self):
        """Enforce minimum delay between requests per SEC fair-access policy."""
        elapsed = time.monotonic() - self._last_request_time
        if elapsed < self.request_delay:
            time.sleep(self.request_delay - elapsed)
        self._last_request_time = time.monotonic()

    def _get(self, url: str, params: dict | None = None) -> dict | list:
        """Execute a rate-limited GET request and return parsed JSON."""
        self._rate_limit()
        resp = self._session.get(url, params=params, timeout=30)
        resp.raise_for_status()
        return resp.json()

    def _get_text(self, url: str) -> str:
        """Fetch raw text content (for filing documents)."""
        self._rate_limit()
        resp = self._session.get(url, timeout=60)
        resp.raise_for_status()
        return resp.text

    # ──────────────────────────────────────────────────────────────────────────
    # CIK Resolution
    # ──────────────────────────────────────────────────────────────────────────

    def _load_ticker_map(self) -> None:
        """Load the full ticker → CIK mapping from EDGAR (cached in session)."""
        if self._ticker_to_cik:
            return
        logger.info("Loading EDGAR company tickers map...")
        data = self._get(COMPANY_TICKERS_URL)
        for entry in data.values():
            ticker = entry.get("ticker", "").upper()
            cik = str(entry.get("cik_str", "")).zfill(10)
            if ticker:
                self._ticker_to_cik[ticker] = cik
        logger.info(f"Loaded {len(self._ticker_to_cik)} tickers from EDGAR.")

    def get_cik(self, ticker: str) -> str:
        """
        Map a stock ticker to a zero-padded 10-digit CIK string.
        Raises ValueError if ticker not found.
        """
        self._load_ticker_map()
        ticker = ticker.upper().strip()
        cik = self._ticker_to_cik.get(ticker)
        if not cik:
            raise ValueError(f"Ticker '{ticker}' not found in EDGAR company list.")
        return cik

    # ──────────────────────────────────────────────────────────────────────────
    # Filing Discovery
    # ──────────────────────────────────────────────────────────────────────────

    def get_recent_filings(
        self,
        ticker: str,
        form_types: list[str] | None = None,
        limit: int = 5,
    ) -> list[Filing]:
        """
        Fetch recent filings for a ticker, optionally filtered by form type.

        Args:
            ticker: Stock ticker symbol (e.g. "AAPL")
            form_types: List of form types to filter (e.g. ["10-K", "10-Q"])
            limit: Max number of filings to return per form type
        Returns:
            List of Filing dataclass instances
        """
        if form_types is None:
            form_types = ["10-K", "10-Q"]

        cik = self.get_cik(ticker)
        url = SUBMISSIONS_URL.format(cik=cik)
        logger.info(f"Fetching submissions for {ticker} (CIK: {cik})")
        data = self._get(url)

        company_name = data.get("name", ticker)
        recent = data.get("filings", {}).get("recent", {})

        forms = recent.get("form", [])
        dates = recent.get("filingDate", [])
        accessions = recent.get("accessionNumber", [])

        results: list[Filing] = []
        counts: dict[str, int] = {}

        for form, date, accession in zip(forms, dates, accessions):
            if form not in form_types:
                continue
            if counts.get(form, 0) >= limit:
                continue

            # Build the filing index URL
            acc_clean = accession.replace("-", "")
            doc_url = (
                f"https://www.sec.gov/Archives/edgar/data/"
                f"{int(cik)}/{acc_clean}/index.json"
            )

            results.append(
                Filing(
                    cik=cik,
                    company_name=company_name,
                    form_type=form,
                    filed_date=date,
                    accession_number=accession,
                    document_url=doc_url,
                )
            )
            counts[form] = counts.get(form, 0) + 1

        logger.info(
            f"Found {len(results)} filings for {ticker}: "
            + ", ".join(f"{k}: {v}" for k, v in counts.items())
        )
        return results

    # ──────────────────────────────────────────────────────────────────────────
    # Document Fetching
    # ──────────────────────────────────────────────────────────────────────────

    def get_filing_document_text(self, filing: Filing) -> str:
        """
        Fetch the primary document text for a filing.
        Tries to find the main .htm/.html document from the filing index.
        Falls back to the full filing .txt if no HTML found.
        Returns plain text (HTML stripped).
        """
        try:
            index_data = self._get(filing.document_url)
        except Exception as e:
            logger.warning(f"Could not load filing index: {e}")
            return ""

        documents = index_data.get("directory", {}).get("item", [])
        if not documents:
            return ""

        # Prefer the primary document (first .htm file that's not XBRL/stylesheet)
        primary_url = None
        acc_clean = filing.accession_number.replace("-", "")
        base = (
            f"https://www.sec.gov/Archives/edgar/data/"
            f"{int(filing.cik)}/{acc_clean}/"
        )

        for doc in documents:
            name = doc.get("name", "")
            doc_type = doc.get("type", "")
            if doc_type in ("EX-", "XML", "XBRL", "R", "JSON"):
                continue
            if name.endswith((".htm", ".html")) and not name.startswith("R"):
                primary_url = base + name
                break

        if not primary_url and documents:
            primary_url = base + documents[0].get("name", "")

        if not primary_url:
            return ""

        logger.info(f"Fetching document: {primary_url}")
        try:
            raw = self._get_text(primary_url)
            return self._strip_html(raw)
        except Exception as e:
            logger.warning(f"Failed to fetch filing document: {e}")
            return ""

    @staticmethod
    def _strip_html(html: str) -> str:
        """Strip HTML tags and normalize whitespace."""
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(html, "lxml")
        # Remove script/style elements
        for tag in soup(["script", "style", "head"]):
            tag.decompose()
        text = soup.get_text(separator="\n")
        # Collapse blank lines
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        return "\n".join(lines)

    # ──────────────────────────────────────────────────────────────────────────
    # Structured Financial Data (XBRL)
    # ──────────────────────────────────────────────────────────────────────────

    def get_company_facts(self, ticker: str) -> dict:
        """
        Fetch all XBRL-tagged financial facts for a company.
        Returns raw facts dict (keyed by taxonomy → concept → unit → filings).
        Use extract_key_metrics() to get a clean summary.
        """
        cik = self.get_cik(ticker)
        url = COMPANY_FACTS_URL.format(cik=cik)
        logger.info(f"Fetching XBRL facts for {ticker}")
        return self._get(url)

    def extract_key_metrics(self, ticker: str) -> dict[str, list[dict]]:
        """
        Extract the most important financial metrics from XBRL facts.

        Returns dict mapping metric_name → list of {value, unit, filed, form, period}.
        Sorted descending by filing date.
        """
        facts = self.get_company_facts(ticker)
        us_gaap = facts.get("facts", {}).get("us-gaap", {})

        # Key metrics to extract (XBRL concept names)
        TARGET_CONCEPTS = {
            "Revenues": "revenue",
            "RevenueFromContractWithCustomerExcludingAssessedTax": "revenue",
            "NetIncomeLoss": "net_income",
            "GrossProfit": "gross_profit",
            "OperatingIncomeLoss": "operating_income",
            "EarningsPerShareBasic": "eps_basic",
            "EarningsPerShareDiluted": "eps_diluted",
            "LongTermDebt": "long_term_debt",
            "CashAndCashEquivalentsAtCarryingValue": "cash",
            "Assets": "total_assets",
            "Liabilities": "total_liabilities",
            "StockholdersEquity": "stockholders_equity",
            "CommonStockSharesOutstanding": "shares_outstanding",
            "OperatingCashFlow": "operating_cash_flow",
            "NetCashProvidedByUsedInOperatingActivities": "operating_cash_flow",
        }

        result: dict[str, list[dict]] = {}
        seen: set[str] = set()  # avoid duplicate metric names

        for concept, metric_name in TARGET_CONCEPTS.items():
            if metric_name in seen or concept not in us_gaap:
                continue
            seen.add(metric_name)

            units_data = us_gaap[concept].get("units", {})
            # Primary unit: USD for financials, shares for counts, pure for ratios
            for unit_type, entries in units_data.items():
                # Only annual (10-K) and quarterly (10-Q) filings
                relevant = [
                    {
                        "value": e["val"],
                        "unit": unit_type,
                        "filed": e.get("filed", ""),
                        "form": e.get("form", ""),
                        "period_end": e.get("end", ""),
                        "accession": e.get("accn", ""),
                    }
                    for e in entries
                    if e.get("form") in ("10-K", "10-Q")
                ]
                if relevant:
                    relevant.sort(key=lambda x: x["filed"], reverse=True)
                    result[metric_name] = relevant[:8]  # last 8 periods
                    break

        return result
