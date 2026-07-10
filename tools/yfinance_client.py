"""
yfinance wrapper — clean interface for price history and fundamentals.
Adds retry logic and structured return types.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional

import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)


@dataclass
class PriceHistory:
    ticker: str
    start_date: str
    end_date: str
    df: pd.DataFrame  # columns: Open, High, Low, Close, Volume
    current_price: float = 0.0
    fifty_two_week_high: float = 0.0
    fifty_two_week_low: float = 0.0


@dataclass
class CompanyFundamentals:
    ticker: str
    name: str
    sector: str
    industry: str
    market_cap: Optional[float]
    pe_ratio: Optional[float]
    forward_pe: Optional[float]
    price_to_book: Optional[float]
    dividend_yield: Optional[float]
    beta: Optional[float]
    fifty_two_week_high: Optional[float]
    fifty_two_week_low: Optional[float]
    employees: Optional[int]
    description: str
    website: str
    raw_info: dict = field(default_factory=dict)


class YFinanceClient:
    """Thin, typed wrapper around yfinance."""

    def get_price_history(
        self,
        ticker: str,
        days: int = 90,
    ) -> PriceHistory:
        """
        Fetch OHLCV price history for a ticker.

        Args:
            ticker: Stock symbol (e.g. "AAPL")
            days: Number of calendar days of history to fetch

        Returns:
            PriceHistory dataclass with a clean DataFrame
        """
        end = datetime.now()
        start = end - timedelta(days=days)
        start_str = start.strftime("%Y-%m-%d")
        end_str = end.strftime("%Y-%m-%d")

        logger.info(f"Fetching {days}-day price history for {ticker}")
        stock = yf.Ticker(ticker)

        df = stock.history(start=start_str, end=end_str, auto_adjust=True)
        if df.empty:
            logger.warning(f"No price data returned for {ticker}")

        df.index = df.index.strftime("%Y-%m-%d")
        df = df[["Open", "High", "Low", "Close", "Volume"]].round(4)

        info = stock.info
        return PriceHistory(
            ticker=ticker.upper(),
            start_date=start_str,
            end_date=end_str,
            df=df,
            current_price=float(info.get("currentPrice", 0) or info.get("regularMarketPrice", 0) or 0),
            fifty_two_week_high=float(info.get("fiftyTwoWeekHigh", 0) or 0),
            fifty_two_week_low=float(info.get("fiftyTwoWeekLow", 0) or 0),
        )

    def get_fundamentals(self, ticker: str) -> CompanyFundamentals:
        """
        Fetch company fundamentals and description.

        Returns:
            CompanyFundamentals dataclass with cleaned fields.
        """
        logger.info(f"Fetching fundamentals for {ticker}")
        stock = yf.Ticker(ticker)
        info = stock.info

        def safe_float(key: str) -> Optional[float]:
            val = info.get(key)
            try:
                return float(val) if val is not None else None
            except (TypeError, ValueError):
                return None

        def safe_int(key: str) -> Optional[int]:
            val = info.get(key)
            try:
                return int(val) if val is not None else None
            except (TypeError, ValueError):
                return None

        return CompanyFundamentals(
            ticker=ticker.upper(),
            name=info.get("longName", ticker),
            sector=info.get("sector", "Unknown"),
            industry=info.get("industry", "Unknown"),
            market_cap=safe_float("marketCap"),
            pe_ratio=safe_float("trailingPE"),
            forward_pe=safe_float("forwardPE"),
            price_to_book=safe_float("priceToBook"),
            dividend_yield=safe_float("dividendYield"),
            beta=safe_float("beta"),
            fifty_two_week_high=safe_float("fiftyTwoWeekHigh"),
            fifty_two_week_low=safe_float("fiftyTwoWeekLow"),
            employees=safe_int("fullTimeEmployees"),
            description=info.get("longBusinessSummary", ""),
            website=info.get("website", ""),
            raw_info=info,
        )

    def get_price_returns(self, ticker: str, days: int = 90) -> dict[str, float]:
        """
        Compute price returns over various periods.

        Returns:
            Dict with keys: return_1w, return_1m, return_3m, return_ytd
        """
        history = self.get_price_history(ticker, days=max(days, 365))
        df = history.df

        if df.empty or len(df) < 2:
            return {}

        latest = df["Close"].iloc[-1]
        result = {}

        periods = {"return_1w": 5, "return_1m": 21, "return_3m": 63}
        for key, trading_days in periods.items():
            if len(df) > trading_days:
                past = df["Close"].iloc[-trading_days - 1]
                result[key] = round((latest - past) / past * 100, 2)

        return result
