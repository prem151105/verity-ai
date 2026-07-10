"""
Financial ratio computation from SEC EDGAR XBRL data.
All ratios are computed with deterministic Python arithmetic (never by the LLM)
and tagged with their source filing so claims are fully auditable.
"""

import logging
from typing import Any

logger = logging.getLogger(__name__)


def compute_financial_ratios(metrics: dict[str, list[dict]]) -> dict[str, Any]:
    """
    Compute standard financial ratios from extracted XBRL metrics.
    All calculations are done here in pure Python (not by the LLM).

    Args:
        metrics: Output from EdgarClient.extract_key_metrics()

    Returns:
        Dict of ratio name → computed value, each tagged with source filing info.
    """

    def latest_value(metric_name: str) -> tuple[float | None, dict]:
        """Get the most recent value for a metric and its filing metadata."""
        entries = metrics.get(metric_name, [])
        if not entries:
            return None, {}
        entry = entries[0]
        return entry.get("value"), entry

    def yoy_growth(metric_name: str) -> float | None:
        """Year-over-year growth rate for annual filings."""
        entries = [e for e in metrics.get(metric_name, []) if e.get("form") == "10-K"]
        if len(entries) < 2:
            return None
        current = entries[0]["value"]
        prior = entries[1]["value"]
        if prior == 0:
            return None
        return round((current - prior) / abs(prior) * 100, 2)

    results: dict[str, Any] = {}

    revenue, rev_meta = latest_value("revenue")
    net_income, ni_meta = latest_value("net_income")
    gross_profit, gp_meta = latest_value("gross_profit")
    operating_income, oi_meta = latest_value("operating_income")
    total_assets, ta_meta = latest_value("total_assets")
    total_liabilities, tl_meta = latest_value("total_liabilities")
    stockholders_equity, se_meta = latest_value("stockholders_equity")
    long_term_debt, ltd_meta = latest_value("long_term_debt")
    cash, cash_meta = latest_value("cash")

    # ── Profitability ratios ──────────────────────────────────────────────────
    if revenue and gross_profit:
        results["gross_margin_pct"] = {
            "value": round(gross_profit / revenue * 100, 2),
            "source": f"Gross Profit ({gp_meta.get('form', '')}, {gp_meta.get('period_end', '')}) / Revenue",
        }

    if revenue and operating_income:
        results["operating_margin_pct"] = {
            "value": round(operating_income / revenue * 100, 2),
            "source": f"Operating Income ({oi_meta.get('form', '')}, {oi_meta.get('period_end', '')}) / Revenue",
        }

    if revenue and net_income:
        results["net_margin_pct"] = {
            "value": round(net_income / revenue * 100, 2),
            "source": f"Net Income ({ni_meta.get('form', '')}, {ni_meta.get('period_end', '')}) / Revenue",
        }

    if total_assets and net_income:
        results["return_on_assets_pct"] = {
            "value": round(net_income / total_assets * 100, 2),
            "source": f"Net Income / Total Assets ({ta_meta.get('period_end', '')})",
        }

    if stockholders_equity and net_income and stockholders_equity != 0:
        results["return_on_equity_pct"] = {
            "value": round(net_income / stockholders_equity * 100, 2),
            "source": f"Net Income / Stockholders Equity ({se_meta.get('period_end', '')})",
        }

    # ── Leverage ratios ───────────────────────────────────────────────────────
    if stockholders_equity and long_term_debt and stockholders_equity != 0:
        results["debt_to_equity"] = {
            "value": round(long_term_debt / stockholders_equity, 3),
            "source": f"Long-term Debt / Equity ({ltd_meta.get('period_end', '')})",
        }

    if total_assets and total_liabilities:
        results["debt_to_assets"] = {
            "value": round(total_liabilities / total_assets, 3),
            "source": f"Total Liabilities / Total Assets ({ta_meta.get('period_end', '')})",
        }

    # ── Growth rates ──────────────────────────────────────────────────────────
    rev_growth = yoy_growth("revenue")
    if rev_growth is not None:
        results["revenue_yoy_growth_pct"] = {
            "value": rev_growth,
            "source": "Revenue YoY Growth (10-K filings)",
        }

    ni_growth = yoy_growth("net_income")
    if ni_growth is not None:
        results["net_income_yoy_growth_pct"] = {
            "value": ni_growth,
            "source": "Net Income YoY Growth (10-K filings)",
        }

    # ── Cash position ─────────────────────────────────────────────────────────
    if cash:
        results["cash_position"] = {
            "value": cash,
            "source": f"Cash & Equivalents ({cash_meta.get('form', '')}, {cash_meta.get('period_end', '')})",
        }

    return results
