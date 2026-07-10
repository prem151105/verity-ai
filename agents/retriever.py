"""
Retriever Agent Node
- Fetches SEC 10-K and 10-Q filings
- Fetches market data via yfinance
- Optionally fetches news
- Chunks + embeds documents into ChromaDB
- Tags each chunk with source metadata for citations
"""

import time
import logging
from agents.state import VerityState
from agents.audit_logger import AuditLogger
from config import settings

logger = logging.getLogger(__name__)


def retriever_node(state: VerityState) -> VerityState:
    """
    LangGraph node: Retriever.
    Pulls all data sources and populates the ChromaDB vector store.
    """
    start = time.monotonic()
    ticker = state["ticker"]
    run_id = state["run_id"]
    collection_name = state["collection_name"]
    audit = AuditLogger(settings.audit_log_dir, run_id)
    tool_calls = []

    from tools.edgar_client import EdgarClient
    from tools.yfinance_client import YFinanceClient
    from tools.vector_store import VectorStore

    edgar = EdgarClient(settings.sec_user_agent, settings.edgar_request_delay)
    yf_client = YFinanceClient()
    vs = VectorStore(settings.chroma_persist_dir, settings.gemini_api_key)

    # ── SEC EDGAR: Recent filings ─────────────────────────────────────────────
    filings_meta = []
    filing_texts = []

    try:
        filings = edgar.get_recent_filings(ticker, form_types=["10-K", "10-Q"], limit=2)
        tool_calls.append({
            "tool": "edgar.get_recent_filings",
            "ticker": ticker,
            "forms": ["10-K", "10-Q"],
            "count": len(filings),
        })

        for filing in filings:
            filings_meta.append({
                "form_type": filing.form_type,
                "filed_date": filing.filed_date,
                "accession_number": filing.accession_number,
                "company_name": filing.company_name,
            })

            source_label = f"{filing.form_type} filed {filing.filed_date} — {filing.company_name}"
            logger.info(f"[Retriever] Fetching document: {source_label}")

            text = edgar.get_filing_document_text(filing)
            tool_calls.append({
                "tool": "edgar.get_filing_document_text",
                "source": source_label,
                "chars_retrieved": len(text),
            })

            if text:
                filing_texts.append({"source": source_label, "text": text[:5000]})  # store preview
                chunks_added = vs.add_document(
                    collection_name=collection_name,
                    text=text,
                    source=source_label,
                    metadata={"form_type": filing.form_type, "filed_date": filing.filed_date},
                )
                tool_calls.append({
                    "tool": "vector_store.add_document",
                    "source": source_label,
                    "chunks": chunks_added,
                })

    except Exception as e:
        logger.error(f"[Retriever] SEC EDGAR retrieval failed: {e}")

    # ── XBRL Financial Facts ──────────────────────────────────────────────────
    key_metrics = {}
    try:
        key_metrics = edgar.extract_key_metrics(ticker)
        tool_calls.append({
            "tool": "edgar.extract_key_metrics",
            "ticker": ticker,
            "metrics_found": list(key_metrics.keys()),
        })
        # Also embed a structured text summary of metrics for retrieval
        metrics_text = _metrics_to_text(ticker, key_metrics)
        vs.add_document(
            collection_name=collection_name,
            text=metrics_text,
            source=f"XBRL Financial Facts — {ticker}",
            metadata={"form_type": "XBRL", "filed_date": ""},
        )
    except Exception as e:
        logger.warning(f"[Retriever] XBRL facts retrieval failed: {e}")

    # ── Market Data: yfinance ─────────────────────────────────────────────────
    market_data = {}
    try:
        price_history = yf_client.get_price_history(ticker, days=90)
        fundamentals = yf_client.get_fundamentals(ticker)
        returns = yf_client.get_price_returns(ticker, days=90)

        market_data = {
            "current_price": price_history.current_price,
            "52w_high": price_history.fifty_two_week_high,
            "52w_low": price_history.fifty_two_week_low,
            "pe_ratio": fundamentals.pe_ratio,
            "forward_pe": fundamentals.forward_pe,
            "price_to_book": fundamentals.price_to_book,
            "market_cap": fundamentals.market_cap,
            "beta": fundamentals.beta,
            "sector": fundamentals.sector,
            "industry": fundamentals.industry,
            "dividend_yield": fundamentals.dividend_yield,
            "employees": fundamentals.employees,
            "description": fundamentals.description[:1500],
            "website": fundamentals.website,
            "price_df_tail": price_history.df.tail(5).to_dict(),
            **returns,
        }
        tool_calls.append({
            "tool": "yfinance.get_fundamentals",
            "ticker": ticker,
            "sector": fundamentals.sector,
            "market_cap": fundamentals.market_cap,
        })

        # Embed market data summary
        mkt_text = _market_data_to_text(ticker, market_data)
        vs.add_document(
            collection_name=collection_name,
            text=mkt_text,
            source=f"Market Data (yfinance) — {ticker}",
            metadata={"form_type": "market_data"},
        )
    except Exception as e:
        logger.warning(f"[Retriever] yfinance retrieval failed: {e}")

    # ── News (optional) ───────────────────────────────────────────────────────
    news_items = []
    if settings.news_api_key:
        try:
            news_items = _fetch_news(ticker, state.get("company_name", ticker))
            if news_items:
                news_text = "\n\n".join(
                    f"**{n['title']}** ({n['published_at']})\n{n.get('description', '')}"
                    for n in news_items
                )
                vs.add_document(
                    collection_name=collection_name,
                    text=news_text,
                    source=f"News (last 30 days) — {ticker}",
                    metadata={"form_type": "news"},
                )
        except Exception as e:
            logger.warning(f"[Retriever] News fetch failed: {e}")

    duration = time.monotonic() - start
    trace_entry = audit.log(
        node="retriever",
        inputs={"ticker": ticker, "collection_name": collection_name},
        outputs={
            "filings_fetched": len(filings_meta),
            "metrics_found": len(key_metrics),
            "market_data_keys": list(market_data.keys()),
            "news_items": len(news_items),
        },
        tool_calls=tool_calls,
        duration_seconds=duration,
    )

    return {
        **state,
        "filings": filings_meta,
        "filing_texts": filing_texts,
        "key_metrics": key_metrics,
        "market_data": market_data,
        "news_items": news_items,
        "trace": state.get("trace", []) + [trace_entry],
    }


def _metrics_to_text(ticker: str, metrics: dict) -> str:
    """Convert extracted XBRL metrics to readable text for embedding."""
    lines = [f"Financial Metrics for {ticker} (XBRL data from SEC EDGAR)\n"]
    for metric_name, entries in metrics.items():
        if entries:
            latest = entries[0]
            lines.append(
                f"{metric_name}: {latest.get('value'):,.0f} {latest.get('unit', '')} "
                f"(Period: {latest.get('period_end', 'N/A')}, Filing: {latest.get('form', 'N/A')})"
            )
    return "\n".join(lines)


def _market_data_to_text(ticker: str, data: dict) -> str:
    """Convert market data dict to readable text for embedding."""
    return f"""
Market Data Summary for {ticker}
Current Price: ${data.get('current_price', 'N/A')}
52-Week High: ${data.get('52w_high', 'N/A')} | Low: ${data.get('52w_low', 'N/A')}
P/E Ratio (trailing): {data.get('pe_ratio', 'N/A')}
Forward P/E: {data.get('forward_pe', 'N/A')}
Price to Book: {data.get('price_to_book', 'N/A')}
Market Cap: ${data.get('market_cap') or 0:,.0f}
Beta: {data.get('beta', 'N/A')}
Sector: {data.get('sector', 'N/A')}
Industry: {data.get('industry', 'N/A')}
Dividend Yield: {data.get('dividend_yield', 'N/A')}
Employees: {data.get('employees', 'N/A'):,}
1-Week Return: {data.get('return_1w', 'N/A')}%
1-Month Return: {data.get('return_1m', 'N/A')}%
3-Month Return: {data.get('return_3m', 'N/A')}%

Company Description:
{data.get('description', '')}
""".strip()


def _fetch_news(ticker: str, company_name: str) -> list[dict]:
    """Fetch recent news via NewsAPI (requires API key)."""
    import requests
    url = "https://newsapi.org/v2/everything"
    params = {
        "q": f'"{ticker}" OR "{company_name}"',
        "sortBy": "publishedAt",
        "pageSize": 10,
        "language": "en",
        "apiKey": settings.news_api_key,
    }
    resp = requests.get(url, params=params, timeout=10)
    resp.raise_for_status()
    articles = resp.json().get("articles", [])
    return [
        {
            "title": a.get("title", ""),
            "url": a.get("url", ""),
            "published_at": a.get("publishedAt", "")[:10],
            "description": a.get("description", ""),
            "source": a.get("source", {}).get("name", ""),
        }
        for a in articles
    ]
