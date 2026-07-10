"""
Analyst Agent Node
- Computes financial ratios using Python code (NOT LLM arithmetic)
- Every number is tagged with its source document
- Produces a structured narrative summary with source citations
"""

import time
import logging
from agents.state import VerityState
from agents.llm import call_llm
from agents.audit_logger import AuditLogger
from tools.code_executor import compute_financial_ratios
from config import settings

logger = logging.getLogger(__name__)

ANALYST_SYSTEM = """
You are the Analyst agent for Verity, a financial research system.
You will receive pre-computed financial ratios and market data (already calculated
by deterministic Python code — do NOT recalculate these numbers yourself).

Your job: write a structured financial analysis narrative that:
1. Interprets each ratio in context (what does a 35% gross margin mean for this industry?)
2. Highlights notable trends (improving/deteriorating margins, leverage changes)
3. Connects market data to fundamentals (is the P/E justified given growth?)
4. Identifies potential risks from the data patterns

CRITICAL RULES:
- Every factual claim you make MUST reference the source provided in brackets [SOURCE].
- Never invent numbers. Use ONLY the numbers provided in the input.
- Format: structured paragraphs, one per analysis area.
- Include inline citations like: "Revenue grew X% YoY [SOURCE: Revenue YoY Growth (10-K filings)]"
"""


def analyst_node(state: VerityState) -> VerityState:
    """
    LangGraph node: Analyst.
    Runs deterministic ratio computation, then LLM narrative generation.
    """
    start = time.monotonic()
    ticker = state["ticker"]
    run_id = state["run_id"]
    audit = AuditLogger(settings.audit_log_dir, run_id)
    tool_calls = []

    # ── Step 1: Deterministic ratio computation ───────────────────────────────
    financial_ratios = {}
    key_metrics = state.get("key_metrics", {})

    try:
        financial_ratios = compute_financial_ratios(key_metrics)
        tool_calls.append({
            "tool": "code_executor.compute_financial_ratios",
            "ratios_computed": list(financial_ratios.keys()),
        })
        logger.info(f"[Analyst] Computed {len(financial_ratios)} financial ratios")
    except Exception as e:
        logger.error(f"[Analyst] Ratio computation failed: {e}")

    # ── Step 2: Retrieve additional context from vector store ─────────────────
    from tools.vector_store import VectorStore
    vs = VectorStore(settings.chroma_persist_dir, settings.gemini_api_key)
    collection_name = state.get("collection_name", "")

    context_chunks = []
    if collection_name:
        queries = [
            "revenue growth and profitability trends",
            "risk factors and business challenges",
            "management guidance and forward outlook",
        ]
        for q in queries:
            chunks = vs.query(collection_name, q, n_results=3)
            context_chunks.extend(chunks)
            tool_calls.append({
                "tool": "vector_store.query",
                "query": q,
                "chunks_returned": len(chunks),
            })

    # ── Step 3: Build analysis prompt with computed ratios ────────────────────
    ratios_text = _format_ratios(financial_ratios)
    market_text = _format_market_data(state.get("market_data", {}))
    context_text = "\n\n".join(
        f"[SOURCE: {c.source}]\n{c.text[:600]}" for c in context_chunks[:6]
    )

    prompt = f"""
Company: {state.get('company_name', ticker)} ({ticker})

## Pre-Computed Financial Ratios (from SEC EDGAR XBRL data)
{ratios_text}

## Market Data (from yfinance)
{market_text}

## Retrieved Context from SEC Filings
{context_text}

Write a structured financial analysis covering:
1. Revenue and profitability trends
2. Balance sheet and leverage position
3. Market valuation context
4. Key risks identified in filings

Remember: cite every number with its [SOURCE] tag.
"""

    analyst_summary = ""
    try:
        analyst_summary = call_llm(prompt, system_instruction=ANALYST_SYSTEM)
        tool_calls.append({
            "tool": "llm.generate",
            "prompt_length": len(prompt),
            "response_length": len(analyst_summary),
        })
    except Exception as e:
        logger.error(f"[Analyst] LLM analysis failed: {e}")
        analyst_summary = f"Analysis could not be generated: {str(e)}\n\n{ratios_text}"

    duration = time.monotonic() - start
    trace_entry = audit.log(
        node="analyst",
        inputs={"ticker": ticker, "metrics_count": len(key_metrics)},
        outputs={"ratios_count": len(financial_ratios), "summary_length": len(analyst_summary)},
        tool_calls=tool_calls,
        duration_seconds=duration,
    )

    return {
        **state,
        "financial_ratios": financial_ratios,
        "analyst_summary": analyst_summary,
        "trace": state.get("trace", []) + [trace_entry],
    }


def _format_ratios(ratios: dict) -> str:
    if not ratios:
        return "No ratios computed (insufficient XBRL data)."
    lines = []
    for name, data in ratios.items():
        val = data.get("value", "N/A")
        src = data.get("source", "")
        if isinstance(val, float):
            val_str = f"{val:,.2f}"
        elif isinstance(val, int):
            val_str = f"{val:,.0f}"
        else:
            val_str = str(val)
        # Add % for percentage metrics
        if "pct" in name or "margin" in name or "growth" in name or "return" in name:
            val_str += "%"
        lines.append(f"- **{name}**: {val_str} [SOURCE: {src}]")
    return "\n".join(lines)


def _format_market_data(market: dict) -> str:
    if not market:
        return "Market data not available."
    lines = [
        f"- Current Price: ${market.get('current_price', 'N/A')}",
        f"- Market Cap: ${market.get('market_cap') or 0:,.0f}",
        f"- P/E (trailing): {market.get('pe_ratio', 'N/A')}",
        f"- Forward P/E: {market.get('forward_pe', 'N/A')}",
        f"- Price/Book: {market.get('price_to_book', 'N/A')}",
        f"- Beta: {market.get('beta', 'N/A')}",
        f"- 52-Week Range: ${market.get('52w_low', 'N/A')} – ${market.get('52w_high', 'N/A')}",
        f"- 1M Return: {market.get('return_1m', 'N/A')}%",
        f"- 3M Return: {market.get('return_3m', 'N/A')}%",
    ]
    return "\n".join(lines)
