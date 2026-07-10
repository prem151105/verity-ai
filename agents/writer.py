"""
Writer Agent Node
- Drafts a structured equity research report
- Attaches a citation pointer to EVERY non-trivial factual claim
- Retrieves supporting passages from ChromaDB for each major section
- Output is a markdown report ready for verification
"""

import json
import time
import logging
from agents.state import VerityState
from agents.llm import call_llm
from agents.audit_logger import AuditLogger
from config import settings

logger = logging.getLogger(__name__)

WRITER_SYSTEM = """
You are the Writer agent for Verity, a financial research system.
Your job: write a comprehensive, professional equity research report.

CRITICAL CITATION RULES:
- Every non-trivial factual claim must have a citation in this exact format:
  [[CITE: source_description | exact_passage_excerpt]]
- "source_description" = where the fact comes from (e.g. "10-K FY2024, Item 7")
- "exact_passage_excerpt" = 1-2 sentences from the retrieved passage that support the claim
- If you do not have a retrieved passage to support a claim, write [UNVERIFIED] instead
- NEVER invent numbers or facts not present in the provided context

REPORT STRUCTURE (use these exact section headers):
## Executive Summary
## Company Overview
## Financial Performance
### Revenue & Growth
### Profitability
### Balance Sheet & Leverage
## Market Valuation
## Recent Developments
## Bull Case
## Bear Case
## Key Risks
## Investment Thesis

Write in a professional, analytical tone. Be specific with numbers.
Always end each section with a brief synthesis, not just a list of facts.
"""


def writer_node(state: VerityState) -> VerityState:
    """
    LangGraph node: Writer.
    Generates a cited draft report from all collected context.
    """
    start = time.monotonic()
    ticker = state["ticker"]
    run_id = state["run_id"]
    audit = AuditLogger(settings.audit_log_dir, run_id)
    tool_calls = []

    # Retrieve targeted passages for each report section
    from tools.vector_store import VectorStore
    vs = VectorStore(settings.chroma_persist_dir, settings.gemini_api_key)
    collection_name = state.get("collection_name", "")

    section_queries = {
        "revenue_and_growth": "revenue growth sales performance quarterly results",
        "profitability": "gross margin operating margin net income profitability",
        "balance_sheet": "debt liabilities equity cash balance sheet leverage",
        "risks": "risk factors uncertainties challenges competition regulatory",
        "outlook": "guidance forecast future outlook management discussion",
        "recent_news": "recent developments news announcements products",
    }

    retrieved_context: dict[str, list] = {}
    for section, query in section_queries.items():
        chunks = vs.query(collection_name, query, n_results=4) if collection_name else []
        retrieved_context[section] = chunks
        tool_calls.append({
            "tool": "vector_store.query",
            "section": section,
            "query": query,
            "chunks": len(chunks),
        })

    # Build the writer prompt
    verifier_feedback = state.get("verifier_feedback", "")
    iteration = state.get("verifier_iteration", 0)

    context_block = _build_context_block(retrieved_context)
    ratios_block = _format_ratios_for_writer(state.get("financial_ratios", {}))
    market_block = _format_market_for_writer(state.get("market_data", {}))

    prompt = f"""
Company: {state.get('company_name', ticker)} ({ticker})
Report Date: {_today()}

## Financial Ratios (pre-computed, deterministic — cite as [SOURCE: Financial Ratios Computation])
{ratios_block}

## Market Data (cite as [SOURCE: Market Data — yfinance])
{market_block}

## Analyst Summary (cite individual sources within this as noted)
{state.get('analyst_summary', 'Not available')}

## Retrieved Passages from SEC Filings and Data Sources
{context_block}

{"## Verifier Feedback (REQUIRED: address all points below before finalizing)" if verifier_feedback else ""}
{verifier_feedback}

Now write the full equity research report following the required structure.
Attach [[CITE: ...]] to every factual claim. Use [UNVERIFIED] for any claim
you cannot support from the context above.
"""

    draft_report = ""
    citations = []

    try:
        draft_report = call_llm(prompt, system_instruction=WRITER_SYSTEM)
        citations = _extract_citations(draft_report)
        tool_calls.append({
            "tool": "llm.generate_report",
            "iteration": iteration,
            "prompt_length": len(prompt),
            "response_length": len(draft_report),
            "citations_found": len(citations),
        })
        logger.info(f"[Writer] Generated report ({len(draft_report)} chars, {len(citations)} citations)")
    except Exception as e:
        logger.error(f"[Writer] Report generation failed: {e}")
        draft_report = f"Report generation failed: {str(e)}"

    duration = time.monotonic() - start
    trace_entry = audit.log(
        node="writer",
        inputs={"ticker": ticker, "iteration": iteration},
        outputs={"report_length": len(draft_report), "citations_count": len(citations)},
        tool_calls=tool_calls,
        duration_seconds=duration,
    )

    return {
        **state,
        "draft_report": draft_report,
        "citations": citations,
        "trace": state.get("trace", []) + [trace_entry],
    }


def _build_context_block(retrieved: dict[str, list]) -> str:
    """Format retrieved chunks grouped by section for the writer prompt."""
    sections = []
    for section_name, chunks in retrieved.items():
        if not chunks:
            continue
        section_lines = [f"\n### {section_name.replace('_', ' ').title()}"]
        for chunk in chunks[:3]:
            section_lines.append(f"[SOURCE: {chunk.source}]\n{chunk.text[:500]}")
        sections.append("\n".join(section_lines))
    return "\n\n".join(sections)


def _format_ratios_for_writer(ratios: dict) -> str:
    if not ratios:
        return "Financial ratios not available."
    lines = []
    for name, data in ratios.items():
        val = data.get("value", "N/A")
        src = data.get("source", "Financial Ratios Computation")
        if isinstance(val, float):
            display = f"{val:.2f}"
            if "pct" in name or "margin" in name or "growth" in name or "return" in name:
                display += "%"
        else:
            display = f"{val:,}" if isinstance(val, int) else str(val)
        lines.append(f"- {name}: **{display}** (Source: {src})")
    return "\n".join(lines)


def _format_market_for_writer(market: dict) -> str:
    if not market:
        return "Market data not available."
    return (
        f"Current Price: ${market.get('current_price', 'N/A')} | "
        f"Market Cap: ${market.get('market_cap') or 0:,.0f} | "
        f"P/E: {market.get('pe_ratio', 'N/A')} | "
        f"Forward P/E: {market.get('forward_pe', 'N/A')} | "
        f"Beta: {market.get('beta', 'N/A')} | "
        f"Sector: {market.get('sector', 'N/A')}"
    )


def _extract_citations(report_text: str) -> list[dict]:
    """
    Parse [[CITE: source | passage]] markers from the report.
    Returns structured citation objects.
    """
    import re
    pattern = r'\[\[CITE:\s*(.*?)\s*\|\s*(.*?)\]\]'
    citations = []
    for match in re.finditer(pattern, report_text, re.DOTALL):
        source = match.group(1).strip()
        passage = match.group(2).strip()
        # Find the claim context (text before the citation marker)
        start_idx = max(0, match.start() - 200)
        claim_context = report_text[start_idx:match.start()].strip()
        # Get the last sentence as the claim
        sentences = claim_context.split('.')
        claim = (sentences[-1] if sentences else claim_context).strip()
        citations.append({
            "claim": claim[:300],
            "source": source,
            "passage": passage[:500],
            "verified": False,
            "confidence": 0.0,
        })
    return citations


def _today() -> str:
    from datetime import datetime
    return datetime.now().strftime("%B %d, %Y")
