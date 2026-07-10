"""
Planner Agent Node
- Receives ticker symbol
- Resolves company name via SEC EDGAR
- Decomposes research task into a structured task list
- Writes: company_name, task_list, collection_name
"""

import time
import logging
from agents.state import VerityState
from agents.llm import call_llm
from agents.audit_logger import AuditLogger
from config import settings

logger = logging.getLogger(__name__)

PLANNER_SYSTEM = """
You are the Planner agent for Verity, a financial research system.
Your job: given a stock ticker and company name, decompose the research task
into a concrete ordered task list.

Rules:
- Be specific and concrete. Each task should be actionable by a data retrieval agent.
- Focus on: SEC filings needed, financial metrics to compute, news context needed.
- Keep tasks to 6-8 items maximum.
- Output ONLY a JSON array of task strings, nothing else.

Example output:
[
  "Retrieve latest 10-K and most recent 10-Q for AAPL from SEC EDGAR",
  "Extract revenue, net income, gross profit, operating income for last 4 quarters",
  "Compute YoY revenue growth, gross margin %, operating margin %, debt/equity ratio",
  "Retrieve 90-day price history and compute price returns",
  "Fetch last 30 days of news coverage",
  "Identify key risk factors from 10-K Item 1A",
  "Summarize management guidance and forward outlook from most recent 10-Q MD&A"
]
"""


def planner_node(state: VerityState) -> VerityState:
    """
    LangGraph node: Planner.
    Resolves company name and creates a research task list.
    """
    start = time.monotonic()
    ticker = state["ticker"].upper().strip()
    run_id = state["run_id"]
    audit = AuditLogger(settings.audit_log_dir, run_id)

    logger.info(f"[Planner] Starting for ticker: {ticker}")

    # Resolve company name via SEC EDGAR
    from tools.edgar_client import EdgarClient
    edgar = EdgarClient(
        user_agent=settings.sec_user_agent,
        request_delay=settings.edgar_request_delay,
    )

    company_name = ticker  # fallback
    try:
        cik = edgar.get_cik(ticker)
        # Get company name from a quick submissions fetch
        import requests as req
        resp = req.get(
            f"https://data.sec.gov/submissions/CIK{cik}.json",
            headers={"User-Agent": settings.sec_user_agent},
            timeout=10,
        )
        if resp.ok:
            company_name = resp.json().get("name", ticker)
        logger.info(f"[Planner] Resolved {ticker} → {company_name} (CIK: {cik})")
    except Exception as e:
        logger.warning(f"[Planner] CIK resolution failed: {e}, using ticker as name")

    # Use LLM to build a specific task list for this company
    import json
    prompt = (
        f"Company: {company_name} ({ticker})\n\n"
        "Create a research task list for generating a comprehensive equity research report. "
        "Include specific SEC filing types, financial metrics, and analysis tasks."
    )

    task_list = []
    try:
        raw = call_llm(prompt, system_instruction=PLANNER_SYSTEM)
        # Strip markdown fences if present
        raw = raw.strip().lstrip("```json").lstrip("```").rstrip("```").strip()
        task_list = json.loads(raw)
        if not isinstance(task_list, list):
            task_list = [str(task_list)]
    except Exception as e:
        logger.warning(f"[Planner] LLM task decomposition failed: {e}, using defaults")
        task_list = [
            f"Retrieve latest 10-K and 10-Q for {ticker} from SEC EDGAR",
            "Extract key financial metrics: revenue, net income, gross profit, operating income",
            "Compute financial ratios: gross margin, operating margin, net margin, debt/equity, YoY growth",
            f"Retrieve 90-day price history for {ticker}",
            "Fetch recent news coverage (last 30 days)",
            "Identify key risk factors from 10-K",
            "Analyze management discussion and forward outlook",
        ]

    collection_name = f"verity_{ticker.lower()}_{run_id[:8]}"

    duration = time.monotonic() - start
    trace_entry = audit.log(
        node="planner",
        inputs={"ticker": ticker},
        outputs={"company_name": company_name, "task_count": len(task_list)},
        tool_calls=[{"tool": "sec_edgar.get_cik", "ticker": ticker}],
        duration_seconds=duration,
    )

    return {
        **state,
        "company_name": company_name,
        "task_list": task_list,
        "collection_name": collection_name,
        "trace": state.get("trace", []) + [trace_entry],
        "error": None,
    }
