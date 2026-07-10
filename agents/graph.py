"""
Verity LangGraph State Graph
Wires all agent nodes into a directed state graph with conditional routing.

Graph flow:
  planner → retriever → analyst → writer → verifier →
    (if issues && retries left) → writer (loop)
    (if done) → assembler → END
"""

import uuid
import logging
from datetime import datetime, timezone

from langgraph.graph import StateGraph, END

from agents.state import VerityState
from agents.planner import planner_node
from agents.retriever import retriever_node
from agents.analyst import analyst_node
from agents.writer import writer_node
from agents.verifier import verifier_node, should_loop_to_writer
from agents.assembler import assembler_node

logger = logging.getLogger(__name__)


def build_graph() -> StateGraph:
    """
    Build and compile the Verity LangGraph StateGraph.

    Returns:
        Compiled LangGraph app ready for .invoke() or .stream()
    """
    graph = StateGraph(VerityState)

    # Add nodes
    graph.add_node("planner", planner_node)
    graph.add_node("retriever", retriever_node)
    graph.add_node("analyst", analyst_node)
    graph.add_node("writer", writer_node)
    graph.add_node("verifier", verifier_node)
    graph.add_node("assembler", assembler_node)

    # Linear edges (sequential pipeline)
    graph.set_entry_point("planner")
    graph.add_edge("planner", "retriever")
    graph.add_edge("retriever", "analyst")
    graph.add_edge("analyst", "writer")
    graph.add_edge("writer", "verifier")

    # Conditional edge: verifier → (writer loop | assembler)
    graph.add_conditional_edges(
        "verifier",
        should_loop_to_writer,
        {
            "writer": "writer",
            "assembler": "assembler",
        },
    )

    graph.add_edge("assembler", END)

    return graph.compile()


def run_research(ticker: str) -> dict:
    """
    Run the full Verity research pipeline for a given ticker.

    Args:
        ticker: Stock ticker symbol (e.g. "AAPL")

    Returns:
        Final state dict containing the report and trace.
    """
    run_id = str(uuid.uuid4())
    logger.info(f"Starting Verity research run: {run_id} for ticker: {ticker}")

    graph = build_graph()

    initial_state: VerityState = {
        "ticker": ticker.upper().strip(),
        "run_id": run_id,
        "company_name": "",
        "task_list": [],
        "filings": [],
        "filing_texts": [],
        "market_data": {},
        "news_items": [],
        "collection_name": "",
        "financial_ratios": {},
        "key_metrics": {},
        "analyst_summary": "",
        "draft_report": "",
        "citations": [],
        "verifier_iteration": 0,
        "verifier_feedback": "",
        "unverified_claims": [],
        "final_report": "",
        "confidence_by_section": {},
        "error": None,
        "trace": [],
    }

    try:
        final_state = graph.invoke(initial_state)
        logger.info(f"Research run completed: {run_id}")
        return final_state
    except Exception as e:
        logger.error(f"Research run failed: {run_id} — {e}", exc_info=True)
        return {**initial_state, "error": str(e), "final_report": f"Run failed: {str(e)}"}
