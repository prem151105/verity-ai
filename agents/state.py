"""
Verity LangGraph Shared State
Defines the TypedDict that flows through all agent nodes.
Every node reads from and writes to this state.
"""

from typing import TypedDict, Optional, Annotated
from dataclasses import dataclass, field
import operator


@dataclass
class Citation:
    """Represents a traceable citation for a claim in the report."""
    claim: str
    source: str          # e.g. "10-K FY2024, Item 7 — Management Discussion"
    passage: str         # The exact retrieved text supporting the claim
    verified: bool = False
    confidence: float = 0.0   # 0.0 to 1.0


@dataclass
class AgentTrace:
    """Single trace entry for one agent node execution."""
    node: str
    inputs: dict
    outputs: dict
    tool_calls: list[dict]
    timestamp: str
    duration_seconds: float


class VerityState(TypedDict):
    """Shared state flowing through the LangGraph state graph."""

    # ── Input ─────────────────────────────────────────────────────────────────
    ticker: str
    run_id: str

    # ── Planner output ────────────────────────────────────────────────────────
    company_name: str
    task_list: list[str]       # Subtasks planned by the Planner

    # ── Retrieved data ────────────────────────────────────────────────────────
    filings: list[dict]        # Filing metadata (form, date, accession)
    filing_texts: list[dict]   # {source: str, text: str}
    market_data: dict          # Price history summary + fundamentals
    news_items: list[dict]     # {title, url, published_at, summary}
    collection_name: str       # ChromaDB collection for this run

    # ── Analyst output ────────────────────────────────────────────────────────
    financial_ratios: dict     # Computed ratios (from code_executor, not LLM)
    key_metrics: dict          # Raw XBRL metrics
    analyst_summary: str       # Narrative analysis with source tags

    # ── Writer output ─────────────────────────────────────────────────────────
    draft_report: str          # Full markdown report with inline citations
    citations: list[dict]      # Serialized Citation objects

    # ── Verifier state ────────────────────────────────────────────────────────
    verifier_iteration: int    # Current retry count (max = VERIFIER_MAX_RETRIES)
    verifier_feedback: str     # Feedback from verifier to writer
    unverified_claims: list[str]  # Claims the verifier could NOT verify

    # ── Final output ──────────────────────────────────────────────────────────
    final_report: str
    confidence_by_section: dict[str, float]
    error: Optional[str]

    # ── Observability ─────────────────────────────────────────────────────────
    trace: list[dict]          # Serialized AgentTrace entries (append-only)
