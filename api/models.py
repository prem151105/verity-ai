"""
FastAPI Pydantic schemas for request/response validation.
"""

from pydantic import BaseModel, Field
from typing import Optional


class ResearchRequest(BaseModel):
    ticker: str = Field(..., min_length=1, max_length=10, description="Stock ticker symbol")


class ResearchResponse(BaseModel):
    run_id: str
    ticker: str
    company_name: str
    status: str  # "completed" | "failed"
    message: str = ""


class ReportResponse(BaseModel):
    run_id: str
    ticker: str
    company_name: str
    final_report: str
    citation_count: int
    verified_citation_count: int
    citation_coverage_pct: float
    unverified_claims: list[str]
    confidence_by_section: dict[str, float]
    task_list: list[str]
    financial_ratios: dict
    trace_node_count: int
    error: Optional[str] = None


class CitationDetail(BaseModel):
    claim: str
    source: str
    passage: str
    verified: bool
    confidence: float
    verifier_reasoning: str = ""


class TraceEntry(BaseModel):
    node: str
    timestamp: str
    duration_seconds: float
    inputs: dict
    outputs: dict
    tool_calls: list[dict]
