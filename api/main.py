"""
Verity FastAPI Application
Endpoints:
  POST /research/{ticker}   — kick off a research run (async background)
  GET  /research/{run_id}   — fetch the completed report and trace
  GET  /health              — health check
  GET  /runs                — list recent runs
"""

import asyncio
import json
import logging
import os
import uuid
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, BackgroundTasks, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from api.models import ResearchRequest, ResearchResponse, ReportResponse, TraceEntry
from config import settings

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)

# In-memory store for run results (production: use a DB/Redis)
# For portfolio purposes: results persist as JSONL in audit_logs/
_runs: dict[str, dict] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup/shutdown lifecycle."""
    logger.info("Verity API starting up...")
    Path(settings.audit_log_dir).mkdir(parents=True, exist_ok=True)
    Path(settings.chroma_persist_dir).mkdir(parents=True, exist_ok=True)
    yield
    logger.info("Verity API shutting down...")


app = FastAPI(
    title="Verity — Multi-Agent Financial Research API",
    description=(
        "A 6-agent LangGraph system that autonomously retrieves SEC filings "
        "and market data, generates cited equity research reports, and "
        "verifies every claim against its source."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "Verity",
        "version": "1.0.0",
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "active_runs": sum(1 for r in _runs.values() if r.get("status") == "running"),
    }


@app.post("/research/{ticker}", response_model=ResearchResponse)
async def start_research(ticker: str, background_tasks: BackgroundTasks):
    """
    Kick off a Verity research run for a ticker.
    Returns immediately with a run_id — poll GET /research/{run_id} for results.
    """
    ticker = ticker.upper().strip()
    run_id = str(uuid.uuid4())

    _runs[run_id] = {
        "run_id": run_id,
        "ticker": ticker,
        "status": "running",
        "started_at": datetime.utcnow().isoformat() + "Z",
        "state": None,
    }

    background_tasks.add_task(_run_research_task, run_id, ticker)

    return ResearchResponse(
        run_id=run_id,
        ticker=ticker,
        company_name=ticker,
        status="running",
        message=f"Research started. Poll GET /research/{run_id} for results.",
    )


@app.get("/research/{run_id}", response_model=ReportResponse)
async def get_research(run_id: str):
    """
    Fetch a completed research report and full agent trace.
    """
    if run_id not in _runs:
        # Try loading from disk (survives API restarts)
        state = _load_run_from_disk(run_id)
        if state:
            _runs[run_id] = {"run_id": run_id, "status": "completed", "state": state}
        else:
            raise HTTPException(status_code=404, detail=f"Run '{run_id}' not found.")

    run = _runs[run_id]
    status = run.get("status", "unknown")

    if status == "running":
        raise HTTPException(
            status_code=202,
            detail={
                "status": "running",
                "message": "Research in progress. Please retry in ~30 seconds.",
            },
        )

    if status == "failed":
        raise HTTPException(
            status_code=500,
            detail={"status": "failed", "error": run.get("error", "Unknown error")},
        )

    state = run.get("state", {})
    citations = state.get("citations", [])
    verified = [c for c in citations if c.get("verified")]

    return ReportResponse(
        run_id=run_id,
        ticker=state.get("ticker", ""),
        company_name=state.get("company_name", ""),
        final_report=state.get("final_report", ""),
        citation_count=len(citations),
        verified_citation_count=len(verified),
        citation_coverage_pct=round(len(verified) / max(len(citations), 1) * 100, 1),
        unverified_claims=state.get("unverified_claims", []),
        confidence_by_section=state.get("confidence_by_section", {}),
        task_list=state.get("task_list", []),
        financial_ratios=state.get("financial_ratios", {}),
        trace_node_count=len(state.get("trace", [])),
        error=state.get("error"),
    )


@app.get("/research/{run_id}/trace")
async def get_trace(run_id: str):
    """Return the full agent trace for a run."""
    if run_id not in _runs:
        raise HTTPException(status_code=404, detail="Run not found.")

    state = _runs[run_id].get("state", {})
    if not state:
        raise HTTPException(status_code=202, detail="Run still in progress.")

    return {"run_id": run_id, "trace": state.get("trace", [])}


@app.get("/research/{run_id}/citations")
async def get_citations(run_id: str):
    """Return detailed citation verification results for a run."""
    if run_id not in _runs:
        raise HTTPException(status_code=404, detail="Run not found.")

    state = _runs[run_id].get("state", {})
    citations = state.get("citations", [])
    return {
        "run_id": run_id,
        "total": len(citations),
        "verified": sum(1 for c in citations if c.get("verified")),
        "citations": citations,
    }


@app.get("/runs")
async def list_runs():
    """List recent research runs."""
    return {
        "runs": [
            {
                "run_id": r["run_id"],
                "ticker": r.get("ticker", ""),
                "status": r.get("status", ""),
                "started_at": r.get("started_at", ""),
            }
            for r in _runs.values()
        ]
    }


# ── Background task ────────────────────────────────────────────────────────────

async def _run_research_task(run_id: str, ticker: str):
    """Execute the research graph in a background thread."""
    import concurrent.futures
    loop = asyncio.get_event_loop()

    with concurrent.futures.ThreadPoolExecutor() as pool:
        try:
            from agents.graph import run_research
            state = await loop.run_in_executor(pool, run_research, ticker)
            _runs[run_id]["status"] = "completed"
            _runs[run_id]["state"] = state
            _save_run_to_disk(run_id, state)
            logger.info(f"Research run {run_id} completed for {ticker}")
        except Exception as e:
            logger.error(f"Research run {run_id} failed: {e}", exc_info=True)
            _runs[run_id]["status"] = "failed"
            _runs[run_id]["error"] = str(e)


def _save_run_to_disk(run_id: str, state: dict):
    """Persist final state to disk so it survives API restarts."""
    path = Path(settings.audit_log_dir) / f"result_{run_id}.json"
    try:
        # Serialize (state may have non-JSON-serializable items)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(state, f, default=str, indent=2)
    except Exception as e:
        logger.warning(f"Could not save run to disk: {e}")


def _load_run_from_disk(run_id: str) -> Optional[dict]:
    """Load a previously completed run from disk."""
    path = Path(settings.audit_log_dir) / f"result_{run_id}.json"
    if path.exists():
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return None
