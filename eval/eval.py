"""
Verity Evaluation Harness
Runs the full pipeline on a fixed set of ~12 companies and reports:
  1. Citation coverage % (claims with valid citations)
  2. LLM-as-judge factual consistency score (claim vs. cited passage)
  3. End-to-end latency per report

Run: python eval/eval.py
Results saved to: eval/results.md
"""

import json
import time
import logging
import statistics
from datetime import datetime
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Fixed eval set — covers different sectors, sizes, and data richness
EVAL_TICKERS = [
    "AAPL",   # Large-cap tech, rich EDGAR data
    "MSFT",   # Large-cap tech
    "GOOGL",  # Large-cap tech
    "AMZN",   # E-commerce / cloud
    "NVDA",   # Semiconductors (high growth)
    "JPM",    # Financial — different GAAP treatment
    "JNJ",    # Healthcare
    "PG",     # Consumer staples
    "XOM",    # Energy
    "TSLA",   # EV — high volatility
    "META",   # Social media
    "NFLX",   # Streaming
]

RESULTS_PATH = Path("eval/results.md")


def run_eval():
    from agents.graph import run_research
    from agents.llm import call_llm

    results = []
    errors = []

    for ticker in EVAL_TICKERS:
        logger.info(f"\n{'='*50}\nEvaluating: {ticker}\n{'='*50}")
        run_start = time.monotonic()

        try:
            state = run_research(ticker)
            elapsed = time.monotonic() - run_start

            citations = state.get("citations", [])
            total = len(citations)
            verified = sum(1 for c in citations if c.get("verified"))
            coverage = (verified / total * 100) if total > 0 else 0.0

            # LLM-as-judge: score a sample of verified citations
            consistency_scores = []
            for citation in citations[:5]:  # score up to 5 citations
                if not citation.get("verified"):
                    continue
                score = _llm_judge_consistency(
                    claim=citation.get("claim", ""),
                    passage=citation.get("retrieved_passage", citation.get("passage", "")),
                    call_llm_fn=call_llm,
                )
                if score is not None:
                    consistency_scores.append(score)

            avg_consistency = statistics.mean(consistency_scores) if consistency_scores else None

            results.append({
                "ticker": ticker,
                "company": state.get("company_name", ticker),
                "status": "success",
                "latency_seconds": round(elapsed, 1),
                "citation_count": total,
                "verified_count": verified,
                "citation_coverage_pct": round(coverage, 1),
                "avg_consistency_score": round(avg_consistency, 2) if avg_consistency else None,
                "unverified_claims": len(state.get("unverified_claims", [])),
                "trace_nodes": len(state.get("trace", [])),
                "error": state.get("error"),
            })
            logger.info(f"{ticker}: coverage={coverage:.0f}%, latency={elapsed:.1f}s")

        except Exception as e:
            elapsed = time.monotonic() - run_start
            logger.error(f"{ticker} FAILED: {e}")
            errors.append({"ticker": ticker, "error": str(e)})
            results.append({
                "ticker": ticker,
                "status": "failed",
                "latency_seconds": round(elapsed, 1),
                "error": str(e),
            })

    # Write results to markdown
    _write_results_md(results)
    return results


def _llm_judge_consistency(claim: str, passage: str, call_llm_fn) -> float | None:
    """
    LLM-as-judge: score factual consistency of a claim vs its source passage.
    Returns a score 0.0–1.0 or None on failure.
    """
    if not claim or not passage:
        return None

    prompt = f"""
Rate the factual consistency of this claim against the source passage.
Score: 0.0 = completely contradicted, 0.5 = partially supported, 1.0 = fully supported.
Respond with ONLY a JSON object: {{"score": 0.0-1.0, "reason": "one sentence"}}

Claim: "{claim}"
Source passage: "{passage[:800]}"
"""
    try:
        raw = call_llm_fn(prompt)
        raw = raw.strip().lstrip("```json").lstrip("```").rstrip("```").strip()
        verdict = json.loads(raw)
        return float(verdict.get("score", 0.5))
    except Exception:
        return None


def _write_results_md(results: list[dict]) -> None:
    """Write evaluation results to a markdown file."""
    RESULTS_PATH.parent.mkdir(exist_ok=True)

    successful = [r for r in results if r.get("status") == "success"]
    failed = [r for r in results if r.get("status") == "failed"]

    if successful:
        avg_coverage = statistics.mean(r.get("citation_coverage_pct", 0) for r in successful)
        avg_latency = statistics.mean(r.get("latency_seconds", 0) for r in successful)
        consistency_scores = [r["avg_consistency_score"] for r in successful if r.get("avg_consistency_score")]
        avg_consistency = statistics.mean(consistency_scores) if consistency_scores else None
    else:
        avg_coverage = avg_latency = avg_consistency = 0

    lines = [
        f"# Verity Evaluation Results",
        f"",
        f"**Evaluated:** {datetime.now().strftime('%Y-%m-%d %H:%M UTC')}  ",
        f"**Companies tested:** {len(results)}  ",
        f"**Successful runs:** {len(successful)} / {len(results)}",
        f"",
        f"## Aggregate Metrics",
        f"",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| Average Citation Coverage | **{avg_coverage:.1f}%** |",
        f"| Average LLM-as-Judge Consistency | **{avg_consistency:.2f}** |" if avg_consistency else "| LLM Consistency | N/A |",
        f"| Average End-to-End Latency | **{avg_latency:.1f}s** |",
        f"| Success Rate | **{len(successful)}/{len(results)} ({len(successful)/max(len(results),1)*100:.0f}%)** |",
        f"",
        f"## Per-Company Results",
        f"",
        f"| Ticker | Company | Status | Coverage | Consistency | Latency | Citations | Unverified |",
        f"|--------|---------|--------|----------|-------------|---------|-----------|------------|",
    ]

    for r in results:
        status_icon = "✅" if r.get("status") == "success" else "❌"
        lines.append(
            f"| **{r['ticker']}** | {r.get('company', r['ticker'])} | {status_icon} |"
            f" {r.get('citation_coverage_pct', 'N/A')}% |"
            f" {r.get('avg_consistency_score', 'N/A')} |"
            f" {r.get('latency_seconds', 'N/A')}s |"
            f" {r.get('verified_count', 0)}/{r.get('citation_count', 0)} |"
            f" {r.get('unverified_claims', 'N/A')} |"
        )

    if failed:
        lines += ["", "## Failed Runs", ""]
        for r in failed:
            lines.append(f"- **{r['ticker']}**: {r.get('error', 'Unknown error')}")

    lines += [
        "",
        "---",
        "> Results generated by Verity evaluation harness. ",
        "> All numbers are real, measured from actual pipeline runs.",
        "> Citation coverage = verified citations / total citations in report.",
        "> LLM consistency = average Gemini-as-judge score (0–1) for sampled claim-passage pairs.",
    ]

    RESULTS_PATH.write_text("\n".join(lines), encoding="utf-8")
    logger.info(f"Results written to {RESULTS_PATH}")


if __name__ == "__main__":
    run_eval()
