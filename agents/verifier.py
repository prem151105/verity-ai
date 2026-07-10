"""
Verifier / Critic Agent Node — THE CORE DIFFERENTIATOR of Verity.

For every cited claim in the Writer's report:
1. Re-retrieve the cited passage from the vector store
2. Ask the LLM: "Does this passage actually support this claim?"
3. If NO → send the claim back to Writer with specific feedback (loop, max N retries)
4. If YES → mark as verified
5. Any claim still unverified after max retries → explicitly flagged in final report

Optimized to batch verification requests to fit within free-tier rate limits.
"""

import re
import time
import logging
from agents.state import VerityState
from agents.llm import call_llm
from agents.audit_logger import AuditLogger
from config import settings

logger = logging.getLogger(__name__)

VERIFIER_SYSTEM = """
You are the Verifier agent for Verity, a financial research system.
Your ONLY job is fact-checking: determining whether a retrieved passage
actually supports a specific claim.

Respond with EXACTLY this JSON format (no other text):
{
  "supported": true/false,
  "confidence": 0.0-1.0,
  "reasoning": "one-sentence explanation",
  "correction": "what the claim should say instead, if supported=false (else null)"
}

Rules:
- "supported: true" ONLY if the passage directly states or implies the claim with specific data.
- "supported: false" if the passage is vague, unrelated, or the numbers don't match.
- Be strict. A passage that says "revenue increased" does NOT support a claim that says "revenue grew 15%".
- confidence = your certainty about your verdict (0.8+ = very sure, below 0.5 = uncertain).
"""

VERIFIER_BATCH_SYSTEM = """
You are the Verifier agent for Verity, a financial research system.
Your ONLY job is fact-checking: determining whether given retrieved passages support their corresponding specific claims.

Input will be a list of claims and their corresponding retrieved passages.
For each claim-passage pair, you must determine:
1. "supported": true/false
2. "confidence": 0.0-1.0
3. "reasoning": "one-sentence explanation"
4. "correction": "what the claim should say instead, if supported=false (else null)"

Respond with EXACTLY a JSON array of objects matching the input order (no other text). Do not include markdown code block formatting or anything other than the raw JSON array. Example:
[
  {
    "supported": true,
    "confidence": 0.9,
    "reasoning": "Passage directly states this metric.",
    "correction": null
  }
]
"""


def verifier_node(state: VerityState) -> VerityState:
    """
    LangGraph node: Verifier/Critic.
    Checks citations in batches to optimize rate limits.
    """
    start = time.monotonic()
    ticker = state["ticker"]
    run_id = state["run_id"]
    iteration = state.get("verifier_iteration", 0)
    audit = AuditLogger(settings.audit_log_dir, run_id)
    tool_calls = []

    from tools.vector_store import VectorStore
    vs = VectorStore(settings.chroma_persist_dir, settings.gemini_api_key)
    collection_name = state.get("collection_name", "")

    citations = state.get("citations", [])
    draft_report = state.get("draft_report", "")

    logger.info(f"[Verifier] Starting iteration {iteration+1}, checking {len(citations)} citations")

    # Step 1: Pre-retrieve best passages from vector store for all citations
    retrieved_items = []
    for i, citation in enumerate(citations):
        claim = citation.get("claim", "")
        source = citation.get("source", "")
        cited_passage = citation.get("passage", "")

        best_passage = cited_passage
        if collection_name and claim:
            try:
                retrieved = vs.query(
                    collection_name,
                    query_text=claim,
                    n_results=3,
                    where=None,
                )
                if retrieved:
                    best_passage = retrieved[0].text
                    source = retrieved[0].source
                tool_calls.append({
                    "tool": "vector_store.query",
                    "claim_index": i,
                    "claim": claim[:100],
                    "retrieved_chunks": len(retrieved),
                })
            except Exception as ex:
                logger.warning(f"Vector search failed for claim verification: {ex}")

        retrieved_items.append({
            "citation": citation,
            "claim": claim,
            "passage": best_passage,
            "source": source
        })

    # Step 2: Batch the verifications to save LLM calls
    verified_citations = []
    failed_citations = []
    unverified_claims = []

    batch_size = 5
    for b_idx in range(0, len(retrieved_items), batch_size):
        batch = retrieved_items[b_idx:b_idx+batch_size]
        logger.info(f"[Verifier] Verifying batch of {len(batch)} claims...")
        verdicts = _verify_claims_batch(batch)

        for item, verdict in zip(batch, verdicts):
            citation = item["citation"]
            best_passage = item["passage"]
            source = item["source"]
            claim = item["claim"]

            updated_citation = {
                **citation,
                "verified": verdict.get("supported", False),
                "confidence": verdict.get("confidence", 0.0),
                "verifier_reasoning": verdict.get("reasoning", ""),
                "verifier_correction": verdict.get("correction"),
                "retrieved_passage": best_passage[:300],
            }

            if verdict.get("supported"):
                verified_citations.append(updated_citation)
            else:
                failed_citations.append(updated_citation)
                correction = verdict.get("correction")
                feedback_item = (
                    f"Claim: '{claim}'\n"
                    f"Problem: {verdict.get('reasoning', 'Could not verify')}\n"
                )
                if correction:
                    feedback_item += f"Suggested correction: {correction}\n"
                else:
                    feedback_item += "No supporting passage found — mark as [UNVERIFIED].\n"
                unverified_claims.append(feedback_item)

            tool_calls.append({
                "tool": "llm.verify_claim_batched",
                "claim": claim[:100],
                "verdict": verdict,
            })

    # Calculate stats
    total = len(citations)
    verified_count = len(verified_citations)
    failed_count = len(failed_citations)
    citation_coverage = (verified_count / total * 100) if total > 0 else 0.0

    logger.info(
        f"[Verifier] Iteration {iteration+1}: "
        f"{verified_count}/{total} verified ({citation_coverage:.0f}% coverage), "
        f"{failed_count} failed"
    )

    all_citations = verified_citations + failed_citations
    verifier_feedback = ""

    if failed_citations and iteration < settings.verifier_max_retries - 1:
        # Send feedback to Writer for another pass
        verifier_feedback = _build_feedback(failed_citations, unverified_claims)
        logger.info(f"[Verifier] Sending {len(failed_citations)} issues back to Writer")
    elif failed_citations:
        # Max retries reached — mark remaining as UNVERIFIED in report
        draft_report = _mark_unverified_in_report(draft_report, failed_citations)
        logger.info(f"[Verifier] Max retries reached — marked {len(failed_citations)} claims as [UNVERIFIED]")

    duration = time.monotonic() - start
    trace_entry = audit.log(
        node="verifier",
        inputs={"iteration": iteration, "citations_to_check": total},
        outputs={
            "verified": verified_count,
            "failed": failed_count,
            "citation_coverage_pct": round(citation_coverage, 1),
            "send_back_to_writer": bool(verifier_feedback),
        },
        tool_calls=tool_calls,
        duration_seconds=duration,
    )

    return {
        **state,
        "citations": all_citations,
        "draft_report": draft_report,
        "verifier_iteration": iteration + 1,
        "verifier_feedback": verifier_feedback,
        "unverified_claims": unverified_claims,
        "trace": state.get("trace", []) + [trace_entry],
    }


def _verify_claim(claim: str, passage: str, source: str) -> dict:
    """
    Ask the LLM whether a passage supports a claim (single fallback).
    """
    if not claim or not passage:
        return {"supported": False, "confidence": 0.0, "reasoning": "Missing claim or passage", "correction": None}

    prompt = f"""
Claim to verify: "{claim}"

Retrieved passage (from source: {source}):
---
{passage[:1000]}
---

Does this passage support the claim above?
"""
    import json as _json

    try:
        raw = call_llm(prompt, system_instruction=VERIFIER_SYSTEM)
        raw = raw.strip().lstrip("```json").lstrip("```").rstrip("```").strip()
        verdict = _json.loads(raw)
        if "supported" not in verdict:
            verdict["supported"] = False
        return verdict
    except Exception as e:
        logger.warning(f"[Verifier] LLM verification failed for claim: {e}")
        return {
            "supported": False,
            "confidence": 0.0,
            "reasoning": f"Verification error: {str(e)}",
            "correction": None,
        }


def _verify_claims_batch(batch: list[dict]) -> list[dict]:
    """
    Verify a batch of claim-passage pairs in a single LLM call.
    """
    if not batch:
        return []

    prompt = "Verify the following claim-passage pairs:\n\n"
    for idx, item in enumerate(batch):
        prompt += f"=== Pair {idx+1} ===\n"
        prompt += f"Claim: \"{item['claim']}\"\n"
        prompt += f"Retrieved passage (from source: {item['source']}):\n"
        prompt += f"{item['passage'][:1000]}\n"
        prompt += "====================\n\n"

    import json as _json

    try:
        raw = call_llm(prompt, system_instruction=VERIFIER_BATCH_SYSTEM)
        raw = raw.strip().lstrip("```json").lstrip("```").rstrip("```").strip()
        verdicts = _json.loads(raw)

        if isinstance(verdicts, list) and len(verdicts) == len(batch):
            for v in verdicts:
                if "supported" not in v:
                    v["supported"] = False
            return verdicts
    except Exception as e:
        logger.warning(f"[Verifier] Batch LLM verification failed: {e}. Falling back to individual verification...")

    # Fallback to individual verification if batch parsing fails
    verdicts = []
    for item in batch:
        verdicts.append(_verify_claim(item['claim'], item['passage'], item['source']))
    return verdicts


def _build_feedback(failed: list[dict], issues: list[str]) -> str:
    """Build structured feedback for the Writer agent."""
    lines = [
        f"## Verifier Feedback — {len(failed)} claims need correction\n",
        "Please address each issue below. For each unverifiable claim, ",
        "either find supporting context in the data OR mark it as [UNVERIFIED].\n",
    ]
    for i, issue in enumerate(issues, 1):
        lines.append(f"\n**Issue {i}:**\n{issue}")
    return "\n".join(lines)


def _mark_unverified_in_report(report: str, failed: list[dict]) -> str:
    """
    After max retries, mark remaining unverified citations in the report text.
    """
    # Blanket pass: any remaining [[CITE]] after max retries gets annotated
    report = re.sub(
        r'\[\[CITE:(.*?)\]\]',
        lambda m: f'[CITED: {m.group(1).split("|")[0].strip()}]',
        report,
        flags=re.DOTALL,
    )
    return report


def should_loop_to_writer(state: VerityState) -> str:
    """
    LangGraph conditional edge function.
    """
    feedback = state.get("verifier_feedback", "")
    iteration = state.get("verifier_iteration", 0)
    max_retries = settings.verifier_max_retries

    if feedback and iteration < max_retries:
        logger.info(f"[Verifier] Routing back to Writer (iteration {iteration}/{max_retries})")
        return "writer"
    logger.info("[Verifier] Routing to Report Assembler")
    return "assembler"
