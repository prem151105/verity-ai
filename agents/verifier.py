"""Source-anchored BACE verification inside the LangGraph rewrite loop.

Independent document evidence, optional local NLI, budgeted remote judgments,
and explicit abstention. The writer's quoted passage is never trusted.
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

def verifier_node(state: VerityState) -> VerityState:
    """
    LangGraph node: Verifier/Critic.
    Checks citations through the source-anchored verification cascade.
    """
    start = time.monotonic()
    run_id = state["run_id"]
    iteration = state.get("verifier_iteration", 0)
    audit = AuditLogger(settings.audit_log_dir, run_id)
    tool_calls = []

    from research.cascade import Cascade, Policy
    citations = state.get("citations", [])
    draft_report = state.get("draft_report", "")
    scorer = None
    if settings.verification_backend in {"torch", "onnx"}:
        try:
            from research.nli import get_scorer
            scorer = get_scorer(settings.verification_backend)
        except Exception as exc:
            logger.warning("Local verifier unavailable: %s", exc)
    budget = max(0, settings.verification_remote_budget - state.get("verification_remote_used", 0))
    outcome = Cascade(
        Policy(remote_budget=budget), scorer=scorer, remote=_verify_claim,
    ).verify(citations, state.get("filing_texts", []))
    retrieved_items = [
        {"citation": c, "claim": c.get("claim", ""),
         "passage": v.get("evidence", {}).get("text", ""),
         "source": v.get("evidence", {}).get("source", c.get("source", ""))}
        for c, v in zip(citations, outcome["verdicts"])
    ]

    # Preserve citation order while assembling the cascade judgments.
    ordered_citations = []
    verified_citations = []
    failed_citations = []
    unverified_claims = []

    for b_idx in range(0, len(retrieved_items), 5):
        batch = retrieved_items[b_idx:b_idx+5]
        verdicts = outcome["verdicts"][b_idx:b_idx+5]

        for item, verdict in zip(batch, verdicts):
            citation = item["citation"]
            best_passage = item["passage"]
            source = item["source"]
            claim = item["claim"]

            updated_citation = {
                **citation,
                "verified": verdict.get("supported", False),
                "verification_route": verdict.get("route"),
                "evidence": verdict.get("evidence"),
                "confidence": verdict.get("confidence", 0.0),
                "verifier_reasoning": verdict.get("reasoning", ""),
                "verifier_correction": verdict.get("correction"),
                "retrieved_passage": best_passage[:300],
            }

            ordered_citations.append(updated_citation)
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
                "tool": "cascade.verify_claim",
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

    all_citations = ordered_citations
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
            "cascade": outcome["metrics"],
        },
        tool_calls=tool_calls,
        duration_seconds=duration,
    )

    return {
        **state,
        "citations": all_citations,
        "draft_report": draft_report,
        "verification_remote_used": state.get("verification_remote_used", 0) + outcome["metrics"]["remote_attempts"],
        "verification_metrics": outcome["metrics"],
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
    failed_keys = {(c.get("source", "").strip(), c.get("passage", "").strip()) for c in failed}
    def annotate(match):
        source, _, passage = match.group(1).partition("|")
        if (source.strip(), passage.strip()) in failed_keys:
            return "[UNVERIFIED] " + match.group(0)
        return match.group(0)
    return re.sub(r'\[\[CITE:(.*?)\]\]', annotate, report, flags=re.DOTALL)



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
