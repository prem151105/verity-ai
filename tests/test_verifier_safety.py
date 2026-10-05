from unittest.mock import patch
from agents.verifier import verifier_node, _mark_unverified_in_report


@patch("agents.verifier.AuditLogger")
@patch("agents.verifier.call_llm")
def test_writer_passage_cannot_verify_itself(remote, audit):
    audit.return_value.log.return_value = {}
    state = {"ticker": "TEST", "run_id": "safety", "citations": [
        {"claim": "Revenue increased.", "source": "invented", "passage": "Revenue increased."}],
        "draft_report": "Revenue increased. [[CITE: invented | Revenue increased.]]",
        "verifier_iteration": 1}
    output = verifier_node(state)
    assert not output["citations"][0]["verified"]
    assert "[UNVERIFIED]" in output["draft_report"]
    remote.assert_not_called()


def test_only_failed_citation_marked_and_numbering_preserved():
    report = "Good [[CITE: A | alpha]] Bad [[CITE: B | beta]]"
    result = _mark_unverified_in_report(report, [{"source": "B", "passage": "beta"}])
    assert result == "Good [[CITE: A | alpha]] Bad [UNVERIFIED] [[CITE: B | beta]]"


@patch("agents.verifier.AuditLogger")
@patch("agents.verifier._verify_claim")
def test_remote_budget_persists_across_iterations(remote, audit):
    audit.return_value.log.return_value = {}
    remote.return_value = {"supported": False, "confidence": .99, "reasoning": "Unsupported"}
    state = {"ticker": "TEST", "run_id": "budget", "citations": [
        {"claim": "Revenue increased quickly.", "source": "source", "passage": "irrelevant"}],
        "filing_texts": [{"source": "source", "text": "Revenue declined."}],
        "verification_remote_used": 0}
    with patch("agents.verifier.settings.verification_remote_budget", 1):
        first = verifier_node(state)
        second = verifier_node(first)
    assert remote.call_count == 1
    assert second["verification_remote_used"] == 1
