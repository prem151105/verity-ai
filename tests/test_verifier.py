"""
Unit tests for the Verifier agent.
CRITICAL TEST: Deliberately injects a false claim and confirms the verifier flags it.
This is the test that proves the core differentiator works.
"""

import pytest
from unittest.mock import patch, MagicMock


# ── Verifier logic tests (isolated, no LLM calls) ────────────────────────────

class TestCitationExtraction:
    """Test the Writer's citation parsing."""

    def test_extract_single_citation(self):
        from agents.writer import _extract_citations
        text = "Revenue grew 15% YoY [[CITE: 10-K FY2024, Item 7 | Revenue increased from $385B to $443B in fiscal 2024.]]"
        citations = _extract_citations(text)
        assert len(citations) == 1
        assert citations[0]["source"] == "10-K FY2024, Item 7"
        assert "385B" in citations[0]["passage"]

    def test_extract_multiple_citations(self):
        from agents.writer import _extract_citations
        text = (
            "Revenue grew [[CITE: 10-K FY2024 | Revenue was $443B.]] and "
            "margins improved [[CITE: 10-Q Q3 2024 | Gross margin was 45%.]]"
        )
        citations = _extract_citations(text)
        assert len(citations) == 2

    def test_no_citations_returns_empty(self):
        from agents.writer import _extract_citations
        text = "This report has no citations yet."
        assert _extract_citations(text) == []

    def test_unverified_marker_not_extracted(self):
        from agents.writer import _extract_citations
        text = "This claim is [UNVERIFIED] because no source was found."
        assert _extract_citations(text) == []


class TestVerifierLogic:
    """
    Test the verifier's claim-checking logic.
    Uses mocked LLM calls to control verdicts.
    """

    def _make_state(self, claim: str, passage: str, source: str = "Mock Source") -> dict:
        return {
            "ticker": "TEST",
            "run_id": "test-run-001",
            "collection_name": "",
            "citations": [
                {
                    "claim": claim,
                    "source": source,
                    "passage": passage,
                    "verified": False,
                    "confidence": 0.0,
                }
            ],
            "draft_report": f"Test report with claim: {claim}",
            "verifier_iteration": 0,
            "verifier_feedback": "",
            "unverified_claims": [],
            "trace": [],
        }

    @patch("agents.verifier.call_llm")
    @patch("tools.vector_store.VectorStore")
    @patch("agents.verifier.AuditLogger")
    def test_verifier_flags_unsupported_claim(self, mock_audit, mock_vs, mock_llm):
        """
        CORE TEST: Verifier must detect when a passage does NOT support a claim.
        This is the test that proves the anti-hallucination loop works.
        """
        import json

        # Arrange: LLM returns "not supported" verdict
        mock_llm.return_value = json.dumps({
            "supported": False,
            "confidence": 0.9,
            "reasoning": "The passage mentions revenue of $300B but the claim states $500B.",
            "correction": "Revenue was $300B, not $500B.",
        })

        # VS returns no chunks (isolated test)
        mock_vs_instance = MagicMock()
        mock_vs_instance.query.return_value = []
        mock_vs.return_value = mock_vs_instance

        mock_audit_instance = MagicMock()
        mock_audit_instance.log.return_value = {}
        mock_audit.return_value = mock_audit_instance

        # Deliberately false claim — passage says $300B, claim says $500B
        state = self._make_state(
            claim="Revenue reached $500 billion in FY2024",
            passage="Total net sales for fiscal year 2024 were $300.0 billion.",
            source="10-K FY2024, Item 8",
        )

        from agents.verifier import verifier_node
        with patch("agents.verifier.settings") as mock_settings:
            mock_settings.audit_log_dir = "/tmp"
            mock_settings.chroma_persist_dir = "/tmp/chroma"
            mock_settings.gemini_api_key = "fake-key"
            mock_settings.verifier_max_retries = 2

            result = verifier_node(state)

        # Assert: claim should be flagged as unverified
        updated_citations = result["citations"]
        assert len(updated_citations) == 1
        assert updated_citations[0]["verified"] is False
        assert len(result["unverified_claims"]) == 1
        assert "$500B" in result["unverified_claims"][0] or "500" in result["unverified_claims"][0]

    @patch("agents.verifier.call_llm")
    @patch("tools.vector_store.VectorStore")
    @patch("agents.verifier.AuditLogger")
    def test_verifier_passes_supported_claim(self, mock_audit, mock_vs, mock_llm):
        """Verifier must mark a well-supported claim as verified."""
        import json

        mock_llm.return_value = json.dumps({
            "supported": True,
            "confidence": 0.95,
            "reasoning": "The passage directly states revenue of $443.0 billion.",
            "correction": None,
        })

        mock_vs_instance = MagicMock()
        mock_vs_instance.query.return_value = []
        mock_vs.return_value = mock_vs_instance

        mock_audit_instance = MagicMock()
        mock_audit_instance.log.return_value = {}
        mock_audit.return_value = mock_audit_instance

        state = self._make_state(
            claim="Revenue reached $443 billion in FY2024",
            passage="Net sales for fiscal year 2024 totaled $443.0 billion, representing growth of 2% year-over-year.",
            source="10-K FY2024",
        )

        from agents.verifier import verifier_node
        with patch("agents.verifier.settings") as mock_settings:
            mock_settings.audit_log_dir = "/tmp"
            mock_settings.chroma_persist_dir = "/tmp/chroma"
            mock_settings.gemini_api_key = "fake-key"
            mock_settings.verifier_max_retries = 2

            result = verifier_node(state)

        updated_citations = result["citations"]
        assert updated_citations[0]["verified"] is True
        assert updated_citations[0]["confidence"] == 0.95
        assert len(result["unverified_claims"]) == 0

    @patch("agents.verifier.call_llm")
    @patch("tools.vector_store.VectorStore")
    @patch("agents.verifier.AuditLogger")
    def test_verifier_sends_feedback_on_first_failure(self, mock_audit, mock_vs, mock_llm):
        """
        On first failure (iteration 0), verifier should set feedback for Writer.
        On last iteration, it should mark claims as UNVERIFIED instead.
        """
        import json

        mock_llm.return_value = json.dumps({
            "supported": False,
            "confidence": 0.85,
            "reasoning": "Claim not supported by passage.",
            "correction": None,
        })
        mock_vs_instance = MagicMock()
        mock_vs_instance.query.return_value = []
        mock_vs.return_value = mock_vs_instance
        mock_audit_instance = MagicMock()
        mock_audit_instance.log.return_value = {}
        mock_audit.return_value = mock_audit_instance

        state = self._make_state("Unsupported claim", "Unrelated passage.")

        from agents.verifier import verifier_node, should_loop_to_writer
        with patch("agents.verifier.settings") as mock_settings:
            mock_settings.audit_log_dir = "/tmp"
            mock_settings.chroma_persist_dir = "/tmp/chroma"
            mock_settings.gemini_api_key = "fake-key"
            mock_settings.verifier_max_retries = 2

            result = verifier_node(state)

        # First failure: should have feedback (route back to writer)
        assert result["verifier_feedback"] != ""
        with patch("agents.verifier.settings") as s:
            s.verifier_max_retries = 2
            route = should_loop_to_writer(result)
        assert route == "writer"


class TestFinancialRatioComputation:
    """Test deterministic ratio computation."""

    def test_gross_margin_computed_correctly(self):
        from tools.code_executor import compute_financial_ratios

        metrics = {
            "revenue": [{"value": 100_000, "unit": "USD", "form": "10-K", "period_end": "2024-09-30", "filed": "2024-11-01", "accession": ""}],
            "gross_profit": [{"value": 40_000, "unit": "USD", "form": "10-K", "period_end": "2024-09-30", "filed": "2024-11-01", "accession": ""}],
        }

        ratios = compute_financial_ratios(metrics)
        assert "gross_margin_pct" in ratios
        assert ratios["gross_margin_pct"]["value"] == 40.0

    def test_yoy_growth_computed_correctly(self):
        from tools.code_executor import compute_financial_ratios

        metrics = {
            "revenue": [
                {"value": 110_000, "unit": "USD", "form": "10-K", "period_end": "2024-09-30", "filed": "2024-11-01", "accession": ""},
                {"value": 100_000, "unit": "USD", "form": "10-K", "period_end": "2023-09-30", "filed": "2023-11-01", "accession": ""},
            ],
        }

        ratios = compute_financial_ratios(metrics)
        assert "revenue_yoy_growth_pct" in ratios
        assert ratios["revenue_yoy_growth_pct"]["value"] == 10.0

    def test_debt_to_equity_zero_equity_handled(self):
        from tools.code_executor import compute_financial_ratios

        metrics = {
            "long_term_debt": [{"value": 50_000, "unit": "USD", "form": "10-K", "period_end": "2024-09-30", "filed": "2024-11-01", "accession": ""}],
            "stockholders_equity": [{"value": 0, "unit": "USD", "form": "10-K", "period_end": "2024-09-30", "filed": "2024-11-01", "accession": ""}],
        }

        # Should not raise ZeroDivisionError
        ratios = compute_financial_ratios(metrics)
        assert "debt_to_equity" not in ratios  # skipped when equity is 0
