import math
import pytest
from research.cascade import Cascade, Policy
from research.evidence import EvidenceIndex

DOCS = [{"source": "filing", "text": "Revenue was $120 million. The company does not guarantee growth."}]


def claim(text, source="filing"):
    return {"claim": text, "source": source}


def test_source_and_numeric_guards_never_call_remote():
    def forbidden(*args):
        pytest.fail("Guard must not invoke remote")
    result = Cascade(Policy(remote_budget=5), remote=forbidden).verify([
        claim("Revenue was $120 million.", "invented"),
        claim("Revenue was $900 million."),
    ], DOCS)
    assert all(not r["supported"] for r in result["verdicts"])
    assert result["metrics"]["remote_attempts"] == 0


def test_exact_sentence_and_negation():
    result = Cascade().verify([claim("Revenue was $120 million."),
                               claim("The company guarantees growth.")], DOCS)
    assert [r["supported"] for r in result["verdicts"]] == [True, False]
    assert result["verdicts"][0]["evidence"]["digest"]


def test_budget_and_deduplication():
    calls = []
    def remote(*args):
        calls.append(args)
        return {"supported": False, "confidence": .99, "reasoning": "Contradiction"}
    result = Cascade(Policy(remote_budget=1), remote=remote).verify([
        claim("The company guarantees growth."), claim("The company guarantees growth."),
        claim("Company growth is assured.")], DOCS)
    assert len(calls) == 1
    assert result["metrics"]["duplicate_hits"] == 1
    assert result["verdicts"][0] == result["verdicts"][1]
    assert result["verdicts"][2]["decision"] == "abstain"


@pytest.mark.parametrize("value", ["true", None, 1])
def test_malformed_remote_supported_fails_closed(value):
    result = Cascade(Policy(remote_budget=1), remote=lambda *a: {
        "supported": value, "confidence": .99, "reasoning": "Bad response"}
    ).verify([claim("The company guarantees growth.")], DOCS)
    assert result["verdicts"][0]["decision"] == "abstain"


def test_invalid_local_output_discards_partial_decisions():
    class Scorer:
        def predict(self, pairs):
            return [[.01, .98, .01], [math.nan, 0, 1]]
    result = Cascade(scorer=Scorer()).verify([
        claim("The company guarantees growth."), claim("Company growth is assured.")], DOCS)
    assert all(r["decision"] == "abstain" for r in result["verdicts"])
    assert result["metrics"]["model_errors"] == 1


def test_nli_batch_order_and_uncertainty():
    class Scorer:
        def predict(self, pairs):
            assert pairs[0][0].startswith("The company does not")
            return [[.98, .01, .01], [.2, .5, .3]]
    result = Cascade(scorer=Scorer()).verify([
        claim("The company guarantees growth."), claim("Company growth is assured.")], DOCS)
    assert [r["decision"] for r in result["verdicts"]] == ["contradicted", "abstain"]


def test_changed_document_not_cached():
    cascade = Cascade()
    assert cascade.verify([claim("Revenue was $120 million.")], DOCS)["verdicts"][0]["supported"]
    changed = [{"source": "filing", "text": "Revenue was $900 million."}]
    assert not cascade.verify([claim("Revenue was $120 million.")], changed)["verdicts"][0]["supported"]


def test_sentence_split_keeps_decimals():
    index = EvidenceIndex([{"source": "filing", "text": "Margin was 2.5%. Revenue rose."}])
    assert index.search("Margin", "filing")[0].text == "Margin was 2.5%."


def test_budget_counts_failures():
    def remote(*args):
        raise TimeoutError()
    result = Cascade(Policy(remote_budget=1), remote=remote).verify([
        claim("The company guarantees growth."), claim("Company growth is assured.")], DOCS)
    assert result["metrics"]["remote_attempts"] == 1
    assert all(r["decision"] == "abstain" for r in result["verdicts"])
