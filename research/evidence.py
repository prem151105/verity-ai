"""Source-scoped BM25 retrieval with content-addressed sentence evidence."""
from collections import Counter
from dataclasses import dataclass
import hashlib
import math
import re


def tokens(text):
    return re.findall(r"\w+", text.lower())


def normalize(text):
    return " ".join(text.lower().split())


@dataclass(frozen=True)
class Evidence:
    source: str
    text: str
    digest: str
    score: float = 0.0


class EvidenceIndex:
    """Built once per report; no embedding service or model download required."""

    def __init__(self, documents):
        self.rows = []
        seen = set()
        for doc in documents:
            source = doc.get("source", "").strip()
            # Preserve decimals; sentence boundaries require punctuation + whitespace.
            sentences = re.split(r"(?<=[.!?])\s+|\n+", doc.get("text", ""))
            # Preserve arithmetic period provenance with the preceding value.
            rows = list(sentences)
            rows.extend(sentences[i-1] + ' ' + sentence for i,sentence in enumerate(sentences)
                        if i and sentence.strip().startswith('Calculation source:'))
            for sentence in rows:
                sentence = sentence.strip()
                if not source or not sentence or (source, sentence) in seen:
                    continue
                seen.add((source, sentence))
                if doc.get('retrieved_at'):
                    sentence += f" [Snapshot retrieved: {doc['retrieved_at']}]"
                digest = hashlib.sha256((source + "\0" + sentence).encode()).hexdigest()
                self.rows.append((Evidence(source, sentence, digest), Counter(tokens(sentence))))
        self.df = Counter(t for _, counts in self.rows for t in counts)
        self.avg_len = sum(sum(c.values()) for _, c in self.rows) / max(1, len(self.rows))
        self.by_source = {}
        self.lengths = {}
        for evidence, counts in self.rows:
            self.by_source.setdefault(evidence.source, []).append((evidence, counts))
            self.lengths[evidence.digest] = sum(counts.values())

    def search(self, claim, source, k=3):
        if k < 1 or not source.strip():
            return []
        query = set(tokens(claim))
        ranked = []
        for evidence, counts in self.by_source.get(source.strip(), []):
            score = 0.0
            for term in query:
                freq = counts[term]
                idf = math.log(1 + (len(self.rows) - self.df[term] + .5) / (self.df[term] + .5))
                denom = freq + 1.5 * (.25 + .75 * self.lengths[evidence.digest] / max(self.avg_len, 1))
                score += idf * freq * 2.5 / denom
            if score > 0:
                ranked.append(Evidence(evidence.source, evidence.text, evidence.digest, score))
        return sorted(ranked, key=lambda e: (-e.score, e.digest))[:k]
