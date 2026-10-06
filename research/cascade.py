"""BACE: Budgeted, Anchored Claim Evaluation (a project-specific cascade).

Scores are routing heuristics, not calibrated factual probabilities.
Remote budget counts attempted claim judgments, including failed requests.
"""
from dataclasses import dataclass, asdict
import math
import re
import time
import hashlib
from research.evidence import EvidenceIndex, normalize
from research.financial import numeric_compatible, field_match


@dataclass(frozen=True)
class Policy:
    threshold: float = .95
    margin: float = .20
    remote_budget: int = 0
    exact: bool = True

    def __post_init__(self):
        if not .5 < self.threshold <= 1 or not 0 <= self.margin <= 1:
            raise ValueError("Invalid decision thresholds")
        if type(self.remote_budget) is not int or self.remote_budget < 0:
            raise ValueError("remote_budget must be a nonnegative integer")


def abstain(reason, route="abstain"):
    return dict(supported=False, confidence=0.0, reasoning=reason,
                correction=None, decision="abstain", route=route)


def valid_verdict(value):
    if not isinstance(value, dict) or type(value.get("supported")) is not bool:
        return False
    score = value.get("confidence")
    return (type(score) in (int, float) and math.isfinite(score) and 0 <= score <= 1
            and isinstance(value.get("reasoning"), str))


def numbers(text):
    # Normalize literal scales, never infer a ratio or calculate a growth rate.
    # $96.221 billion and 96,221,000,000 must not trigger a false mismatch.
    from decimal import Decimal
    scales = {'billion': 10**9, 'bn': 10**9, 'b': 10**9,
              'trillion': 10**12, 'tn': 10**12, 't': 10**12,
              'million': 10**6, 'm': 10**6, 'thousand': 10**3, 'k': 10**3}
    values = set()
    for amount, suffix in re.findall(
        r"([-+]?\d[\d,]*(?:\.\d+)?)\s*(%|trillion|billion|million|thousand|tn|bn|[TBMK]\b)?", text, re.I):
        number = Decimal(amount.replace(',', '')) * scales.get(suffix.lower(), 1)
        values.add(('percent:' if suffix == '%' else '') + str(number.normalize()))
    return values


class Cascade:
    def __init__(self, policy=None, scorer=None, remote=None, remote_batch=None, cache=None):
        self.policy = policy or Policy()
        self.scorer = scorer
        self.remote = remote
        self.remote_batch = remote_batch
        self.cache = cache if cache is not None else {}

    def verify(self, claims, documents):
        started = time.perf_counter()
        index = EvidenceIndex(documents)
        results, pending = [], []
        stats = dict(claims=len(claims), exact=0, financial_fields=0, cache_hits=0,
                     local_pairs=0, remote_attempts=0, remote_requests=0,
                     duplicate_hits=0, model_errors=0)
        source_digests = {}
        source_entities = {}
        for doc in documents:
            source = doc.get('source', '').strip()
            source_digests.setdefault(source, []).append(doc.get('text', '') + repr(doc.get('entity_aliases', []))
                                                       + doc.get('retrieved_at', ''))
            source_entities.setdefault(source, []).extend(doc.get('entity_aliases', []))
        source_digests = {s: hashlib.sha256('\0'.join(parts).encode()).hexdigest()
                          for s, parts in source_digests.items()}
        cache_keys = []
        # Deduplicate only inside this invocation: evidence can never go stale across runs.
        seen, duplicates = {}, []
        for i, claim in enumerate(claims):
            text, source = claim.get("claim", ""), claim.get("source", "")
            key = (normalize(text), source)
            cache_key = hashlib.sha256(repr((key, source_digests.get(source.strip(), ''))).encode()).hexdigest()
            cache_keys.append(cache_key)
            results.append(abstain("No independently retrieved evidence"))
            if cache_key in self.cache:
                results[i] = dict(self.cache[cache_key])
                stats['cache_hits'] += 1
                continue
            if key in seen:
                duplicates.append((i, seen[key]))
                stats["duplicate_hits"] += 1
                continue
            seen[key] = i
            evidence = index.search(text, source, k=6)
            if not text.strip() or not evidence:
                continue
            exact = next((e for e in evidence if normalize(text) == normalize(e.text)), None)
            if self.policy.exact and exact:
                results[i] = dict(supported=True, confidence=1.0,
                                  reasoning="Complete sentence matches supplied source; not a truth guarantee",
                                  correction=None, decision="supported", route="exact",
                                  evidence=asdict(exact))
                stats["exact"] += 1
                continue
            field = next(((e, field_match(text, e.text, source, source_entities.get(source.strip(), []))) for e in evidence
                          if field_match(text, e.text, source, source_entities.get(source.strip(), []))), None)
            if field:
                best, reason = field
                results[i] = dict(supported=True, confidence=1.0, reasoning=reason,
                                  correction=None, decision='supported', route='financial_field',
                                  evidence=asdict(best))
                stats['financial_fields'] += 1
                continue
            # Numeric-compatible candidates beat incidental lexical overlap.
            best = next((e for e in evidence if numeric_compatible(text, e.text)), evidence[0])
            results[i]["evidence"] = asdict(best)
            # Missing numeric strings veto automatic support, not proof of contradiction.
            if not numeric_compatible(text, best.text):
                results[i]["reasoning"] = "Numeric evidence mismatch; requires review"
                results[i]["route"] = "numeric_guard"
                continue
            pending.append((i, text, best))

        unresolved = []
        if pending and self.scorer:
            try:
                scores = self.scorer.predict([(e.text, c) for _, c, e in pending])
                if len(scores) != len(pending):
                    raise ValueError("NLI response length mismatch")
                stats["local_pairs"] = len(pending)
                for (i, text, evidence), row in zip(pending, scores):
                    if len(row) != 3 or any(not math.isfinite(float(p)) or not 0 <= p <= 1 for p in row) or abs(sum(row)-1) > .01:
                        raise ValueError("Invalid NLI probabilities")
                    contradiction, support, neutral = map(float, row)
                    top = max(support, contradiction)
                    gap = top - max(neutral, min(support, contradiction))
                    if top >= self.policy.threshold and gap >= self.policy.margin:
                        results[i] = dict(supported=support > contradiction, confidence=top,
                                          reasoning="Local NLI judgment; uncalibrated model score",
                                          correction=None, decision="supported" if support > contradiction else "contradicted",
                                          route="local_nli", evidence=asdict(evidence))
                    else:
                        unresolved.append((1 - max(row), i, text, evidence))
            except Exception:
                stats["model_errors"] += 1
                # Atomic failure: discard partial model judgments.
                unresolved = []
                for i, text, evidence in pending:
                    results[i] = {**abstain("Local model unavailable or invalid output"), "evidence": asdict(evidence)}
                    unresolved.append((1.0, i, text, evidence))
        else:
            unresolved = [(1.0, i, text, evidence) for i, text, evidence in pending]

        # Most uncertain first; numeric claims get a small explicit priority bonus.
        unresolved.sort(key=lambda x: (-(x[0] + .1 * bool(numbers(x[2]))), x[1]))
        if self.remote_batch and unresolved and self.policy.remote_budget:
            selected = unresolved[:self.policy.remote_budget]
            stats['remote_attempts'] = len(selected)
            try:
                stats['remote_requests'] += 1
                verdicts = self.remote_batch([(i, text, evidence.text, evidence.source)
                                              for _, i, text, evidence in selected])
                if not isinstance(verdicts, dict):
                    raise ValueError('Invalid batch verdict envelope')
                for _, i, text, evidence in selected:
                    verdict = verdicts.get(i)
                    if valid_verdict(verdict) and verdict['confidence'] >= self.policy.threshold:
                        results[i] = {**verdict, 'decision': 'supported' if verdict['supported'] else 'unsupported',
                                      'route': 'remote_batch', 'evidence': asdict(evidence)}
                    elif valid_verdict(verdict):
                        results[i] = {**abstain('Uncertain semantic judgment: ' + verdict['reasoning'], 'remote_uncertain'),
                                      'evidence': asdict(evidence)}
                    else:
                        results[i] = {**abstain('Batch judgment missing, invalid or uncertain', 'remote_error'),
                                      'evidence': asdict(evidence)}
            except Exception:
                stats['model_errors'] += 1
                for _, i, text, evidence in selected:
                    results[i] = {**abstain('Batch judgment failed', 'remote_error'), 'evidence': asdict(evidence)}
            for _, i, text, evidence in unresolved[len(selected):]:
                results[i]['reasoning'] = 'Uncertain evidence; remote budget unavailable or exhausted'
            unresolved = []
        for _, i, text, evidence in unresolved:
            if not self.remote or stats["remote_attempts"] >= self.policy.remote_budget:
                results[i]["reasoning"] = "Uncertain evidence; remote budget unavailable or exhausted"
                continue
            stats["remote_attempts"] += 1
            stats['remote_requests'] += 1
            try:
                verdict = self.remote(text, evidence.text, evidence.source)
                if not valid_verdict(verdict):
                    raise ValueError("Invalid or uncertain remote verdict")
                if verdict['confidence'] < self.policy.threshold:
                    results[i] = {**abstain('Uncertain semantic judgment: ' + verdict['reasoning'], 'remote_uncertain'),
                                  'evidence': asdict(evidence)}
                    continue
                results[i] = {**verdict, "decision": "supported" if verdict["supported"] else "unsupported",
                              "route": "remote", "evidence": asdict(evidence)}
            except Exception:
                results[i] = {**abstain("Remote judgment failed or was uncertain", "remote_error"), "evidence": asdict(evidence)}
        for i, original in duplicates:
            results[i] = dict(results[original])
        for key, result in zip(cache_keys, results):
            if result['decision'] != 'abstain':
                self.cache[key] = dict(result)
        stats["abstentions"] = sum(r["decision"] == "abstain" for r in results)
        stats["latency_ms"] = round((time.perf_counter() - started) * 1000, 3)
        return dict(verdicts=results, metrics=stats)
