"""Bounded agent handoffs: explicit roles, external feedback, observable termination.

Paper-inspired engineering, not a reproduction of MetaGPT, Reflexion, or MAST.
"""
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import time
import uuid

NODES = ('planner', 'retriever', 'analyst', 'skeptic', 'writer', 'verifier', 'adjudicator', 'assembler')


def initial_state(ticker, run_id=None):
    return dict(ticker=ticker.strip().upper(), run_id=run_id or str(uuid.uuid4()),
                company_name='', task_list=[], filings=[], filing_texts=[], market_data={},
                news_items=[], collection_name='', financial_ratios={}, key_metrics={},
                analyst_summary='', skeptic_review='', draft_report='', citations=[],
                verifier_iteration=0, verifier_feedback='', unverified_claims=[],
                verification_remote_used=0, verification_metrics={}, final_report='',
                verification_cache={},
                confidence_by_section={}, error=None, trace=[], events=[], review_history=[])


def default_nodes():
    from agents.planner import planner_node
    from agents.retriever import retriever_node
    from agents.analyst import analyst_node
    from agents.skeptic import skeptic_node
    from agents.writer import writer_node
    from agents.verifier import verifier_node
    from agents.assembler import assembler_node
    return dict(planner=planner_node, retriever=retriever_node, analyst=analyst_node,
                skeptic=skeptic_node, writer=writer_node, verifier=verifier_node,
                adjudicator=adjudicate, assembler=assembler_node)


def adjudicate(state):
    """Stop identical drafts and exhausted revisions; never override a verdict."""
    from config import settings
    fingerprint = hashlib.sha256(state.get('draft_report', '').encode()).hexdigest()
    history = state.get('review_history', [])
    issues = state.get('unverified_claims', [])
    repeat = fingerprint in history
    rewrite = bool(state.get('verifier_feedback')) and not repeat and state.get('verifier_iteration', 0) < settings.verifier_max_retries
    reason = ('no_cited_claims' if not state.get('citations') else
              'unchanged_draft' if repeat else 'revision_requested' if rewrite else
              'review_limit' if issues and state.get('verifier_iteration', 0) >= settings.verifier_max_retries else
              'uncertain_claims' if issues else 'checks_complete')
    draft = state.get('draft_report', '')
    if not rewrite and issues:
        from agents.verifier import _mark_unverified_in_report
        draft = _mark_unverified_in_report(draft.replace('[UNVERIFIED] [[CITE:', '[[CITE:'),
                                          [c for c in state.get('citations', []) if not c.get('verified')])
    return {**state, 'review_history': history + [fingerprint], 'next_node': 'writer' if rewrite else 'assembler',
            'stop_reason': reason, 'draft_report': draft}


class ResearchRuntime:
    def __init__(self, nodes=None, max_steps=32):
        self.nodes = nodes if nodes is not None else default_nodes()
        self.max_steps = max_steps

    def stream(self, initial, on_event=None, cancelled=None):
        state = deepcopy(initial)
        node = 'planner'

        def emit(kind, name, detail='', duration=0):
            event = dict(sequence=len(state['events']) + 1, node=name, kind=kind,
                         detail=detail, duration_seconds=round(duration, 3),
                         timestamp=datetime.now(timezone.utc).isoformat())
            state['events'] = state['events'] + [event]
            if on_event:
                on_event(event, deepcopy(state))

        state.setdefault('events', [])
        for _ in range(self.max_steps):
            if cancelled and cancelled():
                state['status'] = 'cancelled'
                emit('cancelled', node, 'Stopped before the next agent handoff.')
                yield {'cancelled': deepcopy(state)}
                return
            emit('started', node, f'{node.title()} is working')
            start = time.monotonic()
            try:
                update = self.nodes[node](deepcopy(state))
                if not isinstance(update, dict):
                    raise TypeError(f'{node} must return a state mapping')
                if update.get('run_id', state['run_id']) != state['run_id']:
                    raise ValueError('Agent attempted to change run identity')
                events = state['events']
                state.update(update)
                state['events'] = events
                if state.get('error'):
                    raise RuntimeError(state['error'])
                if node == 'retriever' and not state.get('filing_texts'):
                    raise RuntimeError('No independent documents retrieved; report generation stopped.')
                if node == 'writer' and not state.get('draft_report', '').strip():
                    raise RuntimeError('Writer returned an empty report.')
                if node == 'assembler' and not state.get('final_report', '').strip():
                    raise RuntimeError('Assembler returned an empty report.')
                detail = f'{node.title()} handoff complete'
                if node == 'adjudicator':
                    detail = state.get('stop_reason', '')
                elif node == 'verifier':
                    claims = state.get('citations', [])
                    supported = sum(bool(c.get('verified')) for c in claims)
                    detail = f'{supported}/{len(claims)} cited claims supported; {len(claims)-supported} need review'
                emit('completed', node, detail, time.monotonic() - start)
            except Exception as exc:
                state.update(status='failed', error=str(exc))
                emit('failed', node, str(exc), time.monotonic() - start)
                yield {node: deepcopy(state)}
                return
            if node == 'assembler':
                state['status'] = 'completed'
            yield {node: deepcopy(state)}
            if node == 'assembler':
                return
            node = state['next_node'] if node == 'adjudicator' else NODES[NODES.index(node) + 1]
        state.update(status='failed', error='Agent step budget exhausted')
        emit('failed', node, state['error'])
        yield {'budget': deepcopy(state)}

    def invoke(self, initial, **kwargs):
        state = initial
        for event in self.stream(initial, **kwargs):
            state = next(iter(event.values()))
        return state
