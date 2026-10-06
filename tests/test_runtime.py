from copy import deepcopy
from unittest.mock import patch
from agents.demo import demo_runtime
from agents.runtime import ResearchRuntime, initial_state


def test_demo_revises_numeric_error_with_real_evidence():
    states = [next(iter(p.values())) for p in demo_runtime().stream(initial_state('DEMO'))]
    checks = [s for s in states if s['events'][-1]['node'] == 'verifier']
    assert checks[0]['citations'][0]['verified'] is False
    assert checks[0]['citations'][0]['verification_route'] == 'numeric_guard'
    assert checks[1]['citations'][0]['verified'] is True
    assert checks[1]['verification_remote_used'] == 0
    assert states[-1]['status'] == 'completed'
    assert [e['sequence'] for e in states[-1]['events']] == list(range(1, 23))


def test_stagnation_ends_loop_without_overriding_verdict():
    runtime = demo_runtime()
    original_writer = runtime.nodes['writer']
    runtime.nodes['writer'] = lambda s: {**original_writer({**s, 'verifier_iteration': 0}),
                                        'verifier_iteration': s['verifier_iteration']}
    with patch('config.settings.verifier_max_retries', 5):
        state = runtime.invoke(initial_state('DEMO'))
    assert state['stop_reason'] == 'unchanged_draft'
    assert state['verifier_iteration'] == 2
    assert not state['citations'][0]['verified']
    assert '[UNVERIFIED]' in state['final_report']


def test_failure_preserves_checkpoint_and_identity():
    runtime = demo_runtime()
    def broken(s):
        raise RuntimeError('retrieval unavailable')
    runtime.nodes['retriever'] = broken
    checkpoints = []
    state = runtime.invoke(initial_state('DEMO', 'same-id'), on_event=lambda e,s: checkpoints.append(s))
    assert state['status'] == 'failed'
    assert state['run_id'] == 'same-id'
    assert state['company_name']
    assert checkpoints[-1]['events'][-1]['kind'] == 'failed'


def test_cancellation_at_boundary():
    runtime = demo_runtime()
    state = runtime.invoke(initial_state('DEMO'), cancelled=lambda: True)
    assert state['status'] == 'cancelled'
    assert [e['kind'] for e in state['events']] == ['cancelled']


def test_no_documents_stops_before_generation():
    runtime = demo_runtime()
    runtime.nodes['retriever'] = lambda s: s
    state = runtime.invoke(initial_state('DEMO'))
    assert state['status'] == 'failed'
    assert state['events'][-1]['node'] == 'retriever'


def test_hard_step_budget_and_snapshot_isolation():
    runtime = demo_runtime()
    runtime.max_steps = 1
    original = initial_state('DEMO')
    state = runtime.invoke(original)
    assert state['status'] == 'failed'
    assert original['events'] == []


def test_empty_writer_output_fails_instead_of_publishing():
    runtime = demo_runtime()
    runtime.nodes['writer'] = lambda s: {**s, 'draft_report': ''}
    state = runtime.invoke(initial_state('DEMO'))
    assert state['status'] == 'failed'
    assert state['events'][-1]['node'] == 'writer'


def test_zero_citations_are_not_reported_as_checks_complete():
    from agents.runtime import adjudicate
    assert adjudicate(initial_state('DEMO'))['stop_reason'] == 'no_cited_claims'
