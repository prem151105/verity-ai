import asyncio
from unittest.mock import patch
from fastapi.testclient import TestClient
from api import main, store
from agents.demo import run_demo


def test_events_and_citations_survive_reload(tmp_path):
    with patch('config.settings.audit_log_dir', str(tmp_path)):
        state = run_demo()
        run = dict(run_id=state['run_id'], ticker='DEMO', status='completed', state=state)
        store.save(run)
        client = TestClient(main.app)
        result = client.get(f"/research/{state['run_id']}/events?after=20").json()
        assert [e['sequence'] for e in result['events']] == [21, 22]
        assert client.get(f"/research/{state['run_id']}/citations").json()['verified'] == 2
        assert client.get(f"/research/{state['run_id']}").json()['run_id'] == state['run_id']


def test_worker_does_not_report_failed_state_as_success(tmp_path):
    with patch('config.settings.audit_log_dir', str(tmp_path)):
        main._runs['failed-test'] = dict(run_id='failed-test', status='running')
        with patch('agents.graph.run_research', return_value={'status':'failed','error':'no evidence'}):
            asyncio.run(main._run_research_task('failed-test', 'TEST'))
        assert store.load('failed-test')['status'] == 'failed'
        main._runs.pop('failed-test')


def test_restart_marks_interrupted_and_invalid_ticker_rejected(tmp_path):
    with patch('config.settings.audit_log_dir', str(tmp_path)):
        store.save(dict(run_id='interrupted-test', status='running', state={}))
        with TestClient(main.app) as client:
            assert store.load('interrupted-test')['status'] == 'interrupted'
            assert client.post('/research/INVALID!').status_code == 422
            assert client.get('/research/missing/events').status_code == 404
