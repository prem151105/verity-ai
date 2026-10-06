"""Local research service with durable snapshots and live agent events."""
import asyncio
from contextlib import asynccontextmanager
from datetime import datetime, timezone
import re
import uuid
from fastapi import FastAPI, BackgroundTasks, HTTPException
from api.models import ResearchResponse, VerificationRequest
from api import store
from config import settings

_runs = {}

@asynccontextmanager
async def lifespan(app):
    # A stopped process cannot keep executing its old tasks. Preserve checkpoints
    # for inspection and mark interruption honestly instead of pretending to resume.
    for run in store.recent():
        if run['status'] in {'running', 'cancelling'}:
            run.update(status='interrupted', error='Service restarted before completion.')
            store.save(run)
    yield

app = FastAPI(title='Verity Research Observatory', version='2.0.0', lifespan=lifespan)


def find_run(run_id):
    run = _runs.get(run_id) or store.load(run_id)
    if not run:
        raise HTTPException(404, 'Run not found')
    return run


@app.get('/health')
def health():
    return {'status': 'healthy', 'service': 'Verity', 'version': '2.0.0',
            'retrieval_backend': settings.retrieval_backend,
            'research_mode': settings.research_mode,
            'model_configured': bool(settings.gemini_api_key)}


@app.post('/verify')
def verify(request: VerificationRequest):
    from research.cascade import Cascade
    return Cascade().verify([c.model_dump() for c in request.claims],
                            [d.model_dump() for d in request.documents])


@app.post('/research/{ticker}', response_model=ResearchResponse)
async def start_research(ticker: str, background_tasks: BackgroundTasks):
    ticker = ticker.strip().upper()
    if not re.fullmatch(r'[A-Z][A-Z0-9.\-]{0,9}', ticker):
        raise HTTPException(422, 'Enter a valid ticker, such as AAPL or BRK-B.')
    if not settings.gemini_api_key:
        raise HTTPException(503, 'Configure GEMINI_API_KEY on the API server, or use the offline demo.')
    if sum(r['status'] in {'running', 'cancelling'} for r in _runs.values()) >= 2:
        raise HTTPException(429, 'Two research runs are already active.')
    run_id = str(uuid.uuid4())
    _runs[run_id] = dict(run_id=run_id, ticker=ticker, status='running',
                        started_at=datetime.now(timezone.utc).isoformat(), state={}, cancel_requested=False)
    store.save(_runs[run_id])
    background_tasks.add_task(_run_research_task, run_id, ticker)
    return ResearchResponse(run_id=run_id, ticker=ticker, company_name=ticker, status='running',
                            message='Research started. Follow the events endpoint for handoffs.')


@app.get('/research/{run_id}/events')
def events(run_id: str, after: int = 0):
    run = find_run(run_id)
    state = run.get('state') or {}
    return {'run_id': run_id, 'status': run['status'], 'error': run.get('error'),
            'events': [e for e in state.get('events', []) if e['sequence'] > after]}


@app.post('/research/{run_id}/cancel')
def cancel(run_id: str):
    run = find_run(run_id)
    if run['status'] in {'running', 'cancelling'}:
        run.update(cancel_requested=True, status='cancelling')
        _runs[run_id] = run
        store.save(run)
    return {'status': run['status'], 'message': 'Cancellation takes effect at the next agent boundary.'}


@app.get('/research/{run_id}/trace')
def trace(run_id: str):
    run = find_run(run_id)
    return {'run_id': run_id, 'trace': (run.get('state') or {}).get('trace', [])}


@app.get('/research/{run_id}/citations')
def citations(run_id: str):
    items = (find_run(run_id).get('state') or {}).get('citations', [])
    return {'run_id': run_id, 'citations': items, 'total': len(items),
            'verified': sum(bool(c.get('verified')) for c in items)}


@app.get('/research/{run_id}')
def report(run_id: str):
    run = find_run(run_id)
    if run['status'] in {'running', 'cancelling'}:
        from fastapi.responses import JSONResponse
        return JSONResponse(status_code=202, content={'status': run['status'], 'run_id': run_id})
    if run['status'] != 'completed':
        raise HTTPException(409, {'status': run['status'], 'error': run.get('error')})
    state = run['state']
    items = state.get('citations', [])
    verified = sum(bool(c.get('verified')) for c in items)
    # Evidence documents stay out of the default response; individual excerpts
    # and their digests are exposed with citations.
    keys = ('ticker', 'company_name', 'final_report', 'citations', 'unverified_claims',
            'confidence_by_section', 'financial_ratios', 'task_list', 'skeptic_review',
            'verification_metrics', 'verification_remote_used', 'events', 'stop_reason', 'trace')
    return {**{k: state.get(k) for k in keys}, 'run_id': run_id, 'status': run['status'],
            'citation_count': len(items), 'verified_citation_count': verified,
            'citation_coverage_pct': round(verified / len(items) * 100, 1) if items else 0,
            'trace_node_count': len(state.get('trace', [])), 'error': state.get('error')}


@app.get('/runs')
def runs():
    all_runs = {r['run_id']: r for r in store.recent()}
    all_runs.update(_runs)
    return {'runs': [{k: r.get(k) for k in ('run_id', 'ticker', 'status', 'started_at')}
                     for r in sorted(all_runs.values(), key=lambda r: r.get('started_at', ''), reverse=True)[:100]]}


async def _run_research_task(run_id, ticker):
    def checkpoint(event, state):
        _runs[run_id]['state'] = state
        store.save(_runs[run_id])
    try:
        from agents.graph import run_research
        state = await asyncio.to_thread(run_research, ticker, run_id, checkpoint,
                                        lambda: _runs[run_id].get('cancel_requested', False))
        _runs[run_id].update(state=state, status=state.get('status', 'failed'), error=state.get('error'))
    except Exception as exc:
        _runs[run_id].update(status='failed', error=str(exc))
    store.save(_runs[run_id])
