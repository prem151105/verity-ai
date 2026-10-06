"""Verity Research Observatory — an evidence-first local research workspace."""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import json
import os
import time
import requests
import streamlit as st
import streamlit.components.v1 as components
from ui.components import observatory
from agents.demo import demo_runtime
from agents.runtime import initial_state

st.set_page_config(page_title='Verity / Research Observatory', page_icon='◈', layout='wide')
API = os.environ.get('VERITY_API_URL', 'http://127.0.0.1:8000').rstrip('/')
st.markdown("""<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500&display=swap');
html,body,[class*="css"]{font-family:'DM Sans',sans-serif}
.stApp{background:#101515;color:#e9eee7}.block-container{max-width:1380px;padding-top:2.2rem}
header[data-testid="stHeader"]{background:#101515}h1,h2,h3{letter-spacing:-.035em}
[data-testid="stSidebar"]{background:#151d19;border-right:1px solid #2b382f}
[data-testid="stMetric"]{padding:18px;background:#18211b;border:1px solid #303e32;border-radius:9px}
[data-testid="stMetricLabel"]{color:#b4c1b3}div.stButton>button{border-radius:8px}
.brand{display:flex;justify-content:space-between;align-items:center;padding-bottom:22px;border-bottom:1px solid #344136;margin-bottom:36px;font:12px 'IBM Plex Mono',monospace;letter-spacing:2px}.brand strong{font-size:21px;color:#d4e6bc}.brand span{font-size:10px;color:#adbba9}
.eyebrow{font:11px 'IBM Plex Mono',monospace;letter-spacing:2px;color:#b5cf9d;margin:12px 0 18px}
.hero{font:clamp(38px,5vw,72px)/1.04 Georgia,serif;letter-spacing:-2px;margin:0 0 20px;color:#f1f0e6}.hero em{color:#bbd5a3;font-weight:400}
.subtitle{font-size:15px;line-height:1.8;color:#adbbaa;max-width:710px;margin-bottom:28px}
.note{border-left:2px solid #abc990;padding-left:20px;color:#abb9a7;font-size:13px;line-height:1.8;margin-top:24px}
.footer{border-top:1px solid #344136;padding:22px 0;color:#91a08e;font:10px 'IBM Plex Mono',monospace;letter-spacing:1px;margin-top:40px}
</style>""", unsafe_allow_html=True)

for key,value in dict(result=None, run_id=None, events=[], status='idle', demo=False).items():
    if key not in st.session_state:
        st.session_state[key] = value

@st.cache_data(ttl=5)
def api_health():
    try:
        r = requests.get(API + '/health', timeout=2)
        return r.json() if r.ok else {}
    except requests.RequestException:
        return {}

health = api_health()
with st.sidebar:
    st.markdown('### ◈ verity')
    st.caption('THE RESEARCH OBSERVATORY')
    st.divider()
    st.markdown('**Your research brief**')
    ticker = st.text_input('Company ticker', 'NVDA', max_chars=10).strip().upper()
    st.caption('SEC filings · company facts · market context')
    running = st.session_state.status in ('running','cancelling')
    launch = st.button('Begin research ↗', type='primary', use_container_width=True, disabled=running)
    demo = st.button('Explore a sample run', use_container_width=True, disabled=running)
    st.caption('No key needed. Synthetic company, real local evidence checks.')
    if health:
        st.caption('● Research service connected' + ('' if health.get('model_configured') else ' · model key needed'))
        st.caption('Document search: ' + ('local BM25' if health.get('retrieval_backend') == 'bm25' else 'Gemini embeddings'))
        st.caption('Workflow: ' + health.get('research_mode', 'fast'))
    else:
        st.caption('○ Research service offline · sample available')
    st.divider()
    st.markdown('**The research contract**')
    st.caption('01  Source every cited claim\n\n02  Challenge assumptions\n\n03  Keep uncertainty visible')
    with st.expander('Recent research runs'):
        if health:
            try:
                recent = requests.get(API + '/runs', timeout=3).json().get('runs', [])
                for r in recent[:8]:
                    if st.button(f"{r['ticker']} · {r['status']}", key=r['run_id']):
                        st.session_state.update(run_id=r['run_id'], status=r['status'], events=[], result=None, demo=False)
                        st.rerun()
            except requests.RequestException:
                st.caption('Could not fetch history.')
        else:
            st.caption('Start the API to save and revisit runs.')

st.markdown('<div class="brand"><strong>◈ VERITY</strong><span>FINANCIAL RESEARCH / OPEN TO SCRUTINY</span></div>', unsafe_allow_html=True)
a,b = st.columns([3,1])
with a:
    st.markdown('<div class="eyebrow">AN OBSERVATORY FOR EVIDENCE</div><div class="hero">Follow the evidence.<br><em>Question the conclusion.</em></div><div class="subtitle">A team of specialized agents turns company filings into research you can inspect. Watch the handoffs, challenge the assumptions, and trace cited claims back to their sources.</div>', unsafe_allow_html=True)
with b:
    st.markdown('<div class="note">EIGHT SPECIALIZED ROLES<br>One visible research process.<br><br>From the first question to the final citation, uncertainty stays in view.</div>', unsafe_allow_html=True)

if launch:
    try:
        response = requests.post(f'{API}/research/{ticker}', timeout=10)
        if response.ok:
            st.session_state.update(run_id=response.json()['run_id'], result=None, events=[], status='running', demo=False)
        else:
            st.error(response.json().get('detail', 'Could not start research.'))
    except requests.RequestException:
        st.error('Start the local research API to run live research. The sample and evidence lab work offline.')

if demo:
    st.session_state.update(result=None, events=[], demo=True, status='demo', run_id=None)
    placeholder = st.empty()
    for packet in demo_runtime().stream(initial_state('DEMO')):
        state = next(iter(packet.values()))
        with placeholder:
            components.html(observatory(state['events'], demo=True), height=580, scrolling=True)
        time.sleep(.18)
    placeholder.empty()
    st.session_state.update(result=state, events=state['events'], status=state['status'])

@st.fragment(run_every=3 if st.session_state.run_id and st.session_state.result is None else None)
def workspace():
    run_id = st.session_state.run_id
    if run_id and st.session_state.result is None:
        try:
            response = requests.get(f'{API}/research/{run_id}/events', timeout=4)
            if response.ok:
                payload = response.json()
                st.session_state.events = payload['events']
                st.session_state.status = payload['status']
                if payload['status'] == 'completed':
                    result = requests.get(f'{API}/research/{run_id}', timeout=5)
                    result.raise_for_status()
                    st.session_state.result = result.json()
                    st.rerun()
                elif payload['status'] in ('failed', 'interrupted', 'cancelled'):
                    st.warning(f"Run {payload['status']}: {payload.get('error') or 'Stopped at an agent boundary.'}")
            else:
                st.warning('The selected run could not be loaded.')
        except requests.RequestException:
            st.warning('Connection interrupted. Your run remains on the research service; retrying automatically.')
    st.markdown('#### 01 / Research observatory')
    components.html(observatory(st.session_state.events, st.session_state.demo), height=580, scrolling=True)
    if st.session_state.status in ('running','cancelling'):
        st.caption('Showing actual agent events. A model call can take several minutes.')
        if st.button('Stop after current agent', disabled=st.session_state.status == 'cancelling'):
            try:
                r = requests.post(f'{API}/research/{run_id}/cancel', timeout=4)
                r.raise_for_status()
                st.session_state.status = 'cancelling'
                st.info('Stop requested. Waiting for the current agent to finish.')
            except requests.RequestException:
                st.error('Could not deliver the stop request. Please retry.')
workspace()

result = st.session_state.result
if result:
    if result.get('error'):
        st.error(result['error'])
    citations = result.get('citations') or []
    verified = sum(bool(c.get('verified')) for c in citations)
    m1,m2,m3,m4 = st.columns(4)
    m1.metric('Cited claims supported', f'{verified} / {len(citations)}')
    m2.metric('Needs review', str(len(citations)-verified))
    m3.metric('Verification rounds', str(result.get('verifier_iteration') or sum(e['node']=='verifier' and e['kind']=='completed' for e in st.session_state.events)))
    m4.metric('Remote claim judgments', str(result.get('verification_remote_used') or 0))
    st.caption('Support is agreement with retrieved evidence, not a guarantee of truth. Uncited claims are not included in this count.')
    report_tab,evidence_tab,review_tab,trace_tab = st.tabs(['Research note','Evidence ledger','Skeptical review','Run record'])
    with report_tab:
        st.markdown(result.get('final_report') or 'No report available.')
        ratios = result.get('financial_ratios') or {}
        if ratios:
            with st.expander('Computed financial ratios'):
                st.dataframe([{'Metric': name.replace('_', ' ').title(),
                               'Value': data.get('value'), 'Source': data.get('source', '')}
                              for name, data in ratios.items()], use_container_width=True, hide_index=True)
        st.download_button('Download research note ↓', result.get('final_report',''), 'verity-research.md', 'text/markdown')
    with evidence_tab:
        selection = st.selectbox('Show claims', ['All claims','Needs review','Supported'])
        for i,c in enumerate(citations, 1):
            supported = bool(c.get('verified'))
            if selection == 'Needs review' and supported or selection == 'Supported' and not supported:
                continue
            with st.expander(f"{i:02} · {'SUPPORTED' if supported else 'NEEDS REVIEW'} · {c.get('claim','')}"):
                st.caption(c.get('source',''))
                st.write(c.get('verifier_reasoning',''))
                ev = c.get('evidence') or {}
                st.text(ev.get('text') or c.get('retrieved_passage') or 'No independent evidence retrieved.')
                st.caption('Route: ' + str(c.get('verification_route','unknown')))
                if ev.get('digest'):
                    st.code(ev['digest'], language=None)
        if not citations:
            st.info('No cited claims were checked. This does not establish report accuracy.')
    with review_tab:
        st.markdown(result.get('skeptic_review') or 'No skeptical review available.')
        st.caption('A challenge to the analysis, not a separately verified source.')
        for issue in result.get('unverified_claims') or []:
            st.warning(issue)
    with trace_tab:
        st.write('Termination:', result.get('stop_reason','—'))
        st.dataframe(st.session_state.events, use_container_width=True, hide_index=True)
        st.download_button('Export run record ↓', json.dumps(result, indent=2, default=str), 'verity-run.json', 'application/json')

st.markdown('#### 02 / Evidence lab')
with st.expander('Put a claim under the microscope', expanded=not bool(result)):
    st.caption('A local check against your supplied text. No model key, no cloud calls.')
    l,r = st.columns(2)
    with l:
        source_text = st.text_area('Source passage', 'Revenue was $120 million. Operating income was $24 million.', max_chars=50000)
    with r:
        claim = st.text_area('Claim to check', 'Revenue was $900 million.', max_chars=1000)
    if st.button('Check this claim ↗'):
        if not source_text.strip() or not claim.strip():
            st.warning('Enter a source passage and a claim.')
        else:
            from research.cascade import Cascade
            check = Cascade().verify([{'source':'Your source','claim':claim}], [{'source':'Your source','text':source_text}])
            verdict = check['verdicts'][0]
            (st.success if verdict['supported'] else st.warning)(f"{verdict['decision'].upper()} · {verdict['reasoning']}")
            st.caption('Verification route: ' + verdict['route'])
            st.json(verdict, expanded=False)
st.markdown('<div class="footer">VERITY / RESEARCH YOU CAN INSPECT &nbsp; · &nbsp; SOURCE AGREEMENT ≠ TRUTH &nbsp; · &nbsp; RESEARCH PROTOTYPE</div>', unsafe_allow_html=True)
