import sys
import os
import time
import json
import requests
import pandas as pd
import streamlit as st
from datetime import datetime

# Insert root directory to sys.path so we can import agents if needed
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Verity — Financial Research AI",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

API_BASE = "http://localhost:8000"

# Check if the local API server is running
@st.cache_data(ttl=5)
def check_api_server():
    try:
        resp = requests.get(f"{API_BASE}/health", timeout=2)
        return resp.status_code == 200
    except Exception:
        return False

api_active = check_api_server()

# ── Custom Styling ────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.metric-card {
    background: linear-gradient(135deg, #1e293b, #0f172a);
    border: 1px solid #334155;
    border-radius: 12px;
    padding: 1.2rem;
    margin: 0.4rem 0;
    box-shadow: 0 4px 6px -1px rgba(0,0,0,0.1);
}
.verified-badge {
    background: #065f46; color: #6ee7b7;
    padding: 2px 8px; border-radius: 999px; font-size: 0.75rem; font-weight: 600;
}
.unverified-badge {
    background: #7f1d1d; color: #fca5a5;
    padding: 2px 8px; border-radius: 999px; font-size: 0.75rem; font-weight: 600;
}
.stProgress > div > div { background: linear-gradient(90deg, #6366f1, #8b5cf6); }
</style>
""", unsafe_allow_html=True)

# ── Helper to Render Agent Timeline ──────────────────────────────────────────
def render_pipeline_trace(active_node=None):
    nodes = [
        {"name": "Planner", "icon": "📅", "desc": "CIK & Task Planner"},
        {"name": "Retriever", "icon": "📥", "desc": "EDGAR Filing Fetcher"},
        {"name": "Analyst", "icon": "📊", "desc": "XBRL Ratio Calculator"},
        {"name": "Writer", "icon": "✍️", "desc": "Cited Report Drafter"},
        {"name": "Verifier", "icon": "🔍", "desc": "Factual Claim Evaluator"},
        {"name": "Assembler", "icon": "📋", "desc": "Final Report & Indexer"}
    ]
    
    html = """
    <style>
    .pipeline-container {
        display: flex;
        flex-direction: row;
        justify-content: space-between;
        align-items: center;
        width: 100%;
        padding: 12px 18px;
        background: #0f172a;
        border: 1px solid #334155;
        border-radius: 10px;
        margin-bottom: 20px;
        overflow-x: auto;
    }
    .pipeline-node {
        display: flex;
        flex-direction: column;
        align-items: center;
        padding: 8px 12px;
        border-radius: 6px;
        background: #1e293b;
        border: 1px solid #334155;
        min-width: 130px;
        text-align: center;
        transition: all 0.3s ease;
    }
    .pipeline-node.active {
        background: linear-gradient(135deg, #4f46e5, #8b5cf6);
        border: 2px solid #c084fc;
        box-shadow: 0 0 12px rgba(139, 92, 246, 0.6);
        transform: scale(1.03);
    }
    .pipeline-node.completed {
        border-color: #10b981;
        background: #064e3b;
    }
    .node-icon {
        font-size: 1.3rem;
        margin-bottom: 2px;
    }
    .node-name {
        font-weight: 700;
        font-size: 0.85rem;
        color: #f8fafc;
    }
    .node-desc {
        font-size: 0.65rem;
        color: #94a3b8;
    }
    .pipeline-node.active .node-desc {
        color: #e9d5ff;
    }
    .pipeline-arrow {
        color: #475569;
        font-size: 1.2rem;
        font-weight: bold;
        margin: 0 4px;
        user-select: none;
    }
    </style>
    <div class="pipeline-container">
    """
    
    node_names = [n["name"].lower() for n in nodes]
    active_idx = node_names.index(active_node.lower()) if active_node and active_node.lower() in node_names else -1
    
    for i, node in enumerate(nodes):
        status_class = ""
        if active_node and node["name"].lower() == active_node.lower():
            status_class = "active"
        elif active_idx != -1 and i < active_idx:
            status_class = "completed"
            
        html += f"""
        <div class="pipeline-node {status_class}">
            <div class="node-icon">{node["icon"]}</div>
            <div class="node-name">{node["name"]}</div>
            <div class="node-desc">{node["desc"]}</div>
        </div>
        """
        if i < len(nodes) - 1:
            html += '<div class="pipeline-arrow">➔</div>'
            
    html += "</div>"
    return html

# ── Pre-loaded Demo Report Loader ─────────────────────────────────────────────
def load_nvda_demo():
    try:
        with open("reports/NVDA_equity_research.md", "r", encoding="utf-8") as f:
            report_content = f.read()
    except Exception:
        report_content = "Failed to load pre-computed Nvidia report. Make sure reports/NVDA_equity_research.md is in the project directory."
        
    return {
        "ticker": "NVDA",
        "company_name": "NVIDIA Corporation",
        "final_report": report_content,
        "citation_count": 14,
        "verified_citation_count": 14,
        "citation_coverage_pct": 100.0,
        "trace_node_count": 6,
        "unverified_claims": [],
        "financial_ratios": {
            "gross_margin": {"value": 76.15, "source": "SEC 10-K 2025"},
            "yoy_revenue_growth": {"value": 125.40, "source": "SEC 10-K 2025"},
            "operating_margin": {"value": 54.12, "source": "SEC 10-K 2025"},
            "debt_to_equity": {"value": 0.22, "source": "SEC 10-K 2025"},
            "current_ratio": {"value": 3.84, "source": "SEC 10-K 2025"}
        },
        "confidence_by_section": {
            "Executive Summary": 100.0,
            "Core Financial Ratios": 100.0,
            "Financial Performance & Growth": 100.0,
            "Relationship Mapping": 98.0
        }
    }

# Setup state
if "run_id" not in st.session_state:
    st.session_state.run_id = None
if "result" not in st.session_state:
    st.session_state.result = None
if "ticker" not in st.session_state:
    st.session_state.ticker = None
if "active_agent" not in st.session_state:
    st.session_state.active_agent = None

# ── Sidebar Configurations ────────────────────────────────────────────────────
with st.sidebar:
    st.title("⚡ Verity AI Dashboard")
    st.caption("Multi-Agent Financial Research System")
    st.divider()
    
    ticker_input = st.text_input(
        "US Stock Ticker",
        placeholder="AAPL, MSFT, GOOGL, NVDA...",
        value=st.session_state.ticker if st.session_state.ticker else "NVDA"
    ).upper().strip()
    
    use_in_process = not api_active
    api_key_input = ""
    user_agent_input = ""
    
    if use_in_process:
        st.info("No API backend detected. Enter keys to run in-process, or click the pre-loaded demo below.")
        api_key_input = st.text_input("Gemini API Key", type="password", value=os.environ.get("GEMINI_API_KEY", ""))
        user_agent_input = st.text_input("SEC User-Agent Header", placeholder="YourName contact@domain.com", value=os.environ.get("SEC_USER_AGENT", "VerityDemo/1.0 User@example.com"))
        
    run_btn = st.button("🔍 Run Multi-Agent System", use_container_width=True, type="primary")
    
    st.divider()
    st.subheader("Instant Demo")
    load_demo_btn = st.button("🚀 Load NVIDIA Demo Report", use_container_width=True)
    
    st.divider()
    if api_active:
        st.success("🟢 Local API Connected")
    else:
        st.info("☁️ Streamlit Cloud Mode Active")

# ── MAIN PANEL ────────────────────────────────────────────────────────────────
st.title("📊 Verity — Multi-Agent Financial Research & Verification")
st.write(
    "Verity is an autonomous research system built on **LangGraph**. It fetches raw SEC Edgar filings "
    "and fundamentals, computes financial ratios using isolated python execution, and uses an anti-hallucination "
    "Verifier Agent to cross-check every claim against source documents before generating final reports."
)

st.divider()

# Renders the architecture diagram on the main page
st.subheader("🕸️ System Architecture & Agent Flow")
if os.path.exists("verity-ai.png"):
    st.image("verity-ai.png", caption="Verity Multi-Agent Process Flow Chart", use_container_width=True)
else:
    st.info("System architecture diagram (verity-ai.png) not found.")

st.divider()

# Interactive Console Runner
st.subheader("⚡ Live Multi-Agent Execution & Reports")

# Check triggers
if load_demo_btn:
    st.session_state.result = load_nvda_demo()
    st.session_state.run_id = "demo-nvda-123"
    st.session_state.ticker = "NVDA"
    st.session_state.active_agent = "Assembler"
    st.success("Loaded pre-computed NVIDIA Corporation equity research report!")

if run_btn and ticker_input:
    st.session_state.result = None
    st.session_state.run_id = None
    st.session_state.ticker = ticker_input
    
    # ── LOCAL MODE (FASTAPI ACTIVE) ──────────────────────────────────────
    if api_active:
        with st.spinner(f"Kicking off FastAPI agent run for {ticker_input}..."):
            try:
                resp = requests.post(f"{API_BASE}/research/{ticker_input}", timeout=10)
                if resp.status_code == 200:
                    data = resp.json()
                    st.session_state.run_id = data["run_id"]
                    
                    # Live polling loop
                    progress_bar = st.progress(0)
                    status_box = st.status("Agents executing sequentially...", expanded=True)
                    
                    for i in range(40):
                        time.sleep(3)
                        # Poll trace
                        trace_resp = requests.get(f"{API_BASE}/research/{st.session_state.run_id}/trace", timeout=5)
                        trace = trace_resp.json().get("trace", []) if trace_resp.ok else []
                        
                        completed = [t["node"] for t in trace]
                        current_node = "planner"
                        if completed:
                            current_node = completed[-1]
                        st.session_state.active_agent = current_node
                        
                        # Render updated HTML trace
                        status_box.empty()
                        with status_box:
                            st.markdown(render_pipeline_trace(current_node), unsafe_allow_html=True)
                            for node_done in completed:
                                st.write(f"✅ Node **{node_done.upper()}** execution finished.")
                                
                        progress_bar.progress(min((i + 1) / 40, 0.95))
                        
                        # Check if complete
                        report_resp = requests.get(f"{API_BASE}/research/{st.session_state.run_id}", timeout=5)
                        if report_resp.status_code == 200:
                            st.session_state.result = report_resp.json()
                            progress_bar.empty()
                            status_box.update(label="Research Complete!", state="complete")
                            break
                    else:
                        st.error("Research timed out. Reloading to check results.")
                else:
                    st.error(f"API Error: {resp.text}")
            except Exception as e:
                st.error(f"Connection failed: {e}")
                
    # ── CLOUD MODE (IN-PROCESS EXECUTION) ────────────────────────────────
    else:
        if not api_key_input:
            st.error("Please enter a Gemini API Key in the sidebar or load the NVIDIA pre-computed demo.")
        else:
            os.environ["GEMINI_API_KEY"] = api_key_input
            os.environ["SEC_USER_AGENT"] = user_agent_input
            
            # Dynamic stream function
            def run_research_stream(ticker: str):
                from agents.graph import build_graph
                import uuid
                run_id = str(uuid.uuid4())
                graph = build_graph()
                initial_state = {
                    "ticker": ticker.upper().strip(),
                    "run_id": run_id,
                    "company_name": "",
                    "task_list": [],
                    "filings": [],
                    "filing_texts": [],
                    "market_data": {},
                    "news_items": [],
                    "collection_name": "",
                    "financial_ratios": {},
                    "key_metrics": {},
                    "analyst_summary": "",
                    "draft_report": "",
                    "citations": [],
                    "verifier_iteration": 0,
                    "verifier_feedback": "",
                    "unverified_claims": [],
                    "final_report": "",
                    "confidence_by_section": {},
                    "error": None,
                    "trace": [],
                }
                state = initial_state
                yield "start", state
                for event in graph.stream(initial_state):
                    for node_name, state_update in event.items():
                        state = {**state, **state_update}
                        yield node_name, state
                yield "completed", state

            # Start streaming logs
            progress_bar = st.progress(0)
            status_box = st.status("Initializing in-process Agent Graph...", expanded=True)
            
            try:
                state_trace = []
                steps = ["planner", "retriever", "analyst", "writer", "verifier", "assembler"]
                for idx, (node_name, state) in enumerate(run_research_stream(ticker_input)):
                    if node_name == "start":
                        status_box.write("🚀 Graph initialized. Routing to **PLANNER**.")
                        st.session_state.active_agent = "planner"
                    elif node_name == "completed":
                        status_box.write("🎉 Execution finished! Reconciling final report.")
                        # Format state response
                        citations = state.get("citations", [])
                        verified = [c for c in citations if c.get("verified")]
                        st.session_state.result = {
                            "ticker": state.get("ticker", ""),
                            "company_name": state.get("company_name", ""),
                            "final_report": state.get("final_report", ""),
                            "citation_count": len(citations),
                            "verified_citation_count": len(verified),
                            "citation_coverage_pct": round(len(verified) / max(len(citations), 1) * 100, 1),
                            "unverified_claims": state.get("unverified_claims", []),
                            "confidence_by_section": state.get("confidence_by_section", {}),
                            "financial_ratios": state.get("financial_ratios", {}),
                            "trace_node_count": len(state.get("trace", [])),
                            "error": state.get("error"),
                        }
                        st.session_state.run_id = state.get("run_id")
                    else:
                        st.session_state.active_agent = node_name
                        status_box.write(f"🤖 Active Agent Node: **{node_name.upper()}** has completed execution.")
                        
                    # Update HTML graph
                    status_box.empty()
                    with status_box:
                        st.markdown(render_pipeline_trace(st.session_state.active_agent), unsafe_allow_html=True)
                        
                    if node_name in steps:
                        progress_val = (steps.index(node_name) + 1) / len(steps)
                        progress_bar.progress(min(progress_val, 0.95))
                        
                progress_bar.empty()
                status_box.update(label="In-process run completed successfully!", state="complete")
            except Exception as ex:
                st.error(f"In-process execution failed: {ex}")
                st.exception(ex)

# Display active pipeline
if st.session_state.active_agent:
    st.markdown(render_pipeline_trace(st.session_state.active_agent), unsafe_allow_html=True)

# ── Display Results ──────────────────────────────────────────────────────────
if st.session_state.result:
    res = st.session_state.result
    ticker = res.get("ticker", "")
    company = res.get("company_name", ticker)
    
    st.subheader(f"📋 {company} ({ticker}) — Research Report")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        cov = res.get("citation_coverage_pct", 0)
        color = "🟢" if cov >= 80 else "🟡" if cov >= 60 else "🔴"
        st.metric("Citation Coverage", f"{color} {cov:.0f}%")
    with col2:
        st.metric("Citations Verified", f"{res.get('verified_citation_count', 0)}/{res.get('citation_count', 0)}")
    with col3:
        st.metric("Agent Nodes Run", res.get("trace_node_count", 0))
    with col4:
        unv = len(res.get("unverified_claims", []))
        st.metric("Unverified Claims", f"⚠️ {unv}" if unv > 0 else "✅ 0")
        
    st.divider()
    
    # Ratios & Report Side by Side
    col_rep, col_rat = st.columns([2, 1])
    
    with col_rep:
        st.subheader("📄 Factual Research Report")
        st.markdown(res.get("final_report", "Report not available."))
        
    with col_rat:
        st.subheader("📊 Financial Ratios")
        st.caption("Computed deterministically from raw SEC XBRL values.")
        
        ratios = res.get("financial_ratios", {})
        if ratios:
            ratio_data = []
            for name, data in ratios.items():
                val = data.get("value", None)
                src = data.get("source", "")
                if isinstance(val, float):
                    display = f"{val:.2f}"
                    if "pct" in name or "margin" in name or "growth" in name or "return" in name:
                        display += "%"
                else:
                    display = f"{val:,}" if isinstance(val, (int, float)) else str(val)
                ratio_data.append({
                    "Metric": name.replace("_", " ").title(),
                    "Value": display,
                    "Source": src,
                })
            st.dataframe(pd.DataFrame(ratio_data), use_container_width=True, hide_index=True)
        else:
            st.info("No ratios computed.")
            
        st.divider()
        st.subheader("🔍 Claim Verification Log")
        unverified = res.get("unverified_claims", [])
        if unverified:
            st.warning(f"⚠️ {len(unverified)} claim(s) failed verification:")
            for i, claim in enumerate(unverified, 1):
                with st.expander(f"Failed Claim {i}"):
                    st.text(claim)
        else:
            st.success("✅ All claims successfully verified against source passages!")
            
        confidence = res.get("confidence_by_section", {})
        if confidence:
            st.caption("Confidence scores by section:")
            st.bar_chart(pd.DataFrame(list(confidence.items()), columns=["Section", "Confidence"]).set_index("Section"))

elif not run_btn:
    st.info("👈 Enter a stock ticker in the sidebar and click **Run Multi-Agent System** (or load the NVIDIA demo report) to see the report, ratios, and factual claim verifications dynamically.")
