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

# ── Sidebar Navigation ────────────────────────────────────────────────────────
with st.sidebar:
    st.title("⚡ Verity AI")
    st.caption("Multi-Agent Financial Research")
    st.divider()
    
    page = st.radio(
        "Navigation",
        ["📖 Project Overview", "🧠 System Architecture", "⚡ Interactive Live Demo"],
        index=2
    )
    
    st.divider()
    if api_active:
        st.success("🟢 Local API Active (Port 8000)")
    else:
        st.info("☁️ Streamlit Cloud / Standalone Mode")

# ── PAGE 1: PROJECT OVERVIEW ──────────────────────────────────────────────────
if page == "📖 Project Overview":
    st.title("📖 Project Overview: Verity Financial Research")
    st.write(
        "Verity is an advanced multi-agent system built on **LangGraph** designed to automate equity research "
        "by retrieving primary SEC Edgar filings and financial facts, generating rich reports, and "
        "rigorously checking its own statements using a dedicated Verifier Agent."
    )
    
    st.subheader("📋 Table of Contents")
    st.markdown("""
    1. **[Executive Summary](#executive-summary)** — Overview of goals and problem statement.
    2. **[Multi-Agent Scaffold](#multi-agent-scaffold)** — Structure of the 6 specialized AI agents.
    3. **[Core Differentiator: The Verifier](#core-differentiator-the-verifier)** — How we eliminate LLM hallucinations.
    4. **[Financial Data Ingestion](#financial-data-ingestion)** — Details on SEC EDGAR & yfinance API clients.
    5. **[Codebase Structure](#codebase-structure)** — Folder layout and execution entry points.
    """)
    
    st.divider()
    
    st.markdown("### Executive Summary")
    st.info(
        "Traditional financial research pipelines rely on simple LLM prompts, leading to factual hallucinations "
        "and incorrect arithmetic calculations. Verity addresses this by running isolated, deterministic python code "
        "to calculate financial ratios, and verifying every written report claim against raw retrieved documents before shipping."
    )
    
    st.markdown("### Core Features")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        - **🔗 LangGraph Orchestration:** Explicit state graph control over multi-agent handoffs.
        - **📊 Deterministic Financials:** Ratios computed by isolated code executions, not LLM guess-work.
        - **🛡️ 100% Citation Audit Log:** Generates verifiable links from claims directly to SEC text ranges.
        """)
    with col2:
        st.markdown("""
        - **☁️ Cloud Standalone Ready:** Works dynamically in-process on Streamlit Cloud without separate backend servers.
        - **🌓 Dual Theme CSS System:** Responsive typography matching user light/dark preferences.
        - **🧪 Complete Evaluation Suite:** Pre-compiled results verifying accuracy across different stock tickers.
        """)

# ── PAGE 2: SYSTEM ARCHITECTURE ────────────────────────────────────────────────
elif page == "🧠 System Architecture":
    st.title("🧠 System Architecture & Deep Dive")
    st.write(
        "Below is the architectural schematic showing how Verity processes ticker inputs, fetches documents, "
        "computes ratios, writes reports, and executes the Writer ⇄ Verifier feedback loop."
    )
    
    # We display the architecture diagram
    if os.path.exists("verity-ai.png"):
        st.image("verity-ai.png", caption="Verity Multi-Agent Flowchart", use_container_width=True)
    else:
        st.info("System architecture image (verity-ai.png) not found in directory.")
        
    st.divider()
    
    st.subheader("The 6 Specialized Agents")
    
    agent_cols = st.columns(3)
    with agent_cols[0]:
        st.markdown("""
        **1. 📅 Planner Agent**
        Decomposes research tasks, maps stock tickers to central CIK identifiers, and organizes financial metrics.
        
        **2. 📥 Retriever Agent**
        Downloads raw 10-K/10-Q filings, scrapes market metrics from yfinance, and embeds text into ChromaDB.
        """)
    with agent_cols[1]:
        st.markdown("""
        **3. 📊 Analyst Agent**
        Runs deterministic python formulas to extract financial facts and compute Gross Margin, YoY Growth, and Current Ratio.
        
        **4. ✍️ Writer Agent**
        Drafts report segments and attaches a citation tag `[[CITE: source | passage]]` to every claim.
        """)
    with agent_cols[2]:
        st.markdown("""
        **5. 🔍 Verifier Agent**
        Pulls cited source text from ChromaDB and evaluates claim alignment, providing corrective feedback to the Writer.
        
        **6. 📋 Assembler Agent**
        Consolidates sections, marks unverified claims, and constructs the compliance Verification Audit Log.
        """)
        
    st.divider()
    st.subheader("The Anti-Hallucination Loop")
    st.markdown("""
    When the Writer drafts a report, it must cite specific SEC passages. The **Verifier** intercepts the draft, 
    re-queries ChromaDB for that passage independently, and evaluates whether the statement aligns. If the statement 
    is wrong (e.g. citing `$3.84` as `$5.84`), the Verifier generates detailed feedback and routes execution back to the 
    Writer for correction. If correction fails after 2 iterations, the claim is highlighted as `[UNVERIFIED]` to maintain trust.
    """)

# ── PAGE 3: INTERACTIVE LIVE DEMO ──────────────────────────────────────────────
elif page == "⚡ Interactive Live Demo":
    st.title("⚡ Interactive Research Demo")
    
    # Setup state
    if "run_id" not in st.session_state:
        st.session_state.run_id = None
    if "result" not in st.session_state:
        st.session_state.result = None
    if "ticker" not in st.session_state:
        st.session_state.ticker = None
    if "active_agent" not in st.session_state:
        st.session_state.active_agent = None
    if "logs" not in st.session_state:
        st.session_state.logs = []
        
    # Sidebar input controls
    with st.sidebar:
        st.subheader("Research Target")
        ticker_input = st.text_input(
            "US Stock Ticker",
            placeholder="AAPL, MSFT, GOOGL, NVDA...",
            value=st.session_state.ticker if st.session_state.ticker else "NVDA"
        ).upper().strip()
        
        # Action parameters
        use_in_process = not api_active
        api_key_input = ""
        user_agent_input = ""
        
        if use_in_process:
            st.info("No API backend detected. Enter keys to run in-process, or click the pre-loaded demo below.")
            api_key_input = st.text_input("Gemini API Key", type="password", value=os.environ.get("GEMINI_API_KEY", ""))
            user_agent_input = st.text_input("SEC User-Agent Header", placeholder="YourName contact@domain.com", value=os.environ.get("SEC_USER_AGENT", "VerityDemo/1.0 User@example.com"))
            
        run_btn = st.button("🔍 Run Multi-Agent System", use_container_width=True, type="primary")
        
        st.divider()
        st.subheader("Instant Pre-loaded Demo")
        load_demo_btn = st.button("🚀 Load NVIDIA Demo Report", use_container_width=True)

    # Load Demo Report
    if load_demo_btn:
        st.session_state.result = load_nvda_demo()
        st.session_state.run_id = "demo-nvda-123"
        st.session_state.ticker = "NVDA"
        st.session_state.active_agent = "Assembler"
        st.success("Loaded pre-computed NVIDIA Corporation equity research report!")

    # Run execution logic
    if run_btn and ticker_input:
        st.session_state.result = None
        st.session_state.run_id = None
        st.session_state.ticker = ticker_input
        st.session_state.logs = []
        
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

    # ── Display Results ──────────────────────────────────────────────────────────
    if st.session_state.result:
        res = st.session_state.result
        ticker = res.get("ticker", "")
        company = res.get("company_name", ticker)
        
        st.markdown(render_pipeline_trace("Assembler"), unsafe_allow_html=True)
        
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
        
        # Tabs
        tab1, tab2, tab3 = st.tabs(["📄 Final Report & Citations", "📊 Financial Ratios", "🔍 Verifier Audit Logs"])
        
        with tab1:
            st.markdown(res.get("final_report", "Report not available."))
            
        with tab2:
            ratios = res.get("financial_ratios", {})
            if ratios:
                st.subheader("📊 Computed Financial Ratios")
                st.caption("All ratios computed by deterministic Python code from SEC EDGAR XBRL data.")
                
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
                st.info("No ratios computed (insufficient XBRL data).")
                
        with tab3:
            st.subheader("🔍 Verifier Factual Verification Audit")
            unverified = res.get("unverified_claims", [])
            if unverified:
                st.warning(f"⚠️ {len(unverified)} claim(s) could not be verified:")
                for i, claim in enumerate(unverified, 1):
                    with st.expander(f"Unverified Claim {i}"):
                        st.text(claim)
            else:
                st.success("✅ All claims successfully verified against source documents!")
                
            confidence = res.get("confidence_by_section", {})
            if confidence:
                st.subheader("Confidence Score by Section")
                st.bar_chart(pd.DataFrame(list(confidence.items()), columns=["Section", "Confidence"]).set_index("Section"))
                
    elif not run_btn:
        st.info("👈 Enter a stock ticker in the sidebar and click **Run Multi-Agent System** or load the pre-loaded Nvidia report to get started!")
