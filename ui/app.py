"""
Verity — Streamlit Dashboard
Provides:
  - Ticker input → triggers research run via FastAPI
  - Live agent progress viewer (polling trace)
  - Final report display with citations as expandable excerpts
  - Financial ratio visualizations
  - Agent trace inspector
"""

import time
import json
import requests
import pandas as pd
import streamlit as st
from datetime import datetime

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Verity — Financial Research AI",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

API_BASE = "http://localhost:8000"

# ── Styling ───────────────────────────────────────────────────────────────────
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
}
.verified-badge {
    background: #065f46; color: #6ee7b7;
    padding: 2px 8px; border-radius: 999px; font-size: 0.75rem;
}
.unverified-badge {
    background: #7f1d1d; color: #fca5a5;
    padding: 2px 8px; border-radius: 999px; font-size: 0.75rem;
}
.node-card {
    border-left: 3px solid #6366f1;
    padding: 0.5rem 1rem;
    margin: 0.3rem 0;
    background: #1e293b;
    border-radius: 0 8px 8px 0;
}
.stProgress > div > div { background: linear-gradient(90deg, #6366f1, #8b5cf6); }
</style>
""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("⚡ Verity")
    st.caption("Multi-Agent Financial Research System")
    st.divider()

    ticker_input = st.text_input(
        "Stock Ticker",
        placeholder="AAPL, MSFT, GOOGL...",
        help="Enter any publicly traded US stock ticker",
    ).upper().strip()

    run_btn = st.button(
        "🔍 Generate Research Report",
        use_container_width=True,
        type="primary",
        disabled=not ticker_input,
    )

    st.divider()
    st.caption("**How it works:**")
    st.caption("1. 🗂️ **Planner** — decomposes task")
    st.caption("2. 📥 **Retriever** — fetches SEC filings")
    st.caption("3. 📊 **Analyst** — computes ratios")
    st.caption("4. ✍️ **Writer** — drafts cited report")
    st.caption("5. 🔍 **Verifier** — checks every claim")
    st.caption("6. 📋 **Assembler** — final report")

    st.divider()
    if st.button("📜 View Recent Runs"):
        try:
            runs = requests.get(f"{API_BASE}/runs", timeout=5).json()
            for r in runs.get("runs", []):
                st.caption(f"`{r['ticker']}` — {r['status']} ({r['run_id'][:8]}...)")
        except Exception:
            st.warning("API not reachable. Start the server first.")

# ── Main area ─────────────────────────────────────────────────────────────────
st.title("📊 Verity — AI Financial Research")
st.caption("Powered by LangGraph + SEC EDGAR + Gemini 2.0 Flash")

# Initialize session state
if "run_id" not in st.session_state:
    st.session_state.run_id = None
if "result" not in st.session_state:
    st.session_state.result = None
if "ticker" not in st.session_state:
    st.session_state.ticker = None

# ── Trigger research ──────────────────────────────────────────────────────────
if run_btn and ticker_input:
    st.session_state.result = None
    st.session_state.ticker = ticker_input

    with st.spinner(f"Starting research for **{ticker_input}**..."):
        try:
            resp = requests.post(f"{API_BASE}/research/{ticker_input}", timeout=10)
            if resp.status_code == 200:
                data = resp.json()
                st.session_state.run_id = data["run_id"]
                st.success(f"Research started! Run ID: `{data['run_id'][:16]}...`")
            else:
                st.error(f"API error: {resp.text}")
        except requests.exceptions.ConnectionError:
            st.error(
                "⚠️ Cannot connect to Verity API. "
                "Start it with: `uvicorn api.main:app --reload`"
            )

# ── Poll for results ──────────────────────────────────────────────────────────
if st.session_state.run_id and not st.session_state.result:
    run_id = st.session_state.run_id
    ticker = st.session_state.ticker

    progress_area = st.empty()
    status_area = st.empty()

    # Poll until complete
    max_polls = 60
    for i in range(max_polls):
        try:
            resp = requests.get(f"{API_BASE}/research/{run_id}", timeout=10)

            if resp.status_code == 202:
                # Still running — show progress
                with progress_area.container():
                    st.info(f"⏳ Researching **{ticker}**... (this takes 1–3 minutes)")
                    progress = min((i + 1) / max_polls, 0.95)
                    st.progress(progress, text=f"Running agents... ~{int(progress*100)}% estimated")

                    # Try to show trace progress
                    try:
                        trace_resp = requests.get(f"{API_BASE}/research/{run_id}/trace", timeout=5)
                        if trace_resp.status_code == 200:
                            trace = trace_resp.json().get("trace", [])
                            if trace:
                                st.caption(f"✅ Completed nodes: {', '.join(t['node'] for t in trace)}")
                    except Exception:
                        pass

                time.sleep(5)
                continue

            elif resp.status_code == 200:
                st.session_state.result = resp.json()
                progress_area.empty()
                status_area.empty()
                break

            else:
                status_area.error(f"Research failed: {resp.text}")
                break

        except requests.exceptions.ConnectionError:
            status_area.warning("Connection lost. Retrying...")
            time.sleep(3)

    else:
        status_area.warning("Research is taking longer than expected. Refresh to check.")

# ── Display results ───────────────────────────────────────────────────────────
if st.session_state.result:
    result = st.session_state.result
    ticker = result.get("ticker", "")
    company = result.get("company_name", ticker)

    # ── Summary metrics row ────────────────────────────────────────────────
    st.subheader(f"📋 {company} ({ticker}) — Research Report")
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        cov = result.get("citation_coverage_pct", 0)
        color = "🟢" if cov >= 80 else "🟡" if cov >= 60 else "🔴"
        st.metric("Citation Coverage", f"{color} {cov:.0f}%")
    with col2:
        st.metric("Citations Verified", f"{result.get('verified_citation_count', 0)}/{result.get('citation_count', 0)}")
    with col3:
        st.metric("Agent Nodes Run", result.get("trace_node_count", 0))
    with col4:
        unv = len(result.get("unverified_claims", []))
        st.metric("Unverified Claims", f"⚠️ {unv}" if unv > 0 else "✅ 0")

    st.divider()

    # ── Tabs ───────────────────────────────────────────────────────────────
    tab1, tab2, tab3, tab4 = st.tabs([
        "📄 Final Report", "📊 Financial Ratios", "🔍 Verifier Details", "🔬 Agent Trace"
    ])

    with tab1:
        st.markdown(result.get("final_report", "Report not available."))

    with tab2:
        ratios = result.get("financial_ratios", {})
        if ratios:
            st.subheader("📊 Computed Financial Ratios")
            st.caption("All ratios computed by deterministic Python code from SEC EDGAR XBRL data — not by AI.")

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

            df = pd.DataFrame(ratio_data)
            st.dataframe(df, use_container_width=True, hide_index=True)

            # Bar chart for % metrics
            pct_ratios = {
                k: v["value"] for k, v in ratios.items()
                if isinstance(v.get("value"), (int, float))
                and ("pct" in k or "margin" in k or "growth" in k)
            }
            if pct_ratios:
                st.subheader("Percentage Metrics")
                chart_df = pd.DataFrame(
                    list(pct_ratios.items()),
                    columns=["Metric", "Value (%)"]
                ).set_index("Metric")
                st.bar_chart(chart_df)
        else:
            st.info("No financial ratios computed (insufficient XBRL data for this ticker).")

    with tab3:
        st.subheader("🔍 Verifier / Claim Audit")
        unverified = result.get("unverified_claims", [])
        if unverified:
            st.warning(f"⚠️ {len(unverified)} claim(s) could not be verified:")
            for i, claim in enumerate(unverified, 1):
                with st.expander(f"Issue {i}"):
                    st.text(claim)
        else:
            st.success("✅ All claims successfully verified against source documents!")

        # Show confidence by section
        confidence = result.get("confidence_by_section", {})
        if confidence:
            st.subheader("Section Confidence Scores")
            conf_df = pd.DataFrame(
                list(confidence.items()),
                columns=["Section", "Confidence"]
            ).set_index("Section")
            st.bar_chart(conf_df)

    with tab4:
        st.subheader("🔬 Agent Execution Trace")
        st.caption("Full audit trail — every node's inputs, outputs, and tool calls.")

        run_id = st.session_state.run_id
        try:
            trace_resp = requests.get(f"{API_BASE}/research/{run_id}/trace", timeout=5)
            trace = trace_resp.json().get("trace", []) if trace_resp.ok else []
        except Exception:
            trace = []

        node_colors = {
            "planner": "#6366f1",
            "retriever": "#3b82f6",
            "analyst": "#10b981",
            "writer": "#f59e0b",
            "verifier": "#ef4444",
            "assembler": "#8b5cf6",
        }

        for entry in trace:
            node = entry.get("node", "unknown")
            duration = entry.get("duration_seconds", 0)
            color = node_colors.get(node, "#64748b")

            with st.expander(
                f"{'●'} {node.upper()} — {duration:.1f}s",
                expanded=False,
            ):
                col_a, col_b = st.columns(2)
                with col_a:
                    st.caption("**Inputs**")
                    st.json(entry.get("inputs", {}))
                with col_b:
                    st.caption("**Outputs**")
                    st.json(entry.get("outputs", {}))

                tool_calls = entry.get("tool_calls", [])
                if tool_calls:
                    st.caption(f"**Tool Calls ({len(tool_calls)})**")
                    for tc in tool_calls[:5]:
                        st.code(json.dumps(tc, indent=2), language="json")

        if not trace:
            st.info("No trace available (run still in progress or API not connected).")

# ── Landing state ─────────────────────────────────────────────────────────────
if not st.session_state.run_id and not run_btn:
    st.info(
        "👈 Enter a stock ticker in the sidebar and click **Generate Research Report** to begin.\n\n"
        "Verity will:\n"
        "- Fetch real SEC filings (10-K, 10-Q) from EDGAR\n"
        "- Compute financial ratios from XBRL data\n"
        "- Generate a cited equity research report\n"
        "- **Verify every claim against its source document**"
    )

    st.subheader("Example tickers to try")
    example_cols = st.columns(4)
    for i, example in enumerate(["AAPL", "MSFT", "GOOGL", "NVDA"]):
        example_cols[i].code(example)
