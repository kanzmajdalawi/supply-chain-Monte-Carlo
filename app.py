"""
app.py — Supply Chain Monte Carlo Risk Analyzer
================================================
Streamlit live demo. Run with:

    streamlit run app.py

Or deploy to Streamlit Cloud by pushing to GitHub and connecting
the repo at https://share.streamlit.io
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

from sim.inventory import SimParams, run
from analysis.risk import risk_metrics, convergence_study, run_scenarios

# ── Page config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Supply Chain MC Simulator",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
  .metric-card {
    background: #1e1e2e; border-radius: 10px; padding: 16px;
    border: 1px solid #313244;
  }
  .risk-low  { color: #a6e3a1; }
  .risk-med  { color: #f9e2af; }
  .risk-high { color: #f38ba8; }
</style>
""", unsafe_allow_html=True)

# ── Sidebar: Parameters ────────────────────────────────────────────────────
with st.sidebar:
    st.title("⚙️ Parameters")

    st.subheader("Demand")
    mean_demand  = st.slider("Mean daily demand (units)", 50, 500, 200, step=10)
    demand_cv    = st.slider("Demand CV (σ/μ)", 0.05, 0.60, 0.20, step=0.05)
    spike_prob   = st.slider("Spike probability", 0.0, 0.20, 0.05, step=0.01)
    spike_mult   = st.slider("Spike multiplier", 1.5, 5.0, 2.5, step=0.5)

    st.subheader("Supplier")
    fail_prob      = st.slider("Failure probability", 0.01, 0.30, 0.08, step=0.01)
    lead_mean      = st.slider("Mean lead time (days)", 3, 30, 10)
    lead_k         = st.slider("Lead time shape k (Gamma)", 1, 10, 4)
    n_suppliers    = st.slider("Number of suppliers", 1, 5, 2)
    supplier_corr  = st.slider("Supplier correlation ρ", 0.0, 0.9, 0.3, step=0.1)

    st.subheader("Inventory Policy (s,Q)")
    rop      = st.slider("Reorder point (ROP)", 100, 3000, 800, step=50)
    eoq      = st.slider("Order quantity (EOQ)", 200, 5000, 1500, step=100)
    init_inv = st.slider("Initial inventory", 500, 5000, 2000, step=100)
    sim_days = st.slider("Simulation horizon (days)", 30, 365, 90, step=10)

    st.subheader("Monte Carlo")
    n_paths  = st.select_slider("Iterations (N)", [1000, 2500, 5000, 10000, 25000], 5000)
    seed_val = st.number_input("Random seed (0 = random)", 0, 9999, 42)

    run_btn  = st.button("▶ Run Simulation", type="primary", use_container_width=True)

# ── Build params ───────────────────────────────────────────────────────────
params = SimParams(
    mean_demand=mean_demand,
    demand_cv=demand_cv,
    spike_prob=spike_prob,
    spike_mult=spike_mult,
    fail_prob=fail_prob,
    lead_mean=lead_mean,
    lead_k=lead_k,
    n_suppliers=n_suppliers,
    supplier_corr=supplier_corr,
    rop=rop,
    eoq=eoq,
    init_inv=init_inv,
    sim_days=sim_days,
    n_paths=n_paths,
    seed=seed_val if seed_val > 0 else None,
)

# ── Safety stock callout ───────────────────────────────────────────────────
ss_95 = params.safety_stock(0.95)
rec_rop = params.recommended_rop(0.95)
st.sidebar.info(
    f"**Recommended ROP** (95% SL): **{rec_rop:.0f}** units\n\n"
    f"Safety stock: **{ss_95:.0f}** units"
)

# ── Header ─────────────────────────────────────────────────────────────────
st.title("📦 Supply Chain Monte Carlo Risk Analyzer")
st.caption(
    "Simulates multi-supplier, stochastic-demand inventory systems. "
    "Demand: Compound Poisson Log-Normal · Supplier failures: Gaussian Copula · "
    "Lead time: Gamma distribution"
)

# ── Run simulation ─────────────────────────────────────────────────────────
if run_btn or "result" not in st.session_state:
    with st.spinner(f"Running {n_paths:,} Monte Carlo paths over {sim_days} days…"):
        result = run(params)
        st.session_state["result"] = result
        st.session_state["params"] = params

result = st.session_state["result"]
summary = result.summary()

# ── KPI row ───────────────────────────────────────────────────────────────
c1, c2, c3, c4, c5 = st.columns(5)
sl = summary["service_level"]
sl_color = "normal" if sl > 95 else "inverse"

c1.metric("Service Level",    f"{sl:.1f}%",        delta="Target: 95%", delta_color=sl_color)
c2.metric("Stockout Prob.",   f"{summary['stockout_probability']:.1f}%")
c3.metric("Median Final Inv", f"{summary['median_final_inventory']:,.0f} units")
c4.metric("VaR 95 (cost)",    f"${summary['var_95']:,.0f}")
c5.metric("CVaR 95 (cost)",   f"${summary['cvar_95']:,.0f}")

st.divider()

# ── Tabs ──────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs(
    ["📊 Distributions", "📈 Time Series", "⚠️ Scenarios & Sensitivity", "🔬 Benchmarks"]
)

# ── Tab 1: Distributions ──────────────────────────────────────────────────
with tab1:
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Final inventory distribution")
        fig = px.histogram(
            result.final_inventory, nbins=40,
            labels={"value": "Inventory (units)", "count": "Paths"},
            color_discrete_sequence=["#89b4fa"],
        )
        p5 = np.percentile(result.final_inventory, 5)
        fig.add_vline(x=p5, line_dash="dash", line_color="#f38ba8",
                      annotation_text=f"P5: {p5:.0f}")
        fig.update_layout(showlegend=False, height=300, margin=dict(t=20))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Total cost distribution (VaR)")
        costs = result.total_cost
        fig2 = go.Figure()
        var95_val = result.var95
        colors = ["#a6e3a1" if c < var95_val else "#f38ba8" for c in np.histogram(costs, bins=40)[0]]
        fig2.add_trace(go.Histogram(x=costs, nbinsx=40, name="Cost",
                                    marker_color="#89b4fa"))
        fig2.add_vline(x=var95_val, line_dash="dash", line_color="#f38ba8",
                       annotation_text=f"VaR 95: ${var95_val:,.0f}")
        fig2.add_vline(x=result.cvar95, line_dash="dot", line_color="#f9e2af",
                       annotation_text=f"CVaR: ${result.cvar95:,.0f}")
        fig2.update_layout(showlegend=False, height=300, margin=dict(t=20))
        st.plotly_chart(fig2, use_container_width=True)

    st.subheader("Risk metrics table")
    rm = risk_metrics(result.total_cost)
    df_rm = pd.DataFrame([rm]).T.rename(columns={0: "Value"})
    df_rm["Value"] = df_rm["Value"].map(lambda x: f"${x:,.2f}")
    st.dataframe(df_rm, use_container_width=True)

# ── Tab 2: Time Series ────────────────────────────────────────────────────
with tab2:
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Stockout probability over time")
        fig = go.Figure()
        days = list(range(1, sim_days + 1))
        fig.add_trace(go.Scatter(
            x=days, y=result.daily_stockout_prob * 100,
            fill="tozeroy", line=dict(color="#f38ba8"),
            name="Stockout prob (%)",
        ))
        fig.update_layout(yaxis_title="Probability (%)", xaxis_title="Day",
                          height=280, margin=dict(t=20))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Mean inventory level over time")
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(
            x=days, y=result.daily_mean_inventory,
            fill="tozeroy", line=dict(color="#89dceb"),
            name="Mean inventory",
        ))
        fig2.add_hline(y=rop, line_dash="dash", line_color="#f9e2af",
                       annotation_text=f"ROP: {rop}")
        fig2.update_layout(yaxis_title="Units", xaxis_title="Day",
                            height=280, margin=dict(t=20))
        st.plotly_chart(fig2, use_container_width=True)

# ── Tab 3: Scenarios ──────────────────────────────────────────────────────
with tab3:
    st.subheader("Stress scenarios — service level impact")
    with st.spinner("Running 6 stress scenarios…"):
        scenarios = run_scenarios(params, run)

    df_sc = pd.DataFrame([
        {"Scenario": k.replace("_", " ").title(),
         "Service Level (%)": v["service_level"],
         "Stockout Prob (%)": v["stockout_probability"],
         "VaR 95": f"${v['var_95']:,.0f}"}
        for k, v in scenarios.items()
    ])
    st.dataframe(df_sc, use_container_width=True)

    fig_sc = px.bar(
        df_sc, x="Service Level (%)", y="Scenario", orientation="h",
        color="Service Level (%)", color_continuous_scale="RdYlGn",
        range_color=[50, 100],
    )
    fig_sc.update_layout(height=300, margin=dict(t=20), showlegend=False)
    st.plotly_chart(fig_sc, use_container_width=True)

# ── Tab 4: Benchmarks ────────────────────────────────────────────────────
with tab4:
    st.subheader("Convergence study — accuracy vs. iteration count")
    with st.spinner("Running convergence benchmark…"):
        conv = convergence_study(params, run, sizes=[100, 250, 500, 1000, 2500, 5000])

    df_conv = pd.DataFrame(conv)
    st.dataframe(df_conv, use_container_width=True)

    fig_conv = make_subplots(rows=1, cols=2,
        subplot_titles=("95% CI Width vs N", "Runtime (ms) vs N"))
    fig_conv.add_trace(go.Scatter(x=df_conv["n"], y=df_conv["ci_width"],
        mode="lines+markers", name="CI width", line=dict(color="#89b4fa")), row=1, col=1)
    fig_conv.add_hline(y=1.0, line_dash="dash", line_color="#f38ba8",
                       annotation_text="Target 1%", row=1, col=1)
    fig_conv.add_trace(go.Bar(x=df_conv["n"], y=df_conv["runtime_ms"],
        name="Runtime", marker_color="#a6e3a1"), row=1, col=2)
    fig_conv.update_layout(height=320, margin=dict(t=40), showlegend=False)
    st.plotly_chart(fig_conv, use_container_width=True)

    st.info(
        "**Rule of thumb:** N ≥ 5,000 achieves CI width < 1% for most "
        "supply chain scenarios. N = 10,000 recommended for production decisions."
    )
