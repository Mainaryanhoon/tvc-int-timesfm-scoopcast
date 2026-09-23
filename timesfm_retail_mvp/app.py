from __future__ import annotations
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
import streamlit as st

from config import SCENARIOS
from data.data_manager import load_csv, series_options
from forecast.timesfm3_engine import TimesFM3Engine
from scenarios.retail import forecast_retail
from scenarios.viewership import forecast_viewership
from scenarios.inventory import forecast_inventory

# Import single-source architecture & executive UI components
from forecast.architecture import build_forecast_result
from dashboard.executive_summary import render_executive_summary
from dashboard.charts import render_business_forecast_chart, render_inventory_chart
from dashboard.pdf_generator import generate_pdf_report


# ---------------------------------------------------------------------
# Page & Custom CSS Styling
# ---------------------------------------------------------------------
st.set_page_config(
    page_title="ScoopCast — Demand Planner",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
.block-container { max-width: 1440px; padding-top: 1.25rem; padding-bottom: 4rem; }
.scoop-hero { padding: 1.25rem 1.5rem; border: 1px solid #e2e8f0; border-radius: 16px; background: #ffffff; margin-bottom: 1.25rem; }
.scoop-eyebrow { font-size: .72rem; letter-spacing: .12em; font-weight: 700; color: #d93662; text-transform: uppercase; }
.scoop-title { font-size: 2.1rem; font-weight: 800; color: #0f172a; line-height: 1.1; margin-top: .2rem; }
.scoop-subtitle { color: #64748b; font-size: .92rem; margin-top: .3rem; }
.insight-card { padding: 1rem 1.1rem; border: 1px solid #e2e8f0; border-radius: 14px; background: #ffffff; min-height: 105px; }
.insight-title { font-size: .76rem; font-weight: 700; color: #64748b; text-transform: uppercase; letter-spacing: .06em; }
.insight-value { font-size: 1.15rem; font-weight: 700; color: #0f172a; margin-top: .35rem; }
.muted { color: #64748b; font-size: .88rem; }
</style>
""",
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def get_device() -> str:
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    return "cpu"


def clear_result_if_context_changed(context):
    previous = st.session_state.get("forecast_context")
    if previous != context:
        st.session_state.pop("result", None)
        st.session_state.pop("forecast_obj", None)
        st.session_state["forecast_context"] = context


# ---------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------
st.markdown(
    """
<div class="scoop-hero">
    <div class="scoop-eyebrow">Enterprise Demand Planner</div>
    <div class="scoop-title">SCOOPCAST</div>
    <div class="scoop-subtitle">
        Actionable business forecasting without technical complexity.
    </div>
</div>
""",
    unsafe_allow_html=True,
)

mode = st.radio("Forecast scenario", list(SCENARIOS), horizontal=True, label_visibility="collapsed")
cfg = SCENARIOS[mode]


# ---------------------------------------------------------------------
# Sidebar (Business Setup & Context Controls)
# ---------------------------------------------------------------------
with st.sidebar:
    st.markdown("### Data Input")
    
    st.markdown("Upload your historical daily sales data.")
    upload = st.file_uploader("Upload CSV", type=["csv"], label_visibility="collapsed", key=f"upload_{mode}")
    with st.expander("Expected format"):
        st.markdown("**Required:** `date`, `target_value`\n\n**Optional:** `promotion` (0 or 1), `planned_production`")
    
    path = None
    if upload is not None:
        upload_dir = Path(".scoopcast_uploads")
        upload_dir.mkdir(exist_ok=True)
        path = str(upload_dir / upload.name)
        Path(path).write_bytes(upload.getbuffer())
        st.caption(f"Using external file: `{upload.name}`")
    else:
        st.markdown("---")
        st.markdown("### Built-in Dataset")
        st.markdown(f"**{cfg.company}**\n\nSample dataset for exploring the workflow.")

    try:
        df = load_csv(mode, path)
    except Exception as exc:
        st.error(str(exc))
        st.stop()
        
    # Clean Raw Data Preview
    if upload is None and st.checkbox("View raw data", value=False):
        clean_preview = df.copy()
        if "date" in clean_preview.columns:
            clean_preview["date"] = pd.to_datetime(clean_preview["date"]).dt.strftime("%Y-%m-%d")
        
        target_col = getattr(cfg, 'target_col', None)
        if target_col and target_col in clean_preview.columns:
            clean_preview = clean_preview.dropna(subset=[target_col])
            
        st.dataframe(clean_preview, use_container_width=True)

    opts = series_options(df, mode)
    st.divider()
    
    # Target Selection & Horizon
    st.markdown("### Forecast Setup")
    target_label = "Units Demand" if mode == "Inventory" else "Viewers" if mode == "Viewership" else "Sales"
    
    if mode == "Inventory":
        series = [st.selectbox(f"Target ({target_label})", opts, key="inv_component")]
    else:
        default_series = opts[: min(1, len(opts))]
        series = st.multiselect(f"Target ({target_label})", opts, default=default_series, key=f"series_{mode}")
        if not series:
            st.warning("Select at least one target.")
            st.stop()

    horizon = st.slider("Forecast period (Days)", min_value=7, max_value=90, value=50, key=f"horizon_{mode}")
    
    st.divider()

    # Dynamic Business Signals Section
    st.markdown("### Business Signals")
    st.caption("Context provided to the forecast engine.")

    # Known Future Signals
    future_opts = [cfg.signal_labels.get(c, c) for c in getattr(cfg, 'future_signals', [])]
    selected_future_labels = []
    active_future_signals = []
    
    if future_opts:
        selected_future_labels = st.multiselect(
            "Known Future Context",
            options=future_opts,
            default=future_opts,
            key=f"future_signals_{mode}",
        )
        label_to_col = {v: k for k, v in cfg.signal_labels.items()}
        active_future_signals = [label_to_col.get(lbl, lbl) for lbl in selected_future_labels]

    # Historical Context Signals (Tooltips & Clean Headers)
    past_opts = [cfg.signal_labels.get(c, c) for c in getattr(cfg, 'past_only_signals', [])]
    if past_opts:
        st.markdown("**Historical Context**")
        for c in past_opts:
            if "Lead Time" in c:
                st.markdown(f"🔹 {c}", help="Standard transit time required for components to arrive from the supplier. Helps the model gauge replenishment cycles.")
            elif "Disruption" in c:
                st.markdown(f"🔹 {c}", help="Historical logs of supply chain delays and logistical bottlenecks.")
            else:
                st.markdown(f"🔹 {c}")

    st.divider()

    # What-If Scenario Planner
    p_start, p_end, p_int = None, None, 1.0
    what_if_event_str = None
    
    if mode in ("Sales", "Revenue"):
        st.markdown("### What-If Scenario")
        st.markdown("What would you like to change?")
        enable_what_if = st.checkbox("Event: Promotion", value=False, key=f"what_if_{mode}")
        if enable_what_if:
            p_range = st.slider("Promotion window (Days out)", 1, horizon, (5, 12), key=f"p_range_{mode}")
            p_start, p_end = p_range[0] - 1, p_range[1] - 1
            
            p_str = st.select_slider("Expected strength", options=["Low", "Medium", "High"], value="Medium", key=f"p_str_{mode}")
            p_int = {"Low": 1.2, "Medium": 1.5, "High": 2.0}[p_str]
            what_if_event_str = f"Simulated Promotion Overlay (Days {p_range[0]} to {p_range[1]}, Strength: {p_str})"


# Combine active events for the "Why" Engine
active_events_for_engine = selected_future_labels + past_opts
if what_if_event_str:
    active_events_for_engine.append(what_if_event_str)


# ---------------------------------------------------------------------
# Context Guard
# ---------------------------------------------------------------------
upload_id = upload.name if upload is not None else "__demo__"
context = (mode, upload_id, tuple(series), int(horizon), tuple(active_future_signals), p_start, p_end, p_int)
clear_result_if_context_changed(context)

st.markdown(f"### {cfg.name} forecast · {cfg.company}")
st.markdown(f'<div class="muted">Target: <b>{series[0]}</b> &nbsp;|&nbsp; Forecast period: <b>{horizon} days</b></div>', unsafe_allow_html=True)
generate = st.button("Generate Forecast", type="primary")


# ---------------------------------------------------------------------
# Forecast Execution Engine
# ---------------------------------------------------------------------
if generate:
    try:
        with st.spinner("Analyzing data and generating forecast..."):
            if st.session_state.get("engine") is None:
                device = get_device()
                st.session_state.engine_device = device
                st.session_state.engine = TimesFM3Engine(device=device, batch_size=1 if device == "cpu" else 8)

            engine = st.session_state.engine
            df_active = df.copy()

            # Zero-out disabled future signals before inference
            if hasattr(cfg, 'future_signals'):
                disabled_signals = [col for col in cfg.future_signals if col not in active_future_signals]
                for col in disabled_signals:
                    if col in df_active.columns:
                        df_active[col] = 0.0

            # Execute model scenarios
            if mode in ("Sales", "Revenue"):
                raw_result = forecast_retail(df_active, mode, series, horizon, engine, p_start, p_end, p_int)
            elif mode == "Viewership":
                raw_result = forecast_viewership(df_active, series, horizon, engine)
            else:
                bom_path = Path("data/automotive_bom_reference.csv")
                bom = pd.read_csv(bom_path) if bom_path.exists() else pd.DataFrame()
                raw_result = forecast_inventory(df_active, bom, series[0], horizon, engine)

            # Transform raw TimesFM outputs into single source of truth
            target_idx = 0
            if mode == "Inventory":
                base = raw_result["forecast"]
                p50_scen = None
                current_stock = raw_result["current_inventory"]
            else:
                base = raw_result["base"]
                p50_scen = raw_result["scenario"].point[target_idx] if raw_result.get("has_what_if") else None
                current_stock = 0.0

            forecast_obj = build_forecast_result(
                domain=mode,
                target_name=series[target_idx],
                raw_dates=pd.DatetimeIndex(base.dates),
                p50_baseline=base.point[target_idx],
                p10_baseline=base.quantiles[target_idx, :, 0],
                p90_baseline=base.quantiles[target_idx, :, 8],
                horizon_days=horizon,
                p50_scenario=p50_scen,
                current_stock=current_stock,
                lead_time_days=14
            )

            st.session_state.forecast_obj = forecast_obj
            st.session_state.active_events = active_events_for_engine
            st.session_state.hist_df = df_active

    except Exception as exc:
        st.error(f"Forecast failed: {exc}")
        st.code(traceback.format_exc(), language="python")


# ---------------------------------------------------------------------
# UI Rendering
# ---------------------------------------------------------------------
if "forecast_obj" not in st.session_state:
    st.markdown('<div class="insight-card"><div class="insight-title">Ready</div><div class="insight-value">Generate your forecast</div><div class="muted">Upload data and configure the setup in the sidebar to begin.</div></div>', unsafe_allow_html=True)
    st.stop()

forecast = st.session_state.forecast_obj
hist_df = st.session_state.hist_df
events = st.session_state.active_events

# 1. EXECUTIVE SUMMARY
st.markdown("---")
render_executive_summary(forecast, historical_trend_pct=9.6, active_events=events)

# 2. FORECAST OUTLOOK
st.markdown("---")
st.markdown("### FORECAST OUTLOOK")
view_toggle = st.radio("View", ["Daily", "Monthly"], horizontal=True, label_visibility="collapsed")

if mode == "Inventory":
    fig = render_inventory_chart(forecast, view_mode=view_toggle)
else:
    unit_label = "Revenue" if mode == "Revenue" else "Viewers" if mode == "Viewership" else "Units"
    fig = render_business_forecast_chart(forecast, hist_df=hist_df, hist_date_col=cfg.date_col, hist_target_col=cfg.target_col, view_mode=view_toggle, unit_label=unit_label)

st.plotly_chart(fig, use_container_width=True)

# 3. DETAIL REPORT
st.markdown("### DETAIL REPORT")

if mode == "Inventory":
    st.info(
        "**How to read this table:** `Opening Stock` - `Expected Demand` = `Projected Closing Stock`. "
        "Negative closing stock indicates projected backorders if no new inventory is received.", 
        icon="ℹ️"
    )
    rename_dict = {
        "date": "Date",
        "opening_stock": "Opening Stock",
        "expected": "Expected Customer Demand",
        "expected_closing_stock": "Projected Closing Stock",
        "safety_buffer": "Safety Buffer Threshold"
    }
    display_df = forecast.daily_forecast.rename(columns=rename_dict)
    cols_to_show = ["Date", "Opening Stock", "Expected Customer Demand", "Projected Closing Stock", "Safety Buffer Threshold"]
    display_df = display_df[cols_to_show]
else:
    rename_dict = {
        "date": "Date",
        "month": "Month",
        "expected": "Expected Demand",
        "lower": "Conservative (P10)",
        "upper": "Optimistic (P90)"
    }
    display_df = forecast.monthly_summary.rename(columns=rename_dict) if view_toggle == "Monthly" else forecast.daily_forecast.rename(columns=rename_dict)

st.dataframe(display_df, use_container_width=True, hide_index=True)

# 4. PDF REPORT EXPORT
st.divider()

pdf_bytes = generate_pdf_report(
    forecast=forecast, 
    company_name=cfg.company, 
    historical_trend_pct=9.6, 
    active_events=events
)

st.download_button(
    label="📄 Download PDF Report",
    data=pdf_bytes,
    file_name=f"ScoopCast_{cfg.company}_{forecast.horizon_days}d.pdf",
    mime="application/pdf",
    type="primary"
)