import streamlit as st
from forecast.architecture import ForecastResult

def generate_evidence_why(forecast: ForecastResult, historical_trend_pct: float, active_events: list[str]) -> str:
    """Generates the data-driven 'Why' based on active signals and model logic."""
    trend_direction = "above" if historical_trend_pct >= 0 else "below"
    
    why_text = (
        f"**Model Baseline:** TimesFM analyzed the historical data sequence, identifying that recent demand "
        f"is running {abs(historical_trend_pct)}% {trend_direction} the preceding comparable period. "
        "The forecast preserves the recurring seasonal patterns observed in this history.\n\n"
    )
    
    if active_events:
        event_str = ", ".join(active_events)
        why_text += (
            f"**Applied Business Signals:** The forecast trajectory was dynamically adjusted by passing the following "
            f"known future information into the model's context window: **{event_str}**."
        )
    else:
        why_text += "**Applied Business Signals:** Only historical sales patterns were used. No future context signals were provided."
        
    return why_text


def generate_action(forecast: ForecastResult) -> str:
    """Generates the top-level text for the recommended action."""
    formatted_demand = f"{forecast.total_demand:,.2f}" if forecast.domain == "Viewership" else f"{forecast.total_demand:,}"
    unit_label = "units" if forecast.domain in ["Sales", "Inventory"] else "engagements" if forecast.domain == "Viewership" else "revenue"
    
    if forecast.domain == "Inventory":
        action = f"Plan inventory against approximately **{formatted_demand} units** of expected demand across the selected {forecast.horizon_days}-day forecast period.\n\n"
        
        if forecast.inventory_metrics and forecast.inventory_metrics['net_shortage'] > 0:
            shortage = forecast.inventory_metrics['net_shortage']
            action += f"🚨 **Immediate replenishment of {shortage:,} units is recommended.**"
        elif forecast.inventory_metrics:
            stock = forecast.inventory_metrics['current_stock']
            buffer = forecast.inventory_metrics['safety_buffer_units']
            action += f"✅ **Current stock ({stock:,} units) is sufficient.** No immediate replenishment is required to maintain the {buffer:,}-unit safety buffer."
            
    elif forecast.domain == "Viewership":
        action = f"Align content delivery and targeted marketing efforts for an expected **{formatted_demand} {unit_label}** across the selected {forecast.horizon_days}-day period."
        
    else:
        # Sales / Revenue
        action = f"Align retail staffing and marketing strategy against **{formatted_demand} expected {unit_label}** across the selected {forecast.horizon_days}-day period."
        
    return action


def render_executive_summary(forecast: ForecastResult, historical_trend_pct: float = 9.6, active_events: list[str] = None):
    """Renders the top-level Executive Summary UI."""
    if active_events is None:
        active_events = []
        
    unit_label = "units" if forecast.domain in ["Sales", "Inventory"] else "viewers" if forecast.domain == "Viewership" else "revenue"
    demand_display = f"{forecast.total_demand:,.2f}" if forecast.domain == "Viewership" else f"{forecast.total_demand:,}"
    
    with st.container(border=True):
        st.markdown(f"### EXECUTIVE SUMMARY — {forecast.target_name.replace('_', ' ')}")
        
        # 1. Expected Outcome
        st.markdown("#### Expected demand")
        st.markdown(f"## {demand_display} {unit_label}")
        st.markdown("---")
        
        # 2. Evidence (The 'Why')
        st.markdown("#### WHY THIS FORECAST")
        st.write(generate_evidence_why(forecast, historical_trend_pct, active_events))
        st.markdown("---")
        
        # 3. Recommended Action
        st.markdown("#### RECOMMENDED ACTION")
        st.write(generate_action(forecast))
        
        # --- EXPLICIT VISUAL MATH BREAKDOWN FOR INVENTORY ---
        if forecast.domain == "Inventory" and forecast.inventory_metrics:
            shortage = forecast.inventory_metrics['net_shortage']
            if shortage > 0:
                # Clean single-icon info callout
                st.info("Replenishment Calculation Breakdown: How we reached the recommended order quantity.", icon="ℹ️")
                
                c1, c2, c3, c4 = st.columns(4)
                policy_txt = forecast.inventory_metrics['buffer_policy_text']
                
                c1.metric("1. Total Expected Demand", f"{forecast.total_demand:,}")
                c2.metric(
                    "2. Safety Buffer (+)", 
                    f"{forecast.inventory_metrics['safety_buffer_units']:,}",
                    help=f"Calculated as (Average Daily Demand) × (Supplier Lead Time). {policy_txt}"
                )
                c3.metric("3. Current Stock (-)", f"{forecast.inventory_metrics['current_stock']:,}")
                c4.metric("= Recommended Order", f"{shortage:,}")
                
        st.markdown("---")
        
        # 4. Watch / Uncertainty Risk
        st.markdown("#### WATCH")
        st.write(f"Forecast uncertainty increases toward the end of the **{forecast.horizon_days}-day period**. "
                 "Later-period inventory and capacity commitments should be reviewed as new actuals become available.")
        
        # 5. What-If Scenario Impact (If active)
        if forecast.has_scenario:
            st.markdown("---")
            st.markdown("#### SCENARIO IMPACT")
            diff = forecast.scenario_difference
            diff_str = f"+{diff:,.2f}" if forecast.domain == "Viewership" and diff > 0 else f"{diff:,.2f}" if forecast.domain == "Viewership" else f"+{diff:,}" if diff > 0 else f"{diff:,}"
            st.write(f"The applied What-If scenario alters the baseline expectation by **{diff_str} {unit_label}** "
                     f"over the selected {forecast.horizon_days}-day horizon.")