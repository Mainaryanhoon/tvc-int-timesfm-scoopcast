import pandas as pd
import plotly.graph_objects as go


def _resample_history(df: pd.DataFrame, date_col: str, target_col: str, view_mode: str) -> pd.DataFrame:
    """Helper to aggregate historical actuals to match the Daily/Monthly toggle."""
    if df is None or df.empty:
        return pd.DataFrame()
        
    df_copy = df.copy()
    df_copy[date_col] = pd.to_datetime(df_copy[date_col])
    
    if view_mode == "Monthly":
        df_copy = df_copy.set_index(date_col).resample("MS").agg({target_col: "sum"}).reset_index()
        df_copy['display_date'] = df_copy[date_col].dt.strftime('%b %Y')  # e.g., "Sep 2026"
    else:
        df_copy['display_date'] = df_copy[date_col].dt.strftime('%d %b %Y')  # e.g., "06 Sep 2026"
        
    return df_copy


def _apply_staggered_x_labels(series: pd.Series) -> list[str]:
    """
    Formats X-axis date strings with alternating line offsets 
    so labels render horizontally in staggered rows (one up, one down).
    """
    staggered = []
    for idx, val in enumerate(series):
        if idx % 2 == 1:
            staggered.append(f"<br>{val}")  # Shift alternate labels down one line
        else:
            staggered.append(str(val))
    return staggered


def render_business_forecast_chart(
    forecast,  # ForecastResult object
    hist_df: pd.DataFrame = None,
    hist_date_col: str = "date",
    hist_target_col: str = "sales",
    view_mode: str = "Daily",
    unit_label: str = "Units"
) -> go.Figure:
    """Renders the standard Business Forecast Chart (Sales, Revenue, Viewership)."""
    fig = go.Figure()

    # Dynamic Formatting: Allow decimals for Viewership, force integers for physical units
    is_viewership = getattr(forecast, "domain", None) == "Viewership"
    num_fmt = "%{y:,.2f}" if is_viewership else "%{y:,.0f}"
    tick_fmt = ",.2f" if is_viewership else ",d"

    # 1. Select dataset and format X-axis labels
    if view_mode == "Monthly":
        df = forecast.monthly_summary.copy()
        raw_x = df['month']
        x_data = raw_x  # Monthly labels have ample spacing
        dtick_setting = "M1"
    else:
        df = forecast.daily_forecast.copy()
        raw_x = df['date']
        # Apply Staggered Horizontal Formatting for Daily view
        x_data = _apply_staggered_x_labels(raw_x)
        dtick_setting = None

    # 2. Add Historical Actuals
    hist_clean = _resample_history(hist_df, hist_date_col, hist_target_col, view_mode)
    if not hist_clean.empty:
        hist_x = (
            _apply_staggered_x_labels(hist_clean['display_date'])
            if view_mode == "Daily"
            else hist_clean['display_date']
        )
        fig.add_trace(go.Scatter(
            x=hist_x, 
            y=hist_clean[hist_target_col],
            mode='lines', 
            line=dict(color="#64748B", width=2),  # Muted slate for past data
            name="Historical Actuals",
            hovertemplate=f"<b>Date:</b> %{{x}}<br><b>Actual:</b> {num_fmt} {unit_label}<extra></extra>"
        ))

    # 3. P10-P90 Expected Range Band
    fig.add_trace(go.Scatter(
        x=list(x_data) + list(x_data)[::-1],
        y=df['upper'].tolist() + df['lower'].tolist()[::-1],
        fill='toself', 
        fillcolor='rgba(37, 99, 235, 0.12)',  # Soft Blue
        line=dict(color='rgba(255,255,255,0)'),
        hoverinfo="skip", 
        showlegend=False,
        name="Expected Range"
    ))
    
    # 4. Expected Demand (Baseline)
    fig.add_trace(go.Scatter(
        x=x_data, 
        y=df['expected'], 
        mode='lines+markers' if not forecast.has_scenario else 'lines',
        name="Expected Forecast (Baseline)",
        line=dict(color="#2563EB", width=3, dash='dash' if forecast.has_scenario else 'solid'),
        marker=dict(size=8, color="#2563EB", line=dict(width=1, color="white")),
        hovertemplate=f"<b>Date:</b> %{{x}}<br><b>Expected:</b> {num_fmt} {unit_label}<extra></extra>"
    ))

    # 5. What-If Scenario Overlay
    if forecast.has_scenario:
        scen_df = forecast.scenario_daily.copy()
        if view_mode == "Monthly":
            scen_df['raw_date'] = pd.to_datetime(scen_df['date'])
            scen_df = scen_df.set_index('raw_date').resample("MS").agg({'expected': 'sum'}).reset_index()
            scen_x = scen_df['raw_date'].dt.strftime('%b %Y')
        else:
            scen_x = _apply_staggered_x_labels(scen_df['date'])
            
        fig.add_trace(go.Scatter(
            x=scen_x,
            y=scen_df['expected'],
            mode='lines+markers',
            name="What-If Scenario",
            line=dict(color="#10B981", width=3),  # Strong Emerald Green
            marker=dict(size=8, color="#10B981", line=dict(width=1, color="white")),
            hovertemplate=f"<b>Date:</b> %{{x}}<br><b>Scenario:</b> {num_fmt} {unit_label}<extra></extra>"
        ))

    # 6. Forecast Boundary Marker (Replaces add_vline to prevent Plotly string bug)
    if not df.empty:
        max_y = float(df['upper'].max() if 'upper' in df.columns else df['expected'].max())
        first_x = x_data[0] if isinstance(x_data, list) else x_data.iloc[0]
        
        fig.add_trace(go.Scatter(
            x=[first_x, first_x],
            y=[0, max_y * 1.15],
            mode="lines",
            line=dict(color="#94A3B8", width=2, dash="dot"),
            name="Forecast Start",
            showlegend=False,
            hoverinfo="skip"
        ))
        
        fig.add_annotation(
            x=first_x,
            y=max_y * 1.15,
            text="Forecast Start",
            showarrow=False,
            xanchor="left",
            yanchor="bottom",
            font=dict(color="#94A3B8", size=11)
        )

    # 7. Clean Layout configuration
    fig.update_layout(
        height=440, 
        margin=dict(l=15, r=15, t=25, b=45),
        paper_bgcolor="white", 
        plot_bgcolor="white", 
        hovermode="x unified",
        legend=dict(orientation="h", y=1.08, x=0), 
        xaxis=dict(
            showgrid=False, 
            dtick=dtick_setting,
            tickangle=0,            # Force horizontal alignment (no vertical rotation)
            nticks=12,              # Clean horizontal spacing
            showticklabels=True
        ),
        yaxis=dict(gridcolor="#F1F5F9", title=unit_label, tickformat=tick_fmt)
    )
    return fig


def render_inventory_chart(
    forecast,  # ForecastResult object
    view_mode: str = "Daily"
) -> go.Figure:
    """Renders the Inventory Outlook using strictly business terminology."""
    fig = go.Figure()
    
    # 1. Base DataFrame from single-source
    df = forecast.daily_forecast.copy()
    
    # 2. Extract explicit safety buffer from single-source metrics
    buffer_units = forecast.inventory_metrics['safety_buffer_units'] if forecast.inventory_metrics else 0
    df['safety_buffer'] = buffer_units
    
    # 3. Handle Time Aggregation (Strict Supply Chain rules)
    if view_mode == "Monthly":
        df['raw_date'] = pd.to_datetime(df['date'])
        # Rule: Demand is SUMMED. Closing Stock is LAST day of month. Buffer is CONSTANT.
        df = df.set_index('raw_date').resample("MS").agg({
            'expected': 'sum',
            'expected_closing_stock': 'last',
            'safety_buffer': 'last'
        }).reset_index()
        x_data = df['raw_date'].dt.strftime('%b %Y')
        dtick_setting = "M1"
    else:
        raw_x = df['date']
        x_data = _apply_staggered_x_labels(raw_x)
        dtick_setting = None

    # Trace A: Projected Closing Stock (Dark Slate)
    fig.add_trace(go.Scatter(
        x=x_data, y=df['expected_closing_stock'], mode='lines+markers',
        name="Projected Closing Stock",
        line=dict(color="#0F172A", width=3),
        marker=dict(size=8, color="#0F172A", line=dict(width=1, color="white")),
        hovertemplate="<b>Date:</b> %{x}<br><b>Closing Stock:</b> %{y:,.0f} units<extra></extra>"
    ))

    # Trace B: Expected Customer Demand (Red)
    fig.add_trace(go.Scatter(
        x=x_data, y=df['expected'], mode='lines+markers',
        name="Expected Customer Demand",
        line=dict(color="#E11D48", width=3),
        marker=dict(size=8, color="#E11D48", line=dict(width=1, color="white")),
        hovertemplate="<b>Date:</b> %{x}<br><b>Demand:</b> %{y:,.0f} units<extra></extra>"
    ))

    # Trace C: Safety Buffer (Dashed Orange)
    fig.add_trace(go.Scatter(
        x=x_data, y=df['safety_buffer'], mode='lines',
        name="Safety Buffer Threshold",
        line=dict(color="#F59E0B", width=2, dash="dash"),
        hovertemplate="<b>Threshold:</b> %{y:,.0f} units<extra></extra>"
    ))

    # Layout configuration
    fig.update_layout(
        height=440, 
        margin=dict(l=15, r=15, t=25, b=45),
        paper_bgcolor="white", 
        plot_bgcolor="white", 
        hovermode="x unified",
        legend=dict(orientation="h", y=1.08, x=0), 
        xaxis=dict(
            showgrid=False, 
            dtick=dtick_setting,
            tickangle=0,            # Force horizontal alignment
            nticks=12
        ),
        yaxis=dict(gridcolor="#F1F5F9", title="Units", tickformat=",d")
    )
    return fig