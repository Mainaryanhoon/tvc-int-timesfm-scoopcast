import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Optional, Dict, Union

@dataclass
class ForecastResult:
    """
    SINGLE SOURCE OF TRUTH.
    Every chart, table, summary, and PDF consumes this exact object.
    """
    domain: str
    target_name: str
    horizon_days: int
    forecast_start_date: str  # Format: "06 Sep 2026"
    forecast_end_date: str    # Format: "25 Oct 2026"
    
    # 1. Aggregate Metrics (For Executive Summary)
    total_demand: Union[int, float]
    average_daily_demand: Union[int, float]
    
    # 2. Base Forecast Data (For Charts/Tables)
    daily_forecast: pd.DataFrame    # date, expected, lower, upper
    monthly_summary: pd.DataFrame   # month, expected, lower, upper
    
    # 3. Scenario Data (If active)
    has_scenario: bool
    scenario_total: Union[int, float]
    scenario_difference: Union[int, float]
    scenario_daily: Optional[pd.DataFrame]
    
    # 4. Inventory Specific Data (Only populated if domain == 'Inventory')
    inventory_metrics: Optional[Dict] = None


def build_forecast_result(
    domain: str,
    target_name: str,
    raw_dates: pd.DatetimeIndex,
    p50_baseline: np.ndarray,
    p10_baseline: np.ndarray,
    p90_baseline: np.ndarray,
    horizon_days: int,
    p50_scenario: Optional[np.ndarray] = None,
    current_stock: float = 0.0,
    lead_time_days: int = 14
) -> ForecastResult:
    """
    Slices raw TimesFM output strictly to the selected horizon 
    and builds the single-source-of-truth ForecastResult object.
    """
    # 1. SLICE EXACTLY TO SELECTED HORIZON
    dates = raw_dates[:horizon_days]
    p50 = p50_baseline[:horizon_days]
    p10 = p10_baseline[:horizon_days]
    p90 = p90_baseline[:horizon_days]
    
    # 2. BUILD DAILY DATAFRAME (Domain-Aware Rounding)
    if domain == "Viewership":
        exp_vals = np.round(p50, 2)
        low_vals = np.round(p10, 2)
        up_vals = np.round(p90, 2)
    else:
        exp_vals = np.round(p50).astype(int)
        low_vals = np.round(p10).astype(int)
        up_vals = np.round(p90).astype(int)
        
    df_daily = pd.DataFrame({
        'date_raw': dates,
        'date': dates.strftime('%d %b %Y'), # Clean "06 Sep 2026" (No 00:00:00)
        'expected': exp_vals,
        'lower': low_vals,
        'upper': up_vals
    })
    
    # 3. BUILD MONTHLY ROLLUP (Clean "Sep 2026")
    df_monthly = df_daily.copy()
    df_monthly['month_idx'] = df_monthly['date_raw'].dt.to_period('M')
    df_monthly_agg = df_monthly.groupby('month_idx').agg({
        'expected': 'sum',
        'lower': 'sum',
        'upper': 'sum',
        'date_raw': 'first'
    }).reset_index()
    
    df_monthly_agg['month'] = df_monthly_agg['date_raw'].dt.strftime('%b %Y')
    df_monthly_agg = df_monthly_agg.drop(columns=['month_idx', 'date_raw'])
    
    if domain == "Viewership":
        df_monthly_agg['expected'] = np.round(df_monthly_agg['expected'], 2)
        df_monthly_agg['lower'] = np.round(df_monthly_agg['lower'], 2)
        df_monthly_agg['upper'] = np.round(df_monthly_agg['upper'], 2)
    
    # 4. CALCULATE SINGLE-SOURCE TOTALS
    if domain == "Viewership":
        total_demand = float(np.round(df_daily['expected'].sum(), 2))
        avg_daily_demand = float(np.round(total_demand / horizon_days, 2))
    else:
        total_demand = int(df_daily['expected'].sum())
        avg_daily_demand = int(np.round(total_demand / horizon_days))
    
    # 5. HANDLE SCENARIO MATH
    has_scenario = p50_scenario is not None
    scenario_tot = 0.0 if domain == "Viewership" else 0
    scenario_diff = 0.0 if domain == "Viewership" else 0
    df_scenario = None
    
    if has_scenario:
        p50_scen = p50_scenario[:horizon_days]
        scen_vals = np.round(p50_scen, 2) if domain == "Viewership" else np.round(p50_scen).astype(int)
        
        df_scenario = pd.DataFrame({
            'date': dates.strftime('%d %b %Y'),
            'expected': scen_vals
        })
        
        if domain == "Viewership":
            scenario_tot = float(np.round(df_scenario['expected'].sum(), 2))
            scenario_diff = float(np.round(scenario_tot - total_demand, 2))
        else:
            scenario_tot = int(df_scenario['expected'].sum())
            scenario_diff = int(scenario_tot - total_demand)
        
    # 6. HANDLE INVENTORY ACCOUNTING (Explicit Day-by-Day Math)
    inv_metrics = None
    if domain == 'Inventory':
        explicit_safety_buffer = int(avg_daily_demand * lead_time_days)
        
        running_stock = int(current_stock)
        opening_stock_list = []
        projected_closing = []
        
        for dmd in df_daily['expected']:
            opening_stock_list.append(running_stock)
            running_stock -= dmd
            projected_closing.append(running_stock)
            
        df_daily['opening_stock'] = opening_stock_list
        df_daily['expected_closing_stock'] = projected_closing
        df_daily['safety_buffer'] = explicit_safety_buffer
        
        net_shortage = max(0, int(total_demand + explicit_safety_buffer - current_stock))
        
        inv_metrics = {
            'current_stock': int(current_stock),
            'safety_buffer_units': explicit_safety_buffer,
            'buffer_policy_text': f"Equivalent to {lead_time_days} days of expected average demand.",
            'net_shortage': net_shortage,
            'min_closing_stock': min(projected_closing)
        }
        
    return ForecastResult(
        domain=domain,
        target_name=target_name,
        horizon_days=horizon_days,
        forecast_start_date=dates[0].strftime('%d %b %Y'),
        forecast_end_date=dates[-1].strftime('%d %b %Y'),
        total_demand=total_demand,
        average_daily_demand=avg_daily_demand,
        daily_forecast=df_daily.drop(columns=['date_raw']),
        monthly_summary=df_monthly_agg,
        has_scenario=has_scenario,
        scenario_total=scenario_tot,
        scenario_difference=scenario_diff,
        scenario_daily=df_scenario,
        inventory_metrics=inv_metrics
    )