from __future__ import annotations
import numpy as np
import pandas as pd
from config import SCENARIOS
from data.data_manager import aggregate_for_series, build_future_covariates, split_history_future
from forecast.timesfm3_engine import TimesFM3Engine, ForecastResult

def forecast_retail(
    df: pd.DataFrame, 
    scenario: str, 
    series: list[str], 
    horizon: int, 
    engine: TimesFM3Engine,
    promo_start_day: int | None = None,
    promo_end_day: int | None = None,
    promo_intensity: float = 1.0
) -> dict:
    cfg = SCENARIOS[scenario]
    histories, futures = [], []
    
    for s in series:
        full = aggregate_for_series(df, scenario, s)
        h, f = split_history_future(full, scenario, horizon)
        histories.append(h)
        futures.append(build_future_covariates(h, f, scenario, horizon))
        
    end = min(h[cfg.date_col].max() for h in histories)
    histories = [h[h[cfg.date_col] <= end] for h in histories]
    context = min(len(h) for h in histories)
    histories = [h.tail(context) for h in histories]
    
    target = np.stack([h[cfg.target_col].to_numpy(np.float32) for h in histories])
    
    # 1. Base Future Covariates
    common_future_base = futures[0].copy()
    future_dates = pd.DatetimeIndex(common_future_base[cfg.date_col])
    future_channels = [c for c in cfg.future_signals if c in common_future_base.columns]
    
    def build_pf_matrix(fut_df):
        pf_parts = []
        for c in future_channels:
            # FIX: Explicitly check against histories[0].columns (not histories[0] directly)
            hist_vals = histories[0][c].to_numpy(np.float32) if c in histories[0].columns else np.zeros(context, np.float32)
            fut_vals = fut_df[c].to_numpy(np.float32)
            pf_parts.append(np.concatenate([hist_vals, fut_vals]))
        return np.stack(pf_parts) if pf_parts else None

    pf_base = build_pf_matrix(common_future_base)
    base_result = engine.forecast(
        target, horizon, past_only_covariates=None, past_future_covariates=pf_base, dates=future_dates, series_names=series
    )

    # 2. What-If Scenario Covariates (User-tweaked promotion override)
    common_future_scenario = common_future_base.copy()
    has_what_if = False
    
    if promo_start_day is not None and promo_end_day is not None and "promotion_active" in common_future_scenario.columns:
        has_what_if = True
        # FIX: Safe integer positional indexing
        n_rows = len(common_future_scenario)
        s_idx = max(0, min(int(promo_start_day), n_rows - 1))
        e_idx = max(0, min(int(promo_end_day), n_rows - 1))
        
        promo_vals = common_future_scenario["promotion_active"].to_numpy(dtype=np.float32)
        promo_vals[s_idx:e_idx + 1] = float(promo_intensity)
        common_future_scenario["promotion_active"] = promo_vals

    pf_scenario = build_pf_matrix(common_future_scenario)
    scenario_result = engine.forecast(
        target, horizon, past_only_covariates=None, past_future_covariates=pf_scenario, dates=future_dates, series_names=series
    )

    return {
        "base": base_result,
        "scenario": scenario_result,
        "has_what_if": has_what_if
    }