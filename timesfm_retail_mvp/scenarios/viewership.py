from __future__ import annotations
import numpy as np
import pandas as pd
from config import SCENARIOS
from data.data_manager import aggregate_for_series, build_future_covariates, split_history_future
from forecast.timesfm3_engine import TimesFM3Engine, ForecastResult

def forecast_viewership(df: pd.DataFrame, content: list[str], horizon: int, engine: TimesFM3Engine) -> dict:
    cfg = SCENARIOS["Viewership"]
    histories, futures = [], []
    for s in content:
        full = aggregate_for_series(df, "Viewership", s)
        h, f = split_history_future(full, "Viewership", horizon)
        histories.append(h)
        futures.append(build_future_covariates(h, f, "Viewership", horizon))
    end = min(h[cfg.date_col].max() for h in histories)
    histories = [h[h[cfg.date_col] <= end] for h in histories]
    context = min(len(h) for h in histories)
    histories = [h.tail(context) for h in histories]
    target = np.stack([h[cfg.target_col].to_numpy(np.float32) for h in histories])
    fut = futures[0]
    channels = [c for c in cfg.future_signals if c in fut.columns]
    pf = np.stack([np.concatenate([histories[0][c].to_numpy(np.float32), fut[c].to_numpy(np.float32)]) for c in channels]) if channels else None
    past_channels = [c for c in cfg.past_only_signals if c in histories[0].columns]
    past = np.stack([histories[0][c].to_numpy(np.float32) for c in past_channels]) if past_channels else None
    
    result = engine.forecast(
        target, horizon, past_only_covariates=past, past_future_covariates=pf,
        dates=pd.DatetimeIndex(fut[cfg.date_col]), series_names=content
    )
    
    return {
        "base": result,
        "scenario": result,
        "has_what_if": False
    }