from __future__ import annotations
import os
from pathlib import Path
from typing import Optional
import numpy as np
import pandas as pd
from config import SCENARIOS

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"


def _find_col(df: pd.DataFrame, wanted: str) -> str:
    exact = {str(c).strip().lower(): c for c in df.columns}
    if wanted.lower() in exact:
        return exact[wanted.lower()]
    raise ValueError(f"Required column '{wanted}' not found. Columns: {list(df.columns)}")


def load_csv(scenario: str, path: Optional[str] = None) -> pd.DataFrame:
    cfg = SCENARIOS[scenario]
    file_path = Path(path) if path else DATA_DIR / cfg.file
    if not file_path.exists():
        raise FileNotFoundError(f"Dataset not found: {file_path}")
    df = pd.read_csv(file_path)
    rename = {_find_col(df, cfg.date_col): cfg.date_col}
    for c in [cfg.target_col, cfg.series_col, *cfg.future_signals, *cfg.past_only_signals]:
        if c in df.columns:
            continue
        try:
            rename[_find_col(df, c)] = c
        except ValueError:
            pass
    df = df.rename(columns=rename)
    df[cfg.date_col] = pd.to_datetime(df[cfg.date_col], errors="coerce")
    df = df.dropna(subset=[cfg.date_col]).sort_values(cfg.date_col)
    if cfg.series_col not in df.columns:
        df[cfg.series_col] = "All"
    return df.reset_index(drop=True)


def series_options(df: pd.DataFrame, scenario: str) -> list[str]:
    col = SCENARIOS[scenario].series_col
    return sorted(df[col].dropna().astype(str).unique().tolist())


def _future_calendar(df: pd.DataFrame, scenario: str, horizon: int) -> pd.DataFrame:
    cfg = SCENARIOS[scenario]
    last = df[cfg.date_col].max()
    future_dates = pd.date_range(last + pd.Timedelta(days=1), periods=horizon, freq="D")
    future = pd.DataFrame({cfg.date_col: future_dates})
    for c in cfg.future_signals + cfg.past_only_signals:
        future[c] = np.nan
    return future


def split_history_future(df: pd.DataFrame, scenario: str, horizon: int):
    cfg = SCENARIOS[scenario]
    if cfg.target_col not in df.columns:
        raise ValueError(f"Target '{cfg.target_col}' is missing from dataset")
    target_numeric = pd.to_numeric(df[cfg.target_col], errors="coerce")
    df = df.copy()
    df[cfg.target_col] = target_numeric
    last_target_date = df.loc[df[cfg.target_col].notna(), cfg.date_col].max()
    hist = df[df[cfg.date_col] <= last_target_date].copy()
    supplied_future = df[df[cfg.date_col] > last_target_date].copy()
    if len(supplied_future) >= horizon:
        future = supplied_future.iloc[:horizon].copy()
    else:
        future = _future_calendar(hist, scenario, horizon)
        if len(supplied_future):
            future = pd.concat([supplied_future, future], ignore_index=True).iloc[:horizon].copy()
    return hist, future


def aggregate_for_series(df: pd.DataFrame, scenario: str, series: str) -> pd.DataFrame:
    cfg = SCENARIOS[scenario]
    out = df[df[cfg.series_col].astype(str) == str(series)].copy()
    if out.empty:
        raise ValueError(f"No rows found for {series}")
    out = out.sort_values(cfg.date_col)
    if out[cfg.date_col].duplicated().any():
        agg = {cfg.target_col: "sum"}
        for c in cfg.future_signals + cfg.past_only_signals:
            if c not in out.columns: continue
            agg[c] = "mean" if pd.api.types.is_numeric_dtype(out[c]) else "first"
        out = out.groupby(cfg.date_col, as_index=False).agg(agg)
    return out.sort_values(cfg.date_col).reset_index(drop=True)


def build_future_covariates(history: pd.DataFrame, supplied_future: pd.DataFrame, scenario: str, horizon: int) -> pd.DataFrame:
    """Build future dynamic covariates. Safe against multi-row indexing and extended horizons."""
    cfg = SCENARIOS[scenario]
    dates = pd.date_range(history[cfg.date_col].max() + pd.Timedelta(days=1), periods=horizon, freq="D")
    future = pd.DataFrame({cfg.date_col: dates})
    
    # De-duplicate supplied future frame by date before indexing
    supplied = supplied_future.copy()
    if cfg.date_col in supplied.columns and not supplied.empty:
        supplied[cfg.date_col] = pd.to_datetime(supplied[cfg.date_col], errors="coerce")
        supplied = supplied.groupby(cfg.date_col, as_index=False).first()
        supplied = supplied.set_index(cfg.date_col)
        
    hist = history.copy()
    hist["__dow"] = hist[cfg.date_col].dt.dayofweek
    future["__dow"] = future[cfg.date_col].dt.dayofweek
    estimated = []
    
    for c in cfg.future_signals:
        vals = []
        has_supplied = (c in supplied.columns) if not supplied.empty else False
        for _, row in future.iterrows():
            row_date = row[cfg.date_col]
            v = np.nan
            if has_supplied and row_date in supplied.index:
                val_loc = supplied.loc[row_date, c]
                if isinstance(val_loc, pd.Series):
                    val_loc = val_loc.iloc[0]
                v = val_loc
                
            # Safe scalar check for NaN
            if pd.isna(v) or v is None:
                if c in ("temperature_c", "footfall_index") and c in hist.columns:
                    med = hist.groupby("__dow")[c].median()
                    v = med.get(row["__dow"], hist[c].median())
                    estimated.append(c)
                else:
                    v = 0.0
            vals.append(v)
            
        future[c] = pd.Series(pd.to_numeric(vals, errors="coerce")).fillna(0).astype(float).to_numpy()
        
    future["_estimated_signals"] = ",".join(sorted(set(estimated)))
    return future.drop(columns=["__dow"])