from __future__ import annotations
from evaluation.baselines import seasonal_naive

import numpy as np
import pandas as pd

from config import SCENARIOS
from data.data_manager import (
    aggregate_for_series,
    build_future_covariates,
    split_history_future,
)
from evaluation.metrics import evaluate_forecast
from forecast.timesfm3_engine import TimesFM3Engine


def get_backtest_cutoffs(
    df: pd.DataFrame,
    scenario: str,
    series: str,
    horizon: int,
    windows: int = 3,
    step: int | None = None,
) -> list[pd.Timestamp]:
    """
    Generate rolling-origin forecast cutoffs.

    Each cutoff has at least `horizon` known observations after it,
    so those observations can be used as the held-out evaluation set.
    """

    cfg = SCENARIOS[scenario]

    series_df = aggregate_for_series(
        df,
        scenario,
        series,
    )

    dates = pd.DatetimeIndex(
        series_df[cfg.date_col].sort_values().unique()
    )

    if step is None:
        step = horizon

    required = horizon + (windows - 1) * step

    if len(dates) <= required:
        raise ValueError(
            f"Not enough history for {windows} backtest windows "
            f"with horizon={horizon}."
        )

    last_possible_cutoff = dates[-horizon - 1]

    cutoffs = []

    for i in range(windows):
        cutoff = last_possible_cutoff - pd.Timedelta(
            days=(windows - 1 - i) * step
        )

        if cutoff in dates:
            cutoffs.append(cutoff)

    return cutoffs


def _build_retail_model_inputs(
    df: pd.DataFrame,
    scenario: str,
    series: list[str],
    cutoff: pd.Timestamp,
    horizon: int,
):
    """
    Build model inputs using the same preparation path as the
    production retail forecast.

    Actual targets after `cutoff` are hidden from the model,
    while future covariates remain available.
    """

    cfg = SCENARIOS[scenario]

    histories = []
    futures = []

    for s in series:
        full = aggregate_for_series(
            df,
            scenario,
            s,
        )

        # Hide future target values while preserving future signals.
        model_full = full.copy()

        model_full[cfg.target_col] = pd.to_numeric(
            model_full[cfg.target_col],
            errors="coerce",
        )

        model_full.loc[
            model_full[cfg.date_col] > cutoff,
            cfg.target_col,
        ] = np.nan

        history, supplied_future = split_history_future(
            model_full,
            scenario,
            horizon,
        )

        future = build_future_covariates(
            history,
            supplied_future,
            scenario,
            horizon,
        )

        histories.append(history)
        futures.append(future)

    # Same common-end alignment as scenarios/retail.py
    end = min(
        h[cfg.date_col].max()
        for h in histories
    )

    histories = [
        h[h[cfg.date_col] <= end].copy()
        for h in histories
    ]

    context = min(
        len(h)
        for h in histories
    )

    histories = [
        h.tail(context).copy()
        for h in histories
    ]

    target = np.stack(
        [
            h[cfg.target_col].to_numpy(
                dtype=np.float32
            )
            for h in histories
        ]
    )

    # Same shared future covariate logic as retail.py
    common_future = futures[0].copy()

    future_dates = pd.DatetimeIndex(
        common_future[cfg.date_col]
    )

    future_channels = [
        c
        for c in cfg.future_signals
        if c in common_future.columns
    ]

    pf_parts = []

    for c in future_channels:

        if c in histories[0].columns:
            hist_vals = histories[0][c].to_numpy(
                dtype=np.float32
            )
        else:
            hist_vals = np.zeros(
                context,
                dtype=np.float32,
            )

        fut_vals = common_future[c].to_numpy(
            dtype=np.float32
        )

        pf_parts.append(
            np.concatenate(
                [
                    hist_vals,
                    fut_vals,
                ]
            )
        )

    pf = (
        np.stack(pf_parts)
        if pf_parts
        else None
    )

    return (
        target,
        pf,
        future_dates,
    )

def run_series_backtest(
    df,
    scenario,
    series,
    horizon,
    engine,
    windows=3,
) -> pd.DataFrame:
    cfg = SCENARIOS[scenario]

    cutoffs = get_backtest_cutoffs(
        df=df,
        scenario=scenario,
        series=series,
        horizon=horizon,
        windows=windows,
    )

    rows = []

    original = (
        aggregate_for_series(df, scenario, series)
        .sort_values(cfg.date_col)
        .reset_index(drop=True)
    )

    for window_id, cutoff in enumerate(cutoffs, start=1):

        print(
            f"  Backtest window {window_id}/{len(cutoffs)} "
            f"· cutoff={cutoff.date()}"
        )

        future_actual = (
            original[original[cfg.date_col] > cutoff]
            .head(horizon)
        )

        if len(future_actual) < horizon:
            continue

        target, pf, future_dates = _build_retail_model_inputs(
            df,
            scenario,
            [series],
            cutoff,
            horizon,
        )

        result = engine.forecast(
            target=target,
            horizon=horizon,
            past_only_covariates=None,
            past_future_covariates=pf,
            dates=future_dates,
            series_names=[series],
        )

        actual = future_actual[
            cfg.target_col
        ].to_numpy(dtype=float)

        predicted = np.asarray(
            result.point[0],
            dtype=float,
        )

        lower = np.asarray(
            result.quantiles[0, :, 0],
            dtype=float,
        )

        upper = np.asarray(
            result.quantiles[0, :, 8],
            dtype=float,
        )

        metrics = evaluate_forecast(
            actual=actual,
            predicted=predicted,
            lower=lower,
            upper=upper,
        )

        rows.append(
            {
                "window": window_id,
                "cutoff": cutoff,
                "series": series,
                "horizon": horizon,
                **metrics,
            }
        )

    return pd.DataFrame(rows)

def run_baseline_backtest(
    df: pd.DataFrame,
    scenario: str,
    series: str,
    horizon: int,
    windows: int = 3,
) -> pd.DataFrame:
    """
    Rolling-origin evaluation for the seasonal-naive baseline.

    Uses exactly the same cutoff dates as the TimesFM-3
    backtest so the comparison is directly aligned.
    """

    cfg = SCENARIOS[scenario]

    cutoffs = get_backtest_cutoffs(
        df=df,
        scenario=scenario,
        series=series,
        horizon=horizon,
        windows=windows,
    )

    original = aggregate_for_series(
        df,
        scenario,
        series,
    ).sort_values(
        cfg.date_col
    ).reset_index(drop=True)

    rows = []

    for window_id, cutoff in enumerate(
        cutoffs,
        start=1,
    ):

        print(
            f"  Baseline window {window_id}/{len(cutoffs)} "
            f"· cutoff={cutoff.date()}"
        )

        history = original[
            original[cfg.date_col] <= cutoff
        ]

        future_actual = original[
            original[cfg.date_col] > cutoff
        ].head(horizon)

        if len(history) == 0 or len(future_actual) < horizon:
            continue

        actual = future_actual[
            cfg.target_col
        ].to_numpy(dtype=float)

        predicted = seasonal_naive(
            history[cfg.target_col].to_numpy(
                dtype=float
            ),
            horizon=horizon,
            season_length=7,
        )

        metrics = evaluate_forecast(
            actual=actual,
            predicted=predicted,
        )

        rows.append(
            {
                "window": window_id,
                "cutoff": cutoff,
                "series": series,
                "horizon": horizon,
                **metrics,
            }
        )

    return pd.DataFrame(rows)

def summarize_backtest(
    results: pd.DataFrame,
) -> dict:
    """
    Aggregate metrics across rolling windows.
    """

    if results.empty:
        return {}

    metric_columns = [
        "MAE",
        "RMSE",
        "WAPE",
        "Bias",
        "P10_P90_Coverage",
    ]

    summary = {}

    for metric in metric_columns:

        if metric not in results.columns:
            continue

        values = pd.to_numeric(
            results[metric],
            errors="coerce",
        ).dropna()

        if len(values):
            summary[metric] = float(
                values.mean()
            )

    summary["windows"] = int(
        len(results)
    )

    return summary