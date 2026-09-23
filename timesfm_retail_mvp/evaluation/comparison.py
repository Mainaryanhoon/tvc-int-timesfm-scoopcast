from __future__ import annotations

import pandas as pd

from evaluation.backtesting import (
    run_series_backtest,
    run_baseline_backtest,
    summarize_backtest,
)


def compare_series(
    df: pd.DataFrame,
    scenario: str,
    series: str,
    horizon: int,
    engine,
    windows: int = 3,
) -> pd.DataFrame:
    """
    Compare TimesFM-3 against the seasonal-naive baseline
    using identical rolling backtest windows.
    """

    timesfm_results = run_series_backtest(
        df=df,
        scenario=scenario,
        series=series,
        horizon=horizon,
        engine=engine,
        windows=windows,
    )

    baseline_results = run_baseline_backtest(
        df=df,
        scenario=scenario,
        series=series,
        horizon=horizon,
        windows=windows,
    )

    timesfm = summarize_backtest(timesfm_results)
    baseline = summarize_backtest(baseline_results)

    return pd.DataFrame(
        [
            {
                "series": series,
                "model": "TimesFM-3",
                **timesfm,
            },
            {
                "series": series,
                "model": "Seasonal Naive",
                **baseline,
            },
        ]
    )


def compare_multiple_series(
    df: pd.DataFrame,
    scenario: str,
    series_list: list[str],
    horizon: int,
    engine,
    windows: int = 3,
) -> pd.DataFrame:
    """
    Run the same model-vs-baseline comparison across
    multiple series.
    """

    results = []

    for series in series_list:
        print(
            f"\n{'=' * 60}\n"
            f"Evaluating {series}\n"
            f"{'=' * 60}"
        )

        comparison = compare_series(
            df=df,
            scenario=scenario,
            series=series,
            horizon=horizon,
            engine=engine,
            windows=windows,
        )

        results.append(comparison)

    if not results:
        return pd.DataFrame()

    return pd.concat(
        results,
        ignore_index=True,
    )
