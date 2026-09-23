from __future__ import annotations

import numpy as np


def mae(actual: np.ndarray, predicted: np.ndarray) -> float:
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)

    mask = np.isfinite(actual) & np.isfinite(predicted)

    if not mask.any():
        return float("nan")

    return float(np.mean(np.abs(actual[mask] - predicted[mask])))


def rmse(actual: np.ndarray, predicted: np.ndarray) -> float:
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)

    mask = np.isfinite(actual) & np.isfinite(predicted)

    if not mask.any():
        return float("nan")

    return float(
        np.sqrt(np.mean((actual[mask] - predicted[mask]) ** 2))
    )


def wape(actual: np.ndarray, predicted: np.ndarray) -> float:
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)

    mask = np.isfinite(actual) & np.isfinite(predicted)

    if not mask.any():
        return float("nan")

    denominator = np.sum(np.abs(actual[mask]))

    if denominator == 0:
        return float("nan")

    return float(
        np.sum(np.abs(actual[mask] - predicted[mask])) / denominator
    )


def bias(actual: np.ndarray, predicted: np.ndarray) -> float:
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)

    mask = np.isfinite(actual) & np.isfinite(predicted)

    if not mask.any():
        return float("nan")

    return float(np.mean(predicted[mask] - actual[mask]))


def interval_coverage(
    actual: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
) -> float:
    actual = np.asarray(actual, dtype=float)
    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)

    mask = (
        np.isfinite(actual)
        & np.isfinite(lower)
        & np.isfinite(upper)
    )

    if not mask.any():
        return float("nan")

    inside = (
        (actual[mask] >= lower[mask])
        & (actual[mask] <= upper[mask])
    )

    return float(np.mean(inside))


def evaluate_forecast(
    actual: np.ndarray,
    predicted: np.ndarray,
    lower: np.ndarray | None = None,
    upper: np.ndarray | None = None,
) -> dict:
    result = {
        "MAE": mae(actual, predicted),
        "RMSE": rmse(actual, predicted),
        "WAPE": wape(actual, predicted),
        "Bias": bias(actual, predicted),
    }

    if lower is not None and upper is not None:
        result["P10_P90_Coverage"] = interval_coverage(
            actual,
            lower,
            upper,
        )

    return result
