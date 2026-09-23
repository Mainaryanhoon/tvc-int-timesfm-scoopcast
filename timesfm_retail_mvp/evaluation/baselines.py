from __future__ import annotations

import numpy as np


def seasonal_naive(
    history: np.ndarray,
    horizon: int,
    season_length: int = 7,
) -> np.ndarray:
    """
    Seasonal naive forecast.

    Each future value repeats the value from the same
    position one season earlier.
    """

    history = np.asarray(history, dtype=float).reshape(-1)

    if len(history) < season_length:
        raise ValueError(
            f"Need at least {season_length} historical "
            f"observations for seasonal naive forecasting."
        )

    last_season = history[-season_length:]

    repeats = int(
        np.ceil(horizon / season_length)
    )

    forecast = np.tile(
        last_season,
        repeats,
    )

    return forecast[:horizon]

