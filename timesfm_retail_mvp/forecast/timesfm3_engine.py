from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd

from config import CHECKPOINT, QUANTILE_LEVELS

@dataclass
class ForecastResult:
    dates: pd.DatetimeIndex
    series_names: list[str]
    point: np.ndarray          # [series, horizon]
    quantiles: np.ndarray      # [series, horizon, 9]
    model: str = "TimesFM-3.0"

class TimesFM3Engine:
    """Thin adapter strictly for the official TimesFM-3.0 PyTorch evaluator."""
    def __init__(self, device: str = "cuda", batch_size: int = 16):
        from timesfm3 import TimesFM3Evaluator, ModelConfig
        self._forecaster = TimesFM3Evaluator(ModelConfig(
            checkpoint_path=CHECKPOINT,
            per_core_batch_size=batch_size,
            device=device,
        ))

    @staticmethod
    def _clean_matrix(matrix: np.ndarray) -> np.ndarray:
        x = np.asarray(matrix, dtype=np.float32)
        for i in range(x.shape[0]):
            s = pd.Series(x[i]).replace([np.inf, -np.inf], np.nan)
            s = s.interpolate(limit_direction="both").ffill().bfill()
            x[i] = s.to_numpy(dtype=np.float32)
        return x

    def forecast(self, target: np.ndarray, horizon: int,
                 past_only_covariates: np.ndarray | None = None,
                 past_future_covariates: np.ndarray | None = None,
                 dates: pd.DatetimeIndex | None = None,
                 series_names: list[str] | None = None) -> ForecastResult:
        
        target = self._clean_matrix(target)
        if target.ndim == 1:
            target = target[None, :]
            
        past = None if past_only_covariates is None else self._clean_matrix(past_only_covariates)
        pf = None if past_future_covariates is None else self._clean_matrix(past_future_covariates)
        
        if past is not None and past.shape[1] != target.shape[1]:
            raise ValueError("Past-only covariates must have the same context length as target")
        if pf is not None and pf.shape[1] != target.shape[1] + horizon:
            raise ValueError("Past-future covariates must have context+horizon columns")
            
        outputs = list(self._forecaster.predict_batch(
            contexts=[target],
            horizon=horizon,
            past_only_covariates=past,
            past_future_covariates=pf,
            return_quantiles=True,
            use_symmetric_averaging=False,
        ))
        
        out = outputs[0]
        point = np.asarray(out.forecast, dtype=np.float32)
        q = np.asarray(out.quantiles, dtype=np.float32)
        if point.ndim == 1: point = point[None, :]
        if q.ndim == 2: q = q[None, :, :]
        if dates is None: dates = pd.date_range("2000-01-01", periods=horizon, freq="D")
        
        return ForecastResult(pd.DatetimeIndex(dates), series_names or [str(i) for i in range(point.shape[0])], point, q)