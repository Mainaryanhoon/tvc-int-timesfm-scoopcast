from __future__ import annotations

import io
from dataclasses import dataclass

import numpy as np
import pandas as pd

REQUIRED_COLUMNS = {"date", "sales"}
OPTIONAL_COVARIATES = [
    "promotion",
    "promotion_intensity",
    "holiday",
    "units_sold",
    "average_price",
]
FUTURE_COVARIATES = ["promotion", "promotion_intensity", "holiday"]


@dataclass
class PreparedData:
    frame: pd.DataFrame
    target: np.ndarray
    context_covariates: np.ndarray | None
    covariate_names: list[str]
    frequency: str


def load_csv(uploaded_file) -> pd.DataFrame:
    if uploaded_file is None:
        raise ValueError("No CSV uploaded.")
    raw = uploaded_file.getvalue()
    return validate_and_prepare(pd.read_csv(io.BytesIO(raw)))


def validate_and_prepare(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip().lower() for c in df.columns]

    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(sorted(missing))}")

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    if df["date"].isna().any():
        raise ValueError("Some values in `date` could not be parsed.")

    for col in [c for c in OPTIONAL_COVARIATES if c in df.columns]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["sales"] = pd.to_numeric(df["sales"], errors="coerce")
    if df["sales"].isna().any():
        raise ValueError("`sales` contains missing or non-numeric values.")
    if (df["sales"] < 0).any():
        raise ValueError("`sales` cannot contain negative values.")

    df = df.sort_values("date").drop_duplicates("date", keep="last")
    if len(df) < 64:
        raise ValueError("Please provide at least 64 daily observations for the MVP.")

    full_dates = pd.date_range(df["date"].min(), df["date"].max(), freq="D")
    df = df.set_index("date").reindex(full_dates).rename_axis("date").reset_index()

    # Sales gaps are filled conservatively for a demo; a production pipeline should
    # make the imputation policy explicit and domain-specific.
    df["sales"] = df["sales"].interpolate(limit_direction="both")

    for col in OPTIONAL_COVARIATES:
        if col in df.columns:
            if col in {"promotion", "holiday"}:
                df[col] = df[col].fillna(0).clip(0, 1)
            elif col == "promotion_intensity":
                df[col] = df[col].fillna(0).clip(0, 1)
            else:
                df[col] = df[col].interpolate(limit_direction="both")

    # Ensure standard future covariates exist for scenario mode.
    for col in FUTURE_COVARIATES:
        if col not in df.columns:
            df[col] = 0.0

    df["weekend"] = (df["date"].dt.dayofweek >= 5).astype(float)
    return df


def prepare_for_timesfm(df: pd.DataFrame) -> PreparedData:
    covariate_names = [c for c in FUTURE_COVARIATES if c in df.columns]
    context_covariates = None
    if covariate_names:
        context_covariates = df[covariate_names].to_numpy(dtype=np.float32).T

    return PreparedData(
        frame=df,
        target=df["sales"].to_numpy(dtype=np.float32),
        context_covariates=context_covariates,
        covariate_names=covariate_names,
        frequency="D",
    )
