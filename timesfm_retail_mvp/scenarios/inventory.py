from __future__ import annotations

import numpy as np
import pandas as pd

from config import SCENARIOS
from data.data_manager import (
    aggregate_for_series,
    build_future_covariates,
    split_history_future,
)
from forecast.timesfm3_engine import TimesFM3Engine


def component_demand_from_bom(
    production: pd.DataFrame,
    bom: pd.DataFrame,
    date_col="date",
) -> pd.DataFrame:
    req = {"vehicle_model", "component_id", "units_per_vehicle"}

    if not req.issubset(bom.columns):
        raise ValueError(f"BOM requires {sorted(req)}")

    x = production.merge(
        bom,
        on="vehicle_model",
        how="inner",
    )

    x["component_demand"] = (
        x["planned_production_units"]
        * x["units_per_vehicle"]
    )

    return (
        x.groupby(
            [date_col, "component_id"],
            as_index=False,
        )["component_demand"]
        .sum()
    )


def forecast_inventory(
    df: pd.DataFrame,
    bom: pd.DataFrame,
    component: str,
    horizon: int,
    engine: TimesFM3Engine,
) -> dict:
    cfg = SCENARIOS["Inventory"]

    full = aggregate_for_series(
        df,
        "Inventory",
        component,
    )

    h, f = split_history_future(
        full,
        "Inventory",
        horizon,
    )

    # Ensure component demand exists.
    if "component_demand" not in h.columns:
        if (
            "planned_production_units" in h.columns
            and "units_per_vehicle" in h.columns
        ):
            h["component_demand"] = (
                h["planned_production_units"]
                * h["units_per_vehicle"]
            )
        else:
            raise ValueError(
                "Inventory dataset must contain component_demand "
                "or enough BOM-expanded fields to derive it"
            )

    future = build_future_covariates(
        h,
        f,
        "Inventory",
        horizon,
    )

    # Limit context length.
    context = min(len(h), 512)
    h = h.tail(context)

    target = h["component_demand"].to_numpy(
        np.float32
    )[None, :]

    # Future-known covariates.
    channels = [
        c
        for c in cfg.future_signals
        if c in future.columns
    ]

    pf = (
        np.stack(
            [
                np.concatenate(
                    [
                        h[c].to_numpy(np.float32),
                        future[c].to_numpy(np.float32),
                    ]
                )
                for c in channels
            ]
        )
        if channels
        else None
    )

    # Past-only covariates.
    past_channels = [
        c
        for c in cfg.past_only_signals
        if c in h.columns
    ]

    past = (
        np.stack(
            [
                h[c].to_numpy(np.float32)
                for c in past_channels
            ]
        )
        if past_channels
        else None
    )

    # TimesFM-3 forecast.
    forecast = engine.forecast(
        target,
        horizon,
        past_only_covariates=past,
        past_future_covariates=pf,
        dates=pd.DatetimeIndex(
            future[cfg.date_col]
        ),
        series_names=[component],
    )

    # Support the actual dataset column name while
    # retaining compatibility with closing_inventory.
    inventory_col = (
        "current_inventory"
        if "current_inventory" in h.columns
        else (
            "closing_inventory"
            if "closing_inventory" in h.columns
            else None
        )
    )

    # Safe scalar extraction to prevent Series ambiguity on extended horizons
    current_inventory = np.nan
    if inventory_col is not None:
        valid_inv_series = pd.to_numeric(
            h[inventory_col],
            errors="coerce",
        ).dropna()
        if not valid_inv_series.empty:
            current_inventory = float(valid_inv_series.iloc[-1])

    # Scheduled receipts during the forecast horizon.
    receipts = (
        pd.to_numeric(
            future.get("scheduled_receipts", 0),
            errors="coerce",
        )
        .fillna(0)
        .to_numpy(float)
    )

    # Demand cannot be negative.
    demand = np.maximum(
        forecast.point[0],
        0,
    )

    # Project inventory forward.
    projected = (
        current_inventory
        + np.cumsum(receipts - demand)
        if np.isfinite(current_inventory)
        else np.full(horizon, np.nan)
    )

    # Supplier lead time.
    lead = 7.0
    if "supplier_lead_time_days" in h.columns:
        lead_series = pd.to_numeric(
            h["supplier_lead_time_days"],
            errors="coerce",
        ).dropna()
        if not lead_series.empty:
            lead = float(lead_series.median())

    # Estimate recent demand volatility.
    recent_std = (
        float(
            pd.Series(
                h["component_demand"]
            )
            .rolling(7)
            .std()
            .median()
        )
        if len(h) >= 8
        else float(
            pd.Series(
                h["component_demand"]
            ).std()
        )
    )

    # Safety stock (includes demand variability buffer + lead time buffer).
    mean_daily_demand = float(pd.Series(h["component_demand"]).tail(30).mean())
    demand_var_buffer = (
        1.65
        * max(recent_std, 0)
        * np.sqrt(max(lead, 1) / 7)
    )
    leadtime_delay_buffer = mean_daily_demand * 3.0
    safety_stock = demand_var_buffer + leadtime_delay_buffer

    # Inventory risk status.
    status = (
        np.where(
            projected < safety_stock,
            "SHORTAGE_RISK",
            "OK",
        )
        if np.isfinite(current_inventory)
        else np.array(
            ["UNKNOWN"] * horizon
        )
    )

    return {
        "forecast": forecast,
        "projected_inventory": projected,
        "safety_stock": safety_stock,
        "status": status,
        "current_inventory": current_inventory,
    }