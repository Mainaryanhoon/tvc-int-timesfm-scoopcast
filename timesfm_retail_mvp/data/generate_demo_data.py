from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
DATA_DIR.mkdir(exist_ok=True)

rng = np.random.default_rng(42)


# ============================================================
# 1. FROSTLINE CREAMERY — SALES + REVENUE
# ============================================================

dates = pd.date_range("2025-01-01", "2026-09-01", freq="D")

products = {
    "Vanilla": (1150, 42),
    "Chocolate": (980, 45),
    "Strawberry": (820, 44),
    "Mango": (720, 48),
    "Butterscotch": (620, 46),
}

rows = []

for product, (base_units, price) in products.items():
    for i, date in enumerate(dates):

        # Weekly demand pattern
        weekend = 1.12 if date.dayofweek >= 5 else 1.0

        # Mild long-term growth
        trend = 1.0 + 0.00045 * i

        # Temperature: warmer weather increases ice-cream demand
        temperature = (
            27
            + 7 * np.sin(2 * np.pi * i / 365)
            + rng.normal(0, 1.5)
        )

        temperature_effect = 1 + max(temperature - 25, 0) * 0.018

        # Footfall follows weekday/weekend + seasonality
        footfall = (
            100
            + 18 * (date.dayofweek >= 5)
            + 8 * np.sin(2 * np.pi * i / 365)
            + rng.normal(0, 4)
        )

        footfall = max(footfall, 40)

        # Festival / holiday
        festival = int(
            (date.month == 10 and date.day in range(20, 31))
            or (date.month == 11 and date.day in range(1, 6))
            or (date.month == 8 and date.day in range(15, 20))
        )

        holiday = int(
            date.strftime("%m-%d")
            in {
                "01-01",
                "01-26",
                "08-15",
                "10-02",
                "12-25",
            }
        )

        # Promotions
        promotion = int(
            (date.month == 5 and date.day <= 7)
            or (date.month == 8 and 10 <= date.day <= 16)
            or (date.month == 12 and 18 <= date.day <= 31)
        )

        promotion_effect = 1.0 + 0.18 * promotion
        festival_effect = 1.0 + 0.10 * festival
        holiday_effect = 1.0 + 0.12 * holiday
        footfall_effect = 1.0 + (footfall - 100) * 0.0025

        units = (
            base_units
            * weekend
            * trend
            * temperature_effect
            * footfall_effect
            * promotion_effect
            * festival_effect
            * holiday_effect
        )

        units += rng.normal(0, base_units * 0.06)
        # Physical units must be integers
        units = int(max(round(units), 0))

        # Product-specific pricing
        product_price = price + rng.normal(0, 1.5)

        revenue = units * product_price

        rows.append(
            {
                "date": date.strftime("%Y-%m-%d"),
                "product_name": product,
                "units_sold": units,
                "revenue": round(revenue, 2),
                "temperature_c": round(temperature, 1),
                "footfall_index": int(round(footfall)),  # Integer footfall (people count)
                "festival": festival,
                "holiday": holiday,
                "promotion_active": promotion,
            }
        )

retail = pd.DataFrame(rows)

retail.to_csv(
    DATA_DIR / "retail_sales_revenue_v3.csv",
    index=False,
)

print(
    f"Created {DATA_DIR / 'retail_sales_revenue_v3.csv'} "
    f"({len(retail):,} rows)"
)


# ============================================================
# 2. APEX MOTORS — COMPONENT DEMAND / INVENTORY
# ============================================================

components = {
    "BRAKE_PAD_FRONT": {
        "base_demand": 420,
        "inventory": 5200,
        "lead_time": 12,
    },
    "BRAKE_PAD_REAR": {
        "base_demand": 360,
        "inventory": 4500,
        "lead_time": 14,
    },
    "OIL_FILTER": {
        "base_demand": 650,
        "inventory": 7600,
        "lead_time": 8,
    },
    "AIR_FILTER": {
        "base_demand": 520,
        "inventory": 6200,
        "lead_time": 10,
    },
    "SPARK_PLUG": {
        "base_demand": 780,
        "inventory": 9000,
        "lead_time": 7,
    },
}

# ------------------------------------------------------------
# Historical period
# ------------------------------------------------------------

historical_dates = pd.date_range(
    "2025-01-01",
    "2026-09-01",
    freq="D",
)

rows = []

for component, info in components.items():

    inventory = float(info["inventory"])

    target_inventory = info["base_demand"] * 12

    reorder_point = info["base_demand"] * (
        info["lead_time"] / 7 + 3
    )

    for i, date in enumerate(historical_dates):

        planned_production = (
            1000
            + 150 * np.sin(2 * np.pi * i / 30)
            + rng.normal(0, 50)
        )

        planned_production = int(max(
            round(planned_production),
            500,
        ))

        plant_maintenance = int(
            date.day in {10, 11, 12}
        )

        festival = int(
            date.month == 10
            and 20 <= date.day <= 30
        )

        supplier_disruption = int(
            rng.random() < 0.025
        )

        supplier_lead_time = int(
            info["lead_time"]
            + (5 if supplier_disruption else 0)
        )

        demand = (
            info["base_demand"]
            * (planned_production / 1000)
            * (1 - 0.15 * plant_maintenance)
        )

        if festival:
            demand *= 1.08

        demand += rng.normal(
            0,
            info["base_demand"] * 0.05,
        )

        demand = int(max(
            round(demand),
            0,
        ))

        # Historical replenishment policy
        scheduled_receipts = 0

        projected_after_demand = (
            inventory - demand
        )

        if projected_after_demand <= reorder_point:

            receipt_size = max(
                round(
                    target_inventory
                    - projected_after_demand
                ),
                0,
            )

            if supplier_disruption:
                receipt_size = round(
                    receipt_size * 0.45
                )

            scheduled_receipts = int(receipt_size)

        inventory = (
            inventory
            - demand
            + scheduled_receipts
        )

        inventory = max(
            inventory,
            0,
        )

        rows.append({
            "date": date.strftime("%Y-%m-%d"),
            "component_id": component,
            "component_demand": demand,
            "current_inventory": int(round(inventory)),
            "supplier_lead_time_days": supplier_lead_time,
            "supplier_disruption": supplier_disruption,
            "planned_production_units": planned_production,
            "scheduled_receipts": scheduled_receipts,
            "plant_maintenance": plant_maintenance,
            "festival": festival,
        })


# ------------------------------------------------------------
# Known future planning horizon
# ------------------------------------------------------------

future_dates = pd.date_range(
    "2026-09-02",
    periods=60,
    freq="D",
)

for component, info in components.items():

    component_history = [
        r for r in rows
        if r["component_id"] == component
    ]

    # Start planning from the final historical inventory.
    planning_inventory = float(
        component_history[-1]["current_inventory"]
    )

    target_inventory = (
        info["base_demand"] * 12
    )

    reorder_point = info["base_demand"] * (
        info["lead_time"] / 7 + 3
    )

    for j, date in enumerate(future_dates):

        # Known future production plan.
        planned_production = (
            1000
            + 150 * np.sin(
                2 * np.pi * (
                    len(historical_dates) + j
                ) / 30
            )
        )

        planned_production = int(max(
            round(planned_production),
            500,
        ))

        # Known future calendar events.
        plant_maintenance = int(
            date.day in {10, 11, 12}
        )

        festival = int(
            date.month == 10
            and 20 <= date.day <= 30
        )

        supplier_disruption = int(
            ((date.day + j) % 37) == 0
        )

        supplier_lead_time = int(
            info["lead_time"]
            + (5 if supplier_disruption else 0)
        )

        planned_demand = (
            info["base_demand"]
            * (planned_production / 1000)
            * (1 - 0.15 * plant_maintenance)
        )

        if festival:
            planned_demand *= 1.08

        planned_demand = int(max(
            round(planned_demand),
            0,
        ))

        scheduled_receipts = 0

        projected_after_demand = (
            planning_inventory
            - planned_demand
        )

        if projected_after_demand <= reorder_point:

            scheduled_receipts = max(
                round(
                    target_inventory
                    - projected_after_demand
                ),
                0,
            )

            if supplier_disruption:
                scheduled_receipts = round(
                    scheduled_receipts * 0.45
                )

            scheduled_receipts = int(scheduled_receipts)

        planning_inventory = max(
            planning_inventory
            - planned_demand
            + scheduled_receipts,
            0,
        )

        rows.append({
            "date": date.strftime("%Y-%m-%d"),

            # UNKNOWN TARGET — TimesFM forecasts this.
            "component_id": component,
            "component_demand": np.nan,

            # Inventory position is unobserved in the future.
            "current_inventory": np.nan,

            # Known future business information.
            "supplier_lead_time_days": supplier_lead_time,
            "supplier_disruption": supplier_disruption,
            "planned_production_units": planned_production,
            "scheduled_receipts": scheduled_receipts,
            "plant_maintenance": plant_maintenance,
            "festival": festival,
        })


inventory_df = pd.DataFrame(rows)

inventory_df.to_csv(
    DATA_DIR / "automotive_inventory_v1.csv",
    index=False,
)

print(
    f"Created {DATA_DIR / 'automotive_inventory_v1.csv'} "
    f"({len(inventory_df):,} rows)"
)


# ============================================================
# 3. STREAMPULSE — VIEWERSHIP
# ============================================================

content = {
    "CONTENT_001": 4200,
    "CONTENT_002": 3600,
    "CONTENT_003": 3000,
    "CONTENT_004": 2500,
    "CONTENT_005": 2100,
}

rows = []

for content_id, base_hours in content.items():

    for i, date in enumerate(dates):

        weekend = 1.18 if date.dayofweek >= 5 else 1.0

        release_active = int(
            i < 14 or i in {90, 180, 270, 365, 450}
        )

        marketing_campaign = int(
            date.day in range(1, 8)
        )

        festival = int(
            (date.month == 10 and 20 <= date.day <= 30)
            or (date.month == 8 and 15 <= date.day <= 20)
        )

        holiday = int(
            date.strftime("%m-%d")
            in {
                "01-01",
                "01-26",
                "08-15",
                "10-02",
                "12-25",
            }
        )

        competitor_release = int(
            date.day in {10, 11, 12}
        )

        hours = (
            base_hours
            * weekend
            * (1 + 0.30 * release_active)
            * (1 + 0.18 * marketing_campaign)
            * (1 + 0.10 * festival)
            * (1 + 0.12 * holiday)
            * (1 - 0.10 * competitor_release)
        )

        hours += rng.normal(0, base_hours * 0.05)
        # Decimals preserved for time metrics (watch hours)
        hours = max(round(hours, 2), 0.0)

        rows.append(
            {
                "date": date.strftime("%Y-%m-%d"),
                "content_id": content_id,
                "watch_hours": hours,
                "release_active": release_active,
                "marketing_campaign": marketing_campaign,
                "festival": festival,
                "holiday": holiday,
                "competitor_major_release": competitor_release,
            }
        )

viewership = pd.DataFrame(rows)

viewership.to_csv(
    DATA_DIR / "ott_viewership_v1.csv",
    index=False,
)

print(
    f"Created {DATA_DIR / 'ott_viewership_v1.csv'} "
    f"({len(viewership):,} rows)"
)

print("\nDemo datasets generated successfully.")