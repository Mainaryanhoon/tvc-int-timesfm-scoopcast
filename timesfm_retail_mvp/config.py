from dataclasses import dataclass, field
from typing import Dict, List

@dataclass(frozen=True)
class ScenarioConfig:
    name: str
    company: str
    file: str
    date_col: str
    target_col: str
    series_col: str
    unit: str
    horizon_default: int
    past_only_signals: List[str] = field(default_factory=list)
    future_signals: List[str] = field(default_factory=list)
    signal_labels: Dict[str, str] = field(default_factory=dict)

SCENARIOS = {
    "Sales": ScenarioConfig(
        name="Sales", company="Frostline Creamery", file="retail_sales_revenue_v3.csv",
        date_col="date", target_col="units_sold", series_col="product_name", unit="units", horizon_default=30,
        past_only_signals=[],
        future_signals=["temperature_c", "footfall_index", "festival", "holiday", "promotion_active"],
        signal_labels={"temperature_c": "Temperature", "footfall_index": "Footfall", "festival": "Festival", "holiday": "Holiday", "promotion_active": "Promotion"},
    ),
    "Revenue": ScenarioConfig(
        name="Revenue", company="Frostline Creamery", file="retail_sales_revenue_v3.csv",
        date_col="date", target_col="revenue", series_col="product_name", unit="₹", horizon_default=30,
        past_only_signals=[],
        future_signals=["temperature_c", "footfall_index", "festival", "holiday", "promotion_active"],
        signal_labels={"temperature_c": "Temperature", "footfall_index": "Footfall", "festival": "Festival", "holiday": "Holiday", "promotion_active": "Promotion"},
    ),
    "Inventory": ScenarioConfig(
        name="Inventory", company="Apex Motors", file="automotive_inventory_v1.csv",
        date_col="date", target_col="component_demand", series_col="component_id", unit="components", horizon_default=30,
        past_only_signals=["supplier_lead_time_days", "supplier_disruption"],
        future_signals=["planned_production_units", "scheduled_receipts", "plant_maintenance", "festival"],
        signal_labels={"planned_production_units": "Planned Production", "scheduled_receipts": "Scheduled Receipts", "plant_maintenance": "Plant Maintenance", "festival": "Festival", "supplier_lead_time_days": "Supplier Lead Time", "supplier_disruption": "Supplier Disruption"},
    ),
    "Viewership": ScenarioConfig(
        name="Viewership", company="StreamPulse", file="ott_viewership_v1.csv",
        date_col="date", target_col="watch_hours", series_col="content_id", unit="hours", horizon_default=30,
        past_only_signals=["competitor_major_release"],
        future_signals=["release_active", "marketing_campaign", "festival", "holiday"],
        signal_labels={"release_active": "Release Schedule", "marketing_campaign": "Marketing Campaign", "festival": "Festival", "holiday": "Holiday", "competitor_major_release": "Competitor Release"},
    ),
}

CHECKPOINT = "google/timesfm-3.0-pytorch"
QUANTILE_LEVELS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]