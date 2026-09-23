from __future__ import annotations

import numpy as np
import pandas as pd

rng = np.random.default_rng(42)
dates = pd.date_range("2025-09-01", "2026-09-01", freq="D")
n = len(dates)
t = np.arange(n)

weekly = 1 + 0.10 * (dates.dayofweek >= 5).astype(float)
trend = 1 + 0.0009 * t
annualish = 1 + 0.06 * np.sin(2 * np.pi * t / 365)
promo = np.zeros(n)
for start, end, intensity in [
    ("2025-10-10", "2025-10-16", 0.18),
    ("2025-11-20", "2025-11-30", 0.25),
    ("2025-12-18", "2025-12-31", 0.30),
    ("2026-02-12", "2026-02-18", 0.20),
    ("2026-05-01", "2026-05-07", 0.22),
    ("2026-08-10", "2026-08-16", 0.20),
]:
    mask = (dates >= pd.Timestamp(start)) & (dates <= pd.Timestamp(end))
    promo[mask] = intensity

holiday_dates = pd.to_datetime(["2025-10-20", "2025-11-01", "2025-12-25", "2026-01-01", "2026-03-14", "2026-08-15"])
holiday = dates.isin(holiday_dates).astype(float)
base = 1_150_000 * trend * annualish * weekly
sales = base * (1 + promo) * (1 + 0.12 * holiday) + rng.normal(0, 45_000, n)
sales = np.maximum(sales, 0)

units = sales / (3900 + 120 * np.sin(2 * np.pi * t / 30))
price = sales / units

sample = pd.DataFrame({
    "date": dates,
    "sales": sales.round(0),
    "units_sold": units.round(0),
    "average_price": price.round(0),
    "promotion": (promo > 0).astype(int),
    "promotion_intensity": promo,
    "holiday": holiday,
})
sample.to_csv("/mnt/data/timesfm_retail_mvp/sample_retail_sales.csv", index=False)
print(sample.head())
