---

title: ScoopCast Demand Planner
emoji: ◈
colorFrom: pink
colorTo: indigo
sdk: streamlit
sdk_version: "1.32.0"
app_file: app.py
pinned: false
-------------

# ScoopCast — Demand Planner Powered by TimesFM 3.0

An executive-level, probabilistic demand planning and inventory management platform powered by Google's **TimesFM 3.0 PyTorch zero-shot foundation model**.

ScoopCast bridges the gap between raw probabilistic time-series forecasts and actionable executive decision-making across four operational domains:

* **Sales**
* **Revenue**
* **Inventory (BOM / Automotive)**
* **Viewership**

---

## 🌟 Key Features

### 📈 Multi-Domain Forecasting

* **Sales & Revenue**

  * Retail demand forecasting
  * Promotional lift simulations
  * Revenue projections

* **Inventory**

  * Forward-roll warehouse stock projection
  * Safety buffer calculations using supplier lead time and demand volatility
  * Automated replenishment deficit alerts
  * BOM / automotive component planning

* **Viewership**

  * Media engagement forecasting
  * Watch-hour projections
  * Content-category level demand forecasting

### 📊 Probabilistic Quantiles

TimesFM 3.0 forecasts are presented using probabilistic prediction ranges:

* **P10** — Conservative scenario
* **P50** — Expected / median forecast
* **P90** — Optimistic scenario

This allows planners to reason about uncertainty rather than relying on a single point forecast.

### 🎯 What-If Scenario Planner

Interactive scenario planning allows users to:

* Define promotion schedules
* Adjust promotion intensity
* Compare baseline forecasts against promotional scenarios
* Visualize scenario differences through interactive Plotly charts

### 🔄 Dynamic Resampling

ScoopCast supports multiple forecasting granularities:

* Daily
* Weekly (`W-MON`)
* Monthly (`MS`)

Aggregation follows domain-specific accounting rules:

* **Demand / sales:** Sum
* **Revenue:** Sum
* **Inventory / shelf stock:** Last closing balance

### 🧠 Executive Intelligence & Reporting

The application converts forecast outputs into actionable summaries through:

* Plain-English automated takeaways
* Risk and opportunity observations
* Inventory deficit identification
* One-click printable HTML executive briefs

### 📁 Custom CSV Ingestion

Users can upload their own datasets through a drag-and-drop interface.

Features include:

* Intelligent column resolution
* Flexible CSV structure
* Optional covariate detection
* Session-isolated data processing
* No runtime model fine-tuning

---

## 📊 Dataset Format

### Standard Input Format

The minimum required CSV schema is:

```csv
date,sales
2026-01-01,1234
2026-01-02,1287
2026-01-03,1198
```

The `date` column represents the time dimension, while `sales` represents the primary target series.

### Supported Optional Covariates

#### Retail Signals

```text
units_sold
average_price
promotion
promotion_intensity
holiday
footfall_index
temperature_c
```

#### Inventory / BOM Signals

```text
component_id
planned_production_units
units_per_vehicle
supplier_lead_time_days
scheduled_receipts
closing_inventory
current_inventory
```

Additional columns may be resolved automatically by the application's data manager where applicable.

---

## 📁 Project Structure

```text
scoopcast-demand-planner/
│
├── app.py
│   └── Main Streamlit dashboard interface
│
├── config.py
│   └── Scenario definitions and model constants
│
├── requirements.txt
│   └── PyTorch and application dependencies
│
├── README.md
│   └── Project documentation and Hugging Face Spaces configuration
│
├── forecast/
│   └── timesfm3_engine.py
│       └── TimesFM 3.0 PyTorch model adapter
│
├── scenarios/
│   ├── retail.py
│   │   └── Retail Sales/Revenue and What-If scenario engine
│   │
│   ├── inventory.py
│   │   └── BOM and forward-roll supply chain accounting
│   │
│   └── viewership.py
│       └── Content viewership forecasting
│
├── dashboard/
│   ├── charts.py
│   │   └── Plotly time-series and inventory visualizations
│   │
│   └── insights.py
│       └── Executive takeaways and HTML report generator
│
└── data/
    └── data_manager.py
        └── CSV loader and covariate builder
```

---

## ⚙️ Local Setup & Installation

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/scoopcast-demand-planner.git
cd scoopcast-demand-planner
```

Replace `your-username` with your GitHub username if you are publishing the project to your own repository.

### 2. Create a Virtual Environment

ScoopCast is designed to run with Python 3.11 or 3.12.

```bash
python3 -m venv venv
```

Activate the environment:

**Linux / macOS**

```bash
source venv/bin/activate
```

**Windows**

```bash
venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the Streamlit Application

```bash
python -m streamlit run app.py
```

The application will then be available through the local Streamlit server.

---

## 🤖 Model Behavior & Execution Contract

### Zero-Shot Inference

ScoopCast uses the pretrained **TimesFM 3.0 PyTorch** model:

```text
google/timesfm-3.0-pytorch
```

The model is used in a **zero-shot inference configuration**.

User-uploaded historical data is provided as inference-time context. The application does **not** fine-tune the foundation model at runtime.

The configured historical context can use up to **512 time steps**, depending on the available input data and application configuration.

### Forecast Horizon

ScoopCast supports extended forecasting scenarios of up to **90 days**, subject to the application's configured horizon and available covariate information.

### Covariate Alignment

Past and future covariates are structured to maintain alignment across:

```text
Context Length + Forecast Horizon
```

This ensures that the model receives the required covariate information across the complete forecasting window.

---

## ⚠️ Interpretability & Causal Claims

ScoopCast's What-If scenarios should be interpreted as **model scenario responses**, not as proven causal effects.

For example, if increasing promotional intensity produces a higher forecast, this represents a difference between model scenarios under different inputs.

It does **not** independently establish that the promotion caused the resulting change in demand.

Therefore:

> Scenario differences represent model-based projections and should not be interpreted as causal estimates without an appropriate causal inference design.

---

## 🏭 Inventory Planning Logic

For inventory-focused use cases, ScoopCast combines demand forecasts with operational supply-chain inputs such as:

* Current / closing inventory
* Planned production
* Units required per vehicle
* Supplier lead time
* Scheduled receipts
* Demand uncertainty

The resulting forward-roll projection can be used to identify:

* Potential inventory shortages
* Replenishment requirements
* Safety-buffer requirements
* Future component deficits
* Supply-chain risk windows

The inventory calculations are designed to complement the probabilistic demand forecast rather than replace operational planning systems.

---

## 📈 Forecast Interpretation

A typical forecast can be interpreted as:

```text
                 P90 ───────────── Optimistic demand
                /
Historical ──── P50 ───────────── Expected demand
                \
                 P10 ───────────── Conservative demand
```

The prediction range allows planners to consider multiple demand outcomes when making inventory and operational decisions.

---

## 🎯 Intended Use Cases

ScoopCast is designed for scenarios such as:

| Domain             | Example Use Case                              |
| ------------------ | --------------------------------------------- |
| Sales              | Retail demand forecasting                     |
| Revenue            | Revenue planning and scenario analysis        |
| Promotions         | Promotional lift scenario planning            |
| Inventory          | Stock depletion and replenishment planning    |
| Automotive         | BOM/component demand forecasting              |
| Supply Chain       | Supplier lead-time and safety-buffer analysis |
| Media              | Content viewership forecasting                |
| Executive Planning | Forecast summaries and decision briefs        |

---

## 🧩 Technology Stack

| Component         | Technology                      |
| ----------------- | ------------------------------- |
| Application       | Streamlit                       |
| Forecasting Model | Google TimesFM 3.0              |
| Model Framework   | PyTorch                         |
| Visualization     | Plotly                          |
| Data Input        | CSV                             |
| Reporting         | HTML                            |
| Deployment        | Streamlit / Hugging Face Spaces |

---

## 🚀 Deployment

The repository is structured to support deployment as a **Streamlit application on Hugging Face Spaces**.

The repository metadata at the top of this README specifies:

```yaml
sdk: streamlit
sdk_version: "1.32.0"
app_file: app.py
```

Ensure that all required dependencies are included in `requirements.txt` before deployment.

---

## 📄 License & Attribution

ScoopCast is powered by **Google TimesFM 3.0 PyTorch**.

The application is designed as an executive demand-planning and supply-chain intelligence layer that transforms probabilistic time-series forecasts into operational planning insights.

**Model:** `google/timesfm-3.0-pytorch`

**Built with:** Streamlit · PyTorch · Plotly · TimesFM 3.0
