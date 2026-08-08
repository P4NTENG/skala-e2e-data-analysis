# NYC Yellow Taxi — RatecodeID Prediction

[![Python](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.9-orange.svg)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

End-to-end data analysis and machine learning project predicting NYC Yellow Taxi `RatecodeID` using trip metadata. Built with Pandas, Polars, Seaborn, Plotly, scikit-learn Pipeline.

## Dataset

[NYC Yellow Taxi Trip Data (2026-05)](https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2026-05.parquet)

| | Original | Cleaned |
|---|:---:|:---:|
| Rows | 4,090,836 | 3,021,962 |
| Columns | 20 | 11 (feature subset) |
| Target | `RatecodeID` | 6-class |

## Quick Start

```bash
git clone https://github.com/P4NTENG/skala-e2e-data-analysis.git
cd skala-e2e-data-analysis

python -m venv .venv
source .venv/bin/activate   # or .venv\Scripts\Activate.ps1

pip install -r requirements.txt

# Download dataset into data/
# https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2026-05.parquet

python src/4_ml_pipeline.py
```

## Usage

| Script | Description |
|--------|-------------|
| `src/2_visualization.py` | EDA + statistical analysis charts (Seaborn / Plotly) |
| `src/4_ml_pipeline.py` | ML pipeline: preprocessing → training → evaluation → visualization |

## Project Structure

```
├── data/                    # Dataset (gitignored)
├── output/
│   ├── report.md            # Full results report
│   ├── model_hgb.joblib     # HistGradientBoosting model
│   ├── model_rf.joblib      # RandomForest model
│   └── figures/             # Charts (PNG + interactive HTML)
├── src/
│   ├── explore_distribution.py
│   ├── 2_visualization.py
│   └── 4_ml_pipeline.py
├── requirements.txt
└── README.md
```

## Features

| Feature | Type | Description |
|---------|------|-------------|
| `PULocationID` | categorical (265) | Pickup location |
| `DOLocationID` | categorical (265) | Dropoff location |
| `VendorID` | binary | Taxi vendor |
| `payment_type` | categorical | Payment method |
| `passenger_count` | numeric | Number of passengers |
| `store_and_fwd_flag` | binary | Trip stored in vehicle memory |
| `hour` | numeric | Pickup hour (0-23) |
| `day_of_week` | numeric | Day of week (0-6) |
| `is_weekend` | binary | Weekend flag |

## Models

Two classifiers with `sklearn.pipeline.Pipeline`:

| Model | CV Balanced Accuracy | Test Weighted F1 |
|--------|:---:|:---:|
| RandomForest | 0.7487 | 0.8042 |
| **HistGradientBoosting** | **0.8207** | **0.9090** |

### Per-Class F1 (HistGradientBoosting)

| Standard | JFK | Unknown | Nassau | Negotiated | Newark |
|:---:|:---:|:---:|:---:|:---:|:---:|
| 0.93 | 0.80 | 0.73 | 0.63 | 0.19 | 0.11 |

## Key Findings

- Geographic features (`DOLocationID` + `PULocationID`) account for **64%** of feature importance
- `RatecodeID=99` (Unknown) is perfectly characterized by `VendorID=1 ∩ payment_type=1 ∩ passenger_count=1`
- All features show statistically significant differences across rate codes (p < 0.001, Welch's t-test)
- Minority classes (Newark 0.4%, Negotiated 0.7%) suffer from data scarcity

## Documentation

- [`output/report.md`](output/report.md) — Full results with embedded charts
- [`DATA_CLEANING_NOTES.md`](DATA_CLEANING_NOTES.md) — Data cleaning, feature analysis, statistical tests

## Dependencies

```
pandas >= 3.0
polars >= 1.0
numpy >= 2.0
seaborn >= 0.13
matplotlib >= 3.7
plotly >= 5.15
scipy >= 1.10
scikit-learn >= 1.3
joblib >= 1.3
pyarrow >= 12.0
kaleido >= 0.2
```
