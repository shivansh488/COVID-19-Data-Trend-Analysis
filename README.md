# COVID-19 Data Trend Analysis

> **Subject:** Data Mining & Data Warehouse

## Overview

This project studies COVID-19 datasets to analyze trends in confirmed cases, recoveries, and deaths. Statistical methods, data mining techniques, and visualization tools are used to observe how the pandemic spread over time.

## Key Features

- **Data Preprocessing** — Cleaning, reshaping, and normalizing Johns Hopkins CSSE time-series data.
- **Exploratory Data Analysis** — Summary statistics, distribution plots, top-N country rankings.
- **Standard Techniques** — Rolling averages, STL decomposition, Pearson/lagged correlation, K-Means clustering, Apriori pattern mining.
- **Novel Technique: Epidemic Trajectory Fingerprinting (ETF)** — A hybrid pipeline combining Wavelet Decomposition → DTW → Spectral Embedding → HDBSCAN that does not exist as a standard method.
- **Interactive Visualizations** — Plotly line charts, choropleth maps, correlation heatmaps, cluster scatter plots.

## Dataset

**Johns Hopkins CSSE COVID-19 Time-Series**
- Source: https://github.com/CSSEGISandData/COVID-19
- Files: `time_series_covid19_confirmed_global.csv`, `time_series_covid19_deaths_global.csv`, `time_series_covid19_recovered_global.csv`
- Coverage: 200+ countries/regions, Jan 22 2020 — Mar 9 2023

## Setup

```bash
# 1. Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac

# 2. Install dependencies
pip install -r requirements.txt

# 3. Download dataset
python data/download_data.py

# 4. Launch notebook
jupyter notebook notebooks/COVID19_Trend_Analysis.ipynb
```

## Project Structure

```
COVID-19 Data Trend Analysis/
├── README.md
├── requirements.txt
├── data/
│   ├── download_data.py
│   └── raw/
├── notebooks/
│   └── COVID19_Trend_Analysis.ipynb
├── src/
│   ├── __init__.py
│   ├── data_ingest.py
│   ├── eda.py
│   ├── standard_techniques.py
│   ├── etf.py
│   └── visualization.py
└── tests/
    ├── test_ingest.py
    └── test_etf.py
```

## Techniques Used

| Category | Technique |
|----------|-----------|
| Preprocessing | Forward-fill, interpolation, min-max scaling |
| Time-Series | Daily growth rate, 7/14-day rolling average, STL decomposition |
| Correlation | Pearson matrix, lagged cross-correlation |
| Clustering | K-Means (with Elbow + Silhouette), HDBSCAN |
| Pattern Mining | Apriori on discretized case bands |
| **Novel (ETF)** | Wavelet + DTW + Spectral Embedding + HDBSCAN |

## License

This project is for academic purposes. Dataset © Johns Hopkins University.
