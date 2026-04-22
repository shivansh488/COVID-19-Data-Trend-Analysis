"""
eda.py — Exploratory Data Analysis helpers for COVID-19 data.

Provides summary statistics, distribution analysis, and top-N rankings.
"""

import pandas as pd
import numpy as np


def summary_statistics(df: pd.DataFrame, metric: str = "confirmed") -> pd.DataFrame:
    """Compute per-country summary statistics for a given metric.

    Returns a DataFrame indexed by country with columns:
    total, max_daily, mean_daily, median_daily, std_daily, skewness
    """
    daily_col = f"daily_{metric}" if f"daily_{metric}" in df.columns else metric

    stats = df.groupby("country").agg(
        total=(metric, "max"),
        max_daily=(daily_col, "max"),
        mean_daily=(daily_col, "mean"),
        median_daily=(daily_col, "median"),
        std_daily=(daily_col, "std"),
    )

    # Skewness
    skew = df.groupby("country")[daily_col].skew()
    stats["skewness"] = skew

    stats.sort_values("total", ascending=False, inplace=True)
    return stats.round(2)


def top_countries_table(df: pd.DataFrame, n: int = 20) -> pd.DataFrame:
    """Return a table of top N countries by confirmed, deaths, and recovered.

    Columns: country, total_confirmed, total_deaths, total_recovered, fatality_rate (%)
    """
    latest = df.groupby("country").agg(
        total_confirmed=("confirmed", "max"),
        total_deaths=("deaths", "max"),
        total_recovered=("recovered", "max"),
    ).nlargest(n, "total_confirmed")

    latest["fatality_rate_%"] = (
        (latest["total_deaths"] / latest["total_confirmed"]) * 100
    ).round(2)

    latest.reset_index(inplace=True)
    return latest


def daily_distribution(df: pd.DataFrame, country: str, metric: str = "daily_confirmed") -> dict:
    """Return distribution stats for a country's daily metric.

    Returns dict with: mean, median, std, min, max, q25, q75, iqr
    """
    series = df.loc[df["country"] == country, metric].dropna()
    return {
        "mean": round(series.mean(), 2),
        "median": round(series.median(), 2),
        "std": round(series.std(), 2),
        "min": int(series.min()),
        "max": int(series.max()),
        "q25": round(series.quantile(0.25), 2),
        "q75": round(series.quantile(0.75), 2),
        "iqr": round(series.quantile(0.75) - series.quantile(0.25), 2),
    }


def compute_growth_rate(df: pd.DataFrame, country: str, metric: str = "confirmed") -> pd.Series:
    """Compute daily growth rate for a country: (C_t - C_{t-1}) / C_{t-1}.

    Returns a Series indexed by date. Infinite / NaN values are replaced with 0.
    """
    cdf = df.loc[df["country"] == country].set_index("date")[metric]
    rate = cdf.pct_change().replace([np.inf, -np.inf], 0).fillna(0)
    return rate


def compute_doubling_time(growth_rate: pd.Series) -> pd.Series:
    """Estimate doubling time from daily growth rate: ln(2) / ln(1 + r).

    Returns NaN where r ≤ 0.
    """
    r = growth_rate.replace(0, np.nan)
    return (np.log(2) / np.log(1 + r)).replace([np.inf, -np.inf], np.nan)
