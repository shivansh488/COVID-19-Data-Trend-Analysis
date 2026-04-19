"""
data_ingest.py — Load, clean, and reshape the Johns Hopkins CSSE COVID-19 data.

Public API
----------
load_all()           → dict with keys 'confirmed', 'deaths', 'recovered' (long DataFrames)
load_merged()        → single DataFrame with columns [date, country, confirmed, deaths, recovered, daily_confirmed, daily_deaths]
"""

import os
import pandas as pd
import numpy as np

# ── Paths ────────────────────────────────────────────────────────────────────
_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "raw")

_CSV_MAP = {
    "confirmed": "time_series_covid19_confirmed_global.csv",
    "deaths":    "time_series_covid19_deaths_global.csv",
    "recovered": "time_series_covid19_recovered_global.csv",
}


# ── Helpers ──────────────────────────────────────────────────────────────────

def _wide_to_long(df: pd.DataFrame, value_name: str) -> pd.DataFrame:
    """Melt the JHU wide-format CSV into a tidy long-format DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Raw wide-format DataFrame straight from the CSV.
    value_name : str
        Name for the value column (e.g. 'confirmed', 'deaths').

    Returns
    -------
    pd.DataFrame with columns [country, date, <value_name>]
    """
    # Identify date columns (everything that is NOT Province/State, Country/Region, Lat, Long)
    id_vars = ["Province/State", "Country/Region", "Lat", "Long"]
    date_cols = [c for c in df.columns if c not in id_vars]

    melted = df.melt(
        id_vars=id_vars,
        value_vars=date_cols,
        var_name="date",
        value_name=value_name,
    )

    melted["date"] = pd.to_datetime(melted["date"], format="mixed")
    melted.rename(columns={"Country/Region": "country"}, inplace=True)

    # Aggregate provinces → country level
    aggregated = (
        melted.groupby(["country", "date"], as_index=False)[value_name]
        .sum()
    )
    return aggregated


# ── Public API ───────────────────────────────────────────────────────────────

def load_single(metric: str, data_dir: str | None = None) -> pd.DataFrame:
    """Load and reshape one metric (confirmed / deaths / recovered).

    Parameters
    ----------
    metric : str
        One of 'confirmed', 'deaths', 'recovered'.
    data_dir : str, optional
        Override the default raw data directory.

    Returns
    -------
    pd.DataFrame with columns [country, date, <metric>]
    """
    if metric not in _CSV_MAP:
        raise ValueError(f"Unknown metric '{metric}'. Choose from {list(_CSV_MAP)}")

    path = os.path.join(data_dir or _DATA_DIR, _CSV_MAP[metric])
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Dataset file not found: {path}\n"
            "Run `python data/download_data.py` first."
        )

    raw = pd.read_csv(path)
    return _wide_to_long(raw, metric)


def load_all(data_dir: str | None = None) -> dict[str, pd.DataFrame]:
    """Load all three metrics as separate DataFrames.

    Returns
    -------
    dict  {'confirmed': DataFrame, 'deaths': DataFrame, 'recovered': DataFrame}
    """
    return {m: load_single(m, data_dir) for m in _CSV_MAP}


def load_merged(data_dir: str | None = None) -> pd.DataFrame:
    """Load and merge all three metrics into a single DataFrame.

    Additional engineered columns
    -----------------------------
    - daily_confirmed : day-over-day new confirmed cases (clipped ≥ 0)
    - daily_deaths    : day-over-day new deaths (clipped ≥ 0)

    Returns
    -------
    pd.DataFrame with columns [country, date, confirmed, deaths, recovered,
                               daily_confirmed, daily_deaths]
    """
    frames = load_all(data_dir)

    merged = frames["confirmed"]
    for metric in ["deaths", "recovered"]:
        merged = merged.merge(frames[metric], on=["country", "date"], how="left")

    # Fill any missing recovered values (dataset stopped updating recovered early)
    merged["recovered"] = merged["recovered"].fillna(0).astype(int)

    # Sort for correct diff()
    merged.sort_values(["country", "date"], inplace=True)
    merged.reset_index(drop=True, inplace=True)

    # Daily new cases / deaths
    for col in ["confirmed", "deaths"]:
        daily_col = f"daily_{col}"
        merged[daily_col] = merged.groupby("country")[col].diff().fillna(0)
        merged[daily_col] = merged[daily_col].clip(lower=0).astype(int)

    return merged


def get_country_list(df: pd.DataFrame) -> list[str]:
    """Return sorted list of unique countries in the dataset."""
    return sorted(df["country"].unique().tolist())


def filter_countries(df: pd.DataFrame, countries: list[str]) -> pd.DataFrame:
    """Filter the DataFrame to only the specified countries."""
    return df[df["country"].isin(countries)].copy()


def get_top_countries(df: pd.DataFrame, metric: str = "confirmed", n: int = 20) -> list[str]:
    """Return the top N countries by the maximum cumulative value of a metric."""
    latest = df.groupby("country")[metric].max().nlargest(n)
    return latest.index.tolist()
