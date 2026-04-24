"""
standard_techniques.py — Classical data mining techniques for COVID-19 analysis.

Includes: Rolling averages, STL decomposition, Correlation, K-Means, Apriori.
"""

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from statsmodels.tsa.seasonal import STL
from mlxtend.frequent_patterns import apriori, association_rules


# ── 1. Rolling Averages ─────────────────────────────────────────────────────

def rolling_average(df, country, metric="daily_confirmed", windows=None):
    """Compute rolling averages for a country. Returns DataFrame."""
    if windows is None:
        windows = [7, 14]
    cdf = df.loc[df["country"] == country, ["date", metric]].set_index("date").copy()
    cdf.rename(columns={metric: "raw"}, inplace=True)
    for w in windows:
        cdf[f"ma_{w}"] = cdf["raw"].rolling(window=w, min_periods=1).mean()
    return cdf


# ── 2. STL Decomposition ────────────────────────────────────────────────────

def stl_decompose(df, country, metric="daily_confirmed", period=7):
    """Run STL decomposition. Returns statsmodels STL result."""
    series = (
        df.loc[df["country"] == country, ["date", metric]]
        .set_index("date")[metric].asfreq("D").fillna(0)
    )
    return STL(series, period=period, robust=True).fit()


# ── 3. Correlation Analysis ─────────────────────────────────────────────────

def correlation_matrix(df, countries, metric="daily_confirmed"):
    """Pearson correlation matrix across countries."""
    pivot = df[df["country"].isin(countries)].pivot_table(
        index="date", columns="country", values=metric, fill_value=0
    )
    return pivot.corr()


def lagged_cross_correlation(df, country_a, country_b,
                             metric="daily_confirmed", max_lag=30):
    """Cross-correlation at lags -max_lag … +max_lag. Returns Series."""
    a = df.loc[df["country"] == country_a].set_index("date")[metric].fillna(0)
    b = df.loc[df["country"] == country_b].set_index("date")[metric].fillna(0)
    idx = a.index.intersection(b.index)
    a, b = a.loc[idx].values, b.loc[idx].values
    corrs = {}
    for lag in range(-max_lag, max_lag + 1):
        if lag >= 0:
            r = np.corrcoef(a[:len(a) - lag], b[lag:])[0, 1]
        else:
            r = np.corrcoef(a[-lag:], b[:len(b) + lag])[0, 1]
        corrs[lag] = r
    return pd.Series(corrs, name=f"{country_a} vs {country_b}")


# ── 4. K-Means Clustering ───────────────────────────────────────────────────

def build_country_features(df):
    """Build per-country feature matrix for clustering."""
    g = df.groupby("country")
    features = pd.DataFrame({
        "total_confirmed": g["confirmed"].max(),
        "total_deaths": g["deaths"].max(),
        "peak_daily_confirmed": g["daily_confirmed"].max(),
        "peak_daily_deaths": g["daily_deaths"].max(),
    })

    def _days_to_peak(sub):
        if sub["daily_confirmed"].max() == 0:
            return 0
        peak_date = sub.loc[sub["daily_confirmed"].idxmax(), "date"]
        return (peak_date - sub["date"].min()).days

    features["days_to_peak"] = g.apply(_days_to_peak, include_groups=False)
    features["fatality_rate"] = (
        features["total_deaths"] / features["total_confirmed"].replace(0, 1)
    ).clip(0, 1)
    total_rec = g["recovered"].max()
    features["recovery_rate"] = (
        total_rec / features["total_confirmed"].replace(0, 1)
    ).clip(0, 1)
    return features.fillna(0)


def kmeans_cluster(features, k=5):
    """Run K-Means. Returns (labels, pca_2d, silhouette_score)."""
    scaler = StandardScaler()
    X = scaler.fit_transform(features.values)
    km = KMeans(n_clusters=k, n_init=10, random_state=42)
    labels = km.fit_predict(X)
    pca = PCA(n_components=2, random_state=42)
    pca_2d = pca.fit_transform(X)
    sil = silhouette_score(X, labels)
    return labels, pca_2d, sil


def elbow_search(features, k_range=None):
    """Elbow method: returns DataFrame [k, inertia, silhouette]."""
    if k_range is None:
        k_range = range(2, 11)
    scaler = StandardScaler()
    X = scaler.fit_transform(features.values)
    records = []
    for k in k_range:
        km = KMeans(n_clusters=k, n_init=10, random_state=42)
        km.fit(X)
        sil = silhouette_score(X, km.labels_)
        records.append({"k": k, "inertia": km.inertia_, "silhouette": sil})
    return pd.DataFrame(records)


# ── 5. Apriori Association Mining ────────────────────────────────────────────

def discretize_cases(df, countries, metric="daily_confirmed"):
    """Discretize daily cases into bands and return one-hot DataFrame."""
    bin_edges = [0, 100, 1000, 10_000, np.inf]
    bin_labels = ["Low", "Medium", "High", "Surge"]
    subset = df[df["country"].isin(countries)][["date", "country", metric]].copy()
    subset["band"] = pd.cut(subset[metric], bins=bin_edges,
                            labels=bin_labels, include_lowest=True)
    subset["item"] = subset["country"] + "_" + subset["band"].astype(str)
    onehot = subset.pivot_table(index="date", columns="item",
                                aggfunc="size", fill_value=0)
    return (onehot > 0)


def run_apriori(onehot, min_support=0.05, min_confidence=0.5):
    """Run Apriori and return association rules."""
    freq = apriori(onehot, min_support=min_support, use_colnames=True)
    if freq.empty:
        return pd.DataFrame()
    rules = association_rules(freq, metric="confidence",
                              min_threshold=min_confidence)
    return rules.sort_values("lift", ascending=False)
