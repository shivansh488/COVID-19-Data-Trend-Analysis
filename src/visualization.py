"""
visualization.py — Plotting functions for COVID-19 analysis.

Uses Plotly for interactive charts and Matplotlib/Seaborn for static plots.
"""

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns


# ── Plotly Interactive Charts ────────────────────────────────────────────────

def plot_cumulative(df, countries, metric="confirmed", title=None):
    """Interactive line chart of cumulative metric over time."""
    subset = df[df["country"].isin(countries)]
    fig = px.line(subset, x="date", y=metric, color="country",
                  title=title or f"Cumulative {metric.title()} Cases",
                  labels={"date": "Date", metric: metric.title()})
    fig.update_layout(template="plotly_dark", hovermode="x unified")
    return fig


def plot_daily(df, countries, metric="daily_confirmed",
               rolling_window=7, title=None):
    """Daily cases with rolling average overlay."""
    fig = go.Figure()
    for c in countries:
        cdf = df[df["country"] == c].copy()
        cdf["ma"] = cdf[metric].rolling(rolling_window, min_periods=1).mean()
        fig.add_trace(go.Scatter(
            x=cdf["date"], y=cdf[metric], name=f"{c} (raw)",
            opacity=0.3, line=dict(width=1)))
        fig.add_trace(go.Scatter(
            x=cdf["date"], y=cdf["ma"], name=f"{c} ({rolling_window}d avg)",
            line=dict(width=2)))
    fig.update_layout(
        title=title or f"Daily {metric.replace('daily_', '').title()} Cases",
        template="plotly_dark", hovermode="x unified",
        xaxis_title="Date", yaxis_title="Count")
    return fig


def plot_choropleth(df, date=None, metric="confirmed", title=None):
    """World choropleth map for a given date (or latest)."""
    if date is None:
        date = df["date"].max()
    snap = df[df["date"] == date].copy()
    fig = px.choropleth(
        snap, locations="country", locationmode="country names",
        color=metric, hover_name="country",
        color_continuous_scale="YlOrRd",
        title=title or f"{metric.title()} Cases — {date.strftime('%Y-%m-%d')}")
    fig.update_layout(template="plotly_dark")
    return fig


def plot_top_countries_bar(df, metric="confirmed", n=20, title=None):
    """Horizontal bar chart of top N countries."""
    top = df.groupby("country")[metric].max().nlargest(n).reset_index()
    fig = px.bar(top, x=metric, y="country", orientation="h",
                 title=title or f"Top {n} Countries by {metric.title()}",
                 color=metric, color_continuous_scale="Teal")
    fig.update_layout(template="plotly_dark", yaxis=dict(autorange="reversed"))
    return fig


# ── Seaborn / Matplotlib Static Plots ───────────────────────────────────────

def plot_correlation_heatmap(corr_matrix, title=None, figsize=(12, 10)):
    """Seaborn heatmap of the correlation matrix."""
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(corr_matrix, annot=False, cmap="coolwarm", center=0,
                linewidths=0.5, ax=ax)
    ax.set_title(title or "Correlation Matrix of Daily Cases")
    plt.tight_layout()
    return fig


def plot_stl(stl_result, country, figsize=(14, 8)):
    """Plot STL decomposition panels."""
    fig, axes = plt.subplots(4, 1, figsize=figsize, sharex=True)
    panels = [
        ("Observed", stl_result.observed),
        ("Trend", stl_result.trend),
        ("Seasonal", stl_result.seasonal),
        ("Residual", stl_result.resid),
    ]
    for ax, (label, data) in zip(axes, panels):
        ax.plot(data, linewidth=0.8)
        ax.set_ylabel(label)
    axes[0].set_title(f"STL Decomposition — {country}")
    plt.tight_layout()
    return fig


def plot_elbow(elbow_df, figsize=(10, 4)):
    """Dual-axis elbow plot: inertia + silhouette."""
    fig, ax1 = plt.subplots(figsize=figsize)
    ax1.plot(elbow_df["k"], elbow_df["inertia"], "b-o", label="Inertia")
    ax1.set_xlabel("K")
    ax1.set_ylabel("Inertia", color="b")
    ax2 = ax1.twinx()
    ax2.plot(elbow_df["k"], elbow_df["silhouette"], "r-s", label="Silhouette")
    ax2.set_ylabel("Silhouette Score", color="r")
    ax1.set_title("Elbow Method — K-Means")
    fig.tight_layout()
    return fig


def plot_kmeans_clusters(pca_2d, labels, countries, figsize=(10, 8)):
    """2D scatter of K-Means clusters (PCA projection)."""
    fig, ax = plt.subplots(figsize=figsize)
    scatter = ax.scatter(pca_2d[:, 0], pca_2d[:, 1], c=labels,
                         cmap="tab10", s=60, alpha=0.8, edgecolors="w")
    for i, c in enumerate(countries):
        ax.annotate(c, (pca_2d[i, 0], pca_2d[i, 1]),
                    fontsize=6, alpha=0.7)
    ax.set_title("K-Means Clusters (PCA 2D)")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    plt.colorbar(scatter, ax=ax, label="Cluster")
    plt.tight_layout()
    return fig


# ── ETF Novel Technique Plots ───────────────────────────────────────────────

def plot_dtw_heatmap(dtw_matrix, countries, figsize=(12, 10)):
    """Heatmap of DTW distance matrix."""
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(pd.DataFrame(dtw_matrix, index=countries, columns=countries),
                cmap="viridis_r", ax=ax, linewidths=0.3)
    ax.set_title("DTW Distance Matrix (Wavelet-Transformed)")
    plt.tight_layout()
    return fig


def plot_etf_clusters(results_df, figsize=(10, 8)):
    """Scatter plot of ETF spectral embedding colored by HDBSCAN cluster."""
    fig, ax = plt.subplots(figsize=figsize)
    unique_labels = sorted(results_df["cluster"].unique())
    cmap = plt.cm.get_cmap("tab10", len(unique_labels))

    for cl in unique_labels:
        mask = results_df["cluster"] == cl
        label = f"Cluster {cl}" if cl >= 0 else "Noise"
        ax.scatter(results_df.loc[mask, "emb_0"],
                   results_df.loc[mask, "emb_1"],
                   label=label, s=60, alpha=0.8, edgecolors="w")

    for _, row in results_df.iterrows():
        ax.annotate(row["country"], (row["emb_0"], row["emb_1"]),
                    fontsize=6, alpha=0.7)

    ax.set_title("ETF — Epidemic Trajectory Fingerprinting Clusters")
    ax.set_xlabel("Spectral Dimension 1")
    ax.set_ylabel("Spectral Dimension 2")
    ax.legend(fontsize=8)
    plt.tight_layout()
    return fig


def plot_cluster_curves(curves, results_df, figsize=(16, 10)):
    """Grid of subplots: one per cluster, showing member curves."""
    clusters = sorted([c for c in results_df["cluster"].unique() if c >= 0])
    n = len(clusters)
    if n == 0:
        return None
    cols = min(3, n)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=figsize, sharex=True)
    if n == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    for idx, cl in enumerate(clusters):
        ax = axes[idx]
        members = results_df[results_df["cluster"] == cl]["country"].tolist()
        for m in members:
            ax.plot(curves[m], alpha=0.6, linewidth=0.8, label=m)
        ax.set_title(f"Cluster {cl} ({len(members)} countries)", fontsize=10)
        ax.legend(fontsize=5, ncol=2)

    for idx in range(n, len(axes)):
        axes[idx].set_visible(False)

    fig.suptitle("ETF — Trajectory Shapes by Cluster", fontsize=14)
    plt.tight_layout()
    return fig
