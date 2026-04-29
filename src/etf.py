"""
etf.py — Epidemic Trajectory Fingerprinting (ETF)

A NOVEL hybrid technique that does NOT exist as a standard method.

Pipeline:
  1. Min-max normalize each country's daily-case curve
  2. Multi-scale Wavelet Decomposition (DWT, Daubechies-4, 3 levels)
  3. Pairwise Dynamic Time Warping (DTW) distance matrix
  4. Spectral Embedding (Laplacian Eigenmaps) to 2D/3D
  5. HDBSCAN clustering on the embedded space
  6. Fingerprint visualization and cluster labelling
"""

import numpy as np
import pandas as pd
import pywt
from dtaidistance import dtw
from sklearn.manifold import SpectralEmbedding
from sklearn.preprocessing import MinMaxScaler
import hdbscan


def _normalize_curve(series: np.ndarray) -> np.ndarray:
    """Min-max scale a 1-D array to [0, 1]."""
    s = series.astype(float)
    mn, mx = s.min(), s.max()
    if mx - mn == 0:
        return np.zeros_like(s)
    return (s - mn) / (mx - mn)


def _wavelet_features(curve: np.ndarray, wavelet: str = "db4",
                       level: int = 3) -> np.ndarray:
    """Apply DWT and concatenate approximation + detail coefficients."""
    # Pad to next power of 2 if needed
    n = len(curve)
    target = 2 ** int(np.ceil(np.log2(max(n, 2 ** level))))
    padded = np.pad(curve, (0, target - n), mode="constant")

    coeffs = pywt.wavedec(padded, wavelet, level=level)
    # Concatenate all coefficient arrays
    return np.concatenate(coeffs)


def prepare_curves(df: pd.DataFrame, countries: list[str],
                   metric: str = "daily_confirmed") -> dict[str, np.ndarray]:
    """Extract and normalize daily-case curves for each country.

    Returns dict {country_name: normalized_array}.
    """
    curves = {}
    for c in countries:
        series = df.loc[df["country"] == c, metric].values
        curves[c] = _normalize_curve(series)
    return curves


def wavelet_transform(curves: dict[str, np.ndarray],
                      wavelet: str = "db4",
                      level: int = 3) -> dict[str, np.ndarray]:
    """Apply multi-scale wavelet decomposition to each curve."""
    return {c: _wavelet_features(v, wavelet, level) for c, v in curves.items()}


def compute_dtw_matrix(wavelet_curves: dict[str, np.ndarray]) -> tuple[np.ndarray, list[str]]:
    """Compute pairwise DTW distance matrix on wavelet-transformed curves.

    Returns (distance_matrix, country_order).
    """
    names = list(wavelet_curves.keys())
    n = len(names)
    D = np.zeros((n, n))

    for i in range(n):
        for j in range(i + 1, n):
            d = dtw.distance(wavelet_curves[names[i]],
                             wavelet_curves[names[j]])
            D[i, j] = d
            D[j, i] = d

    return D, names


def dtw_to_affinity(D: np.ndarray, sigma: float | None = None) -> np.ndarray:
    """Convert distance matrix to affinity matrix using Gaussian kernel.

    A_ij = exp(-D_ij^2 / (2 * sigma^2))
    sigma defaults to the median of non-zero distances.
    """
    if sigma is None:
        nonzero = D[D > 0]
        sigma = float(np.median(nonzero)) if len(nonzero) > 0 else 1.0

    return np.exp(-D ** 2 / (2 * sigma ** 2))


def spectral_embed(affinity: np.ndarray, n_components: int = 2) -> np.ndarray:
    """Spectral Embedding from a precomputed affinity matrix."""
    se = SpectralEmbedding(n_components=n_components, affinity="precomputed",
                           random_state=42)
    return se.fit_transform(affinity)


def cluster_hdbscan(embedding: np.ndarray,
                    min_cluster_size: int = 3) -> np.ndarray:
    """HDBSCAN clustering on the spectral embedding."""
    clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size)
    labels = clusterer.fit_predict(embedding)
    return labels


# ── Full Pipeline ────────────────────────────────────────────────────────────

def run_etf(df: pd.DataFrame, countries: list[str],
            metric: str = "daily_confirmed",
            wavelet: str = "db4", level: int = 3,
            n_components: int = 2,
            min_cluster_size: int = 3) -> dict:
    """Run the full ETF pipeline.

    Returns
    -------
    dict with keys:
        countries     : list of country names (in order)
        curves        : dict of normalized curves
        dtw_matrix    : N×N distance matrix
        affinity      : N×N affinity matrix
        embedding     : N×n_components array
        labels        : cluster labels (N,)
        results_df    : DataFrame [country, cluster, emb_x, emb_y]
    """
    # Step 1 & 2
    curves = prepare_curves(df, countries, metric)
    wt_curves = wavelet_transform(curves, wavelet, level)

    # Step 3
    D, names = compute_dtw_matrix(wt_curves)

    # Step 4
    A = dtw_to_affinity(D)
    emb = spectral_embed(A, n_components)

    # Step 5
    labels = cluster_hdbscan(emb, min_cluster_size)

    # Assemble results
    results_df = pd.DataFrame({
        "country": names,
        "cluster": labels,
    })
    for i in range(n_components):
        results_df[f"emb_{i}"] = emb[:, i]

    return {
        "countries": names,
        "curves": curves,
        "dtw_matrix": D,
        "affinity": A,
        "embedding": emb,
        "labels": labels,
        "results_df": results_df,
    }
