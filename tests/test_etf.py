"""Tests for the ETF (Epidemic Trajectory Fingerprinting) module."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
import pytest


def _make_dummy_df():
    """Create a minimal DataFrame for ETF testing."""
    np.random.seed(42)
    dates = pd.date_range("2020-01-01", periods=100, freq="D")
    records = []
    for c in ["A", "B", "C", "D", "E"]:
        vals = np.abs(np.random.randn(100).cumsum())
        for d, v in zip(dates, vals):
            records.append({"country": c, "date": d, "daily_confirmed": v})
    return pd.DataFrame(records)


class TestETFPipeline:
    def test_prepare_curves_length(self):
        from src.etf import prepare_curves
        df = _make_dummy_df()
        curves = prepare_curves(df, ["A", "B", "C"])
        assert len(curves) == 3
        assert len(curves["A"]) == 100

    def test_normalize_range(self):
        from src.etf import _normalize_curve
        arr = np.array([10, 20, 30, 40, 50], dtype=float)
        norm = _normalize_curve(arr)
        assert norm.min() == pytest.approx(0.0)
        assert norm.max() == pytest.approx(1.0)

    def test_dtw_matrix_symmetric(self):
        from src.etf import prepare_curves, wavelet_transform, compute_dtw_matrix
        df = _make_dummy_df()
        curves = prepare_curves(df, ["A", "B", "C"])
        wt = wavelet_transform(curves)
        D, names = compute_dtw_matrix(wt)
        assert D.shape == (3, 3)
        np.testing.assert_array_almost_equal(D, D.T)
        np.testing.assert_array_equal(np.diag(D), 0)

    def test_full_pipeline_returns_clusters(self):
        from src.etf import run_etf
        df = _make_dummy_df()
        result = run_etf(df, ["A", "B", "C", "D", "E"],
                         min_cluster_size=2)
        assert "labels" in result
        assert len(result["labels"]) == 5
        assert "results_df" in result
        assert len(result["results_df"]) == 5
