"""Tests for data_ingest module."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd
import numpy as np
import pytest


def _make_wide_csv(tmp_path, metric="confirmed"):
    """Create a tiny wide-format CSV mimicking JHU structure."""
    data = {
        "Province/State": ["", "", ""],
        "Country/Region": ["US", "India", "Brazil"],
        "Lat": [37.0, 20.0, -14.0],
        "Long": [-95.0, 77.0, -51.0],
        "1/22/20": [1, 0, 0],
        "1/23/20": [2, 1, 0],
        "1/24/20": [5, 3, 1],
    }
    df = pd.DataFrame(data)
    path = tmp_path / f"time_series_covid19_{metric}_global.csv"
    df.to_csv(path, index=False)
    return tmp_path


class TestLoadSingle:
    def test_shape_and_columns(self, tmp_path):
        data_dir = _make_wide_csv(tmp_path, "confirmed")
        from src.data_ingest import load_single
        df = load_single("confirmed", data_dir=str(data_dir))
        assert "country" in df.columns
        assert "date" in df.columns
        assert "confirmed" in df.columns
        assert len(df) == 9  # 3 countries × 3 dates

    def test_no_nans(self, tmp_path):
        data_dir = _make_wide_csv(tmp_path, "confirmed")
        from src.data_ingest import load_single
        df = load_single("confirmed", data_dir=str(data_dir))
        assert df["confirmed"].isna().sum() == 0


class TestLoadMerged:
    def test_daily_columns(self, tmp_path):
        for m in ["confirmed", "deaths", "recovered"]:
            _make_wide_csv(tmp_path, m)
        from src.data_ingest import load_merged
        df = load_merged(data_dir=str(tmp_path))
        assert "daily_confirmed" in df.columns
        assert "daily_deaths" in df.columns

    def test_daily_non_negative(self, tmp_path):
        for m in ["confirmed", "deaths", "recovered"]:
            _make_wide_csv(tmp_path, m)
        from src.data_ingest import load_merged
        df = load_merged(data_dir=str(tmp_path))
        assert (df["daily_confirmed"] >= 0).all()
        assert (df["daily_deaths"] >= 0).all()


class TestHelpers:
    def test_get_top_countries(self, tmp_path):
        for m in ["confirmed", "deaths", "recovered"]:
            _make_wide_csv(tmp_path, m)
        from src.data_ingest import load_merged, get_top_countries
        df = load_merged(data_dir=str(tmp_path))
        top = get_top_countries(df, n=2)
        assert len(top) == 2
        assert top[0] == "US"  # US has highest max confirmed
