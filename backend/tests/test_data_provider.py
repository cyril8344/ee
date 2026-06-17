"""Tests for the unified data provider (synthetic / fallback path)."""
import pandas as pd

import data_provider


def test_synthetic_always_returns_data():
    df = data_provider._fetch_synthetic("2024-01-01", "2024-02-01", 500)
    assert len(df) > 0
    assert list(df.columns) == ["open", "high", "low", "close", "volume"]
    assert df.index.tz is not None


def test_synthetic_is_deterministic():
    a = data_provider._fetch_synthetic("2024-03-01", "2024-03-15", 500)
    b = data_provider._fetch_synthetic("2024-03-01", "2024-03-15", 500)
    pd.testing.assert_frame_equal(a, b)


def test_get_m5_returns_dataframe_and_provider():
    df, provider = data_provider.get_m5(start="2024-01-01", end="2024-01-20", bars=500)
    assert len(df) > 0
    assert provider in ("twelvedata", "polygon", "alphavantage", "yfinance", "synthetic")


def test_available_providers_includes_keyless():
    avail = data_provider.available_providers()
    assert "yfinance" in avail
    assert "synthetic" in avail


def test_ohlc_integrity():
    df = data_provider._fetch_synthetic("2024-01-01", "2024-01-10", 500)
    assert (df["high"] >= df["low"]).all()
    assert (df["high"] >= df["close"]).all()
    assert (df["low"] <= df["close"]).all()
