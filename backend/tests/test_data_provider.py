"""Tests for the unified data provider (synthetic / fallback path)."""
from unittest.mock import patch, MagicMock

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


def test_fetch_twelvedata_range_rotates_key_on_429(monkeypatch):
    """A 429 on the first backtest key must roll over to the second one
    (TWELVEDATA_API_KEY_BACKTEST_2) instead of just re-hitting the same key."""
    monkeypatch.setenv("TWELVEDATA_API_KEY_BACKTEST", "key_a")
    monkeypatch.setenv("TWELVEDATA_API_KEY_BACKTEST_2", "key_b")
    data_provider._td_key_index["backtest"][0] = 0

    idx = pd.date_range("2024-01-01 00:00", periods=3, freq="5min", tz="UTC")
    values = [
        {"datetime": t.strftime("%Y-%m-%d %H:%M:%S"), "open": "100", "high": "101",
         "low": "99", "close": "100", "volume": "10"}
        for t in idx
    ]

    resp_429 = MagicMock(status_code=429)
    resp_ok = MagicMock(status_code=200)
    resp_ok.json.return_value = {"status": "ok", "values": values}
    resp_ok.raise_for_status.return_value = None

    calls = []

    def fake_get(url, params=None, timeout=None):
        calls.append(params["apikey"])
        return resp_429 if params["apikey"] == "key_a" else resp_ok

    with patch("data_provider.requests.get", side_effect=fake_get), \
         patch("data_provider._td_throttle", return_value=None):
        df = data_provider._fetch_twelvedata_range("2024-01-01", "2024-01-03", "XAUUSD")

    assert len(df) > 0
    assert calls[0] == "key_a"   # tente d'abord la clé initiale
    assert "key_b" in calls      # tourne vers la 2e clé au lieu de rester bloqué sur key_a
