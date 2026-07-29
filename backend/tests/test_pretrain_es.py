"""Tests for pretrain_es.py::_load_es_data — SPY-via-Twelve-Data proxy for ES
(replaces the old yfinance ES=F fetch, see pretrain_es.py docstring)."""
from unittest.mock import patch

import pandas as pd
import pytest

import pretrain_es


def _fake_spy(n=150, freq="5min"):
    idx = pd.date_range("2026-01-01", periods=n, freq=freq, tz="UTC")
    df = pd.DataFrame({
        "open": [580.0] * n, "high": [582.0] * n,
        "low": [579.0] * n, "close": [581.5] * n,
        "volume": [100000] * n,
    }, index=idx)
    df.index.name = "time"
    return df


def test_load_es_data_scales_spy_to_es_price_level():
    with patch("pretrain_es._dp._fetch_twelvedata_range", return_value=_fake_spy()):
        df, source = pretrain_es._load_es_data("2026-01-01", "2026-01-02")
    assert source == "twelvedata_spy"
    row = df.iloc[0]
    assert row["open"] == pytest.approx(5800.0)
    assert row["high"] == pytest.approx(5820.0)
    assert row["low"] == pytest.approx(5790.0)
    assert row["close"] == pytest.approx(5815.0)
    # volume is not a price field — must stay unscaled
    assert row["volume"] == pytest.approx(100000)


def test_load_es_data_calls_twelvedata_with_spy_symbol():
    with patch("pretrain_es._dp._fetch_twelvedata_range", return_value=_fake_spy()) as m:
        pretrain_es._load_es_data("2026-01-01", "2026-01-02")
    args, kwargs = m.call_args
    assert kwargs.get("symbol") == "SPY"


def test_load_es_data_falls_back_to_synthetic_when_twelvedata_fails():
    with patch("pretrain_es._dp._fetch_twelvedata_range", side_effect=RuntimeError("boom")):
        df, source = pretrain_es._load_es_data("2026-01-01", "2026-02-01")
    assert source == "synthetic"
    assert len(df) > 0


def test_load_es_data_falls_back_to_synthetic_on_insufficient_bars():
    with patch("pretrain_es._dp._fetch_twelvedata_range", return_value=_fake_spy(n=5)):
        df, source = pretrain_es._load_es_data("2026-01-01", "2026-02-01")
    assert source == "synthetic"
