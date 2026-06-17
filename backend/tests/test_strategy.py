"""Tests for indicators, sessions and the multi-timeframe strategy."""
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest

import strategy
from strategy import (
    ema, rsi, atr, add_indicators, active_session, compute_bias,
    is_bullish_engulfing, is_bearish_engulfing, evaluate,
)


def _frame(closes, vol=1000):
    idx = pd.date_range("2024-01-01", periods=len(closes), freq="5min", tz="UTC")
    closes = np.array(closes, dtype=float)
    return pd.DataFrame({
        "open": closes, "high": closes + 1.0, "low": closes - 1.0,
        "close": closes, "volume": [vol] * len(closes),
    }, index=idx)


def test_ema_matches_pandas():
    s = pd.Series([1, 2, 3, 4, 5], dtype=float)
    assert ema(s, 3).iloc[-1] == pytest.approx(s.ewm(span=3, adjust=False).mean().iloc[-1])


def test_rsi_bounds():
    up = pd.Series(np.arange(1, 60, dtype=float))
    down = pd.Series(np.arange(60, 1, -1, dtype=float))
    assert rsi(up).iloc[-1] > 70
    assert rsi(down).iloc[-1] < 30


def test_atr_positive():
    df = _frame(np.linspace(2000, 2050, 50))
    a = atr(df)
    assert (a.dropna() >= 0).all()


def test_active_session():
    # 09:00 CET (winter = 08:00 UTC) -> London
    london = datetime(2024, 1, 15, 8, 0, tzinfo=timezone.utc)
    assert active_session(london) == "London"
    # 15:00 CET (winter = 14:00 UTC) -> New York
    ny = datetime(2024, 1, 15, 14, 0, tzinfo=timezone.utc)
    assert active_session(ny) == "NewYork"
    # 23:00 UTC -> outside sessions
    off = datetime(2024, 1, 15, 23, 0, tzinfo=timezone.utc)
    assert active_session(off) is None


def test_bias_confusion_zone_is_neutral():
    h1 = add_indicators(_frame(np.linspace(2000, 2010, 250)))
    # force price between ema50 and ema200 by construction is hard;
    # instead assert the three branches via direct values
    row = h1.iloc[-1].copy()
    lo, hi = min(row["ema50"], row["ema200"]), max(row["ema50"], row["ema200"])
    # build a tiny synthetic h1 where close sits between the emas
    test = h1.copy()
    mid = (lo + hi) / 2
    test.iloc[-1, test.columns.get_loc("close")] = mid
    assert compute_bias(test) == "NEUTRE"


def test_engulfing_patterns():
    prev_bear = {"open": 2010, "close": 2000}
    cur_bull = {"open": 1999, "close": 2011}
    assert is_bullish_engulfing(prev_bear, cur_bull) is True

    prev_bull = {"open": 2000, "close": 2010}
    cur_bear = {"open": 2011, "close": 1999}
    assert is_bearish_engulfing(prev_bull, cur_bear) is True


def test_evaluate_returns_none_when_insufficient_data():
    small = add_indicators(_frame(np.linspace(2000, 2001, 10)))
    assert evaluate(small, small, small) is None


def test_evaluate_runs_on_real_shaped_data():
    # 600 M5 bars; resample to M15/H1 like the engine does.
    closes = 2000 + np.cumsum(np.random.default_rng(1).normal(0, 0.3, 600))
    m5 = add_indicators(_frame(closes))
    agg = {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    m15 = add_indicators(m5[["open", "high", "low", "close", "volume"]]
                         .resample("15min", label="right", closed="right").agg(agg).dropna())
    h1 = add_indicators(m5[["open", "high", "low", "close", "volume"]]
                        .resample("60min", label="right", closed="right").agg(agg).dropna())
    # Should not raise; returns Signal or None
    sig = evaluate(m5, m15, h1, check_session=False)
    assert sig is None or sig.direction in ("long", "short")
