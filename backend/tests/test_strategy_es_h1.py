"""Tests for the H1-native ES strategy (strategy_es_h1.py)."""
import numpy as np
import pandas as pd
import pytz

import strategy_es_h1 as strat


def _trending_h1_df(n=400, start="2026-01-05 09:00", direction="up", tz="UTC"):
    """Bougies H1 en tendance nette, alignées sur des jours ouvrés à partir de
    9h UTC (avant l'ouverture RTH 14h30 UTC) pour couvrir des horaires variés."""
    idx = pd.date_range(start, periods=n, freq="60min", tz=tz)
    idx = idx[idx.weekday < 5]
    n = len(idx)
    drift = 0.0006 if direction == "up" else -0.0006
    rng = np.random.default_rng(7)
    rets = rng.normal(drift, 0.0004, n)
    close = 5800.0 * np.exp(np.cumsum(rets))
    spread = np.abs(rng.normal(0, 2.0, n)) + 1.0
    high = close + spread
    low = close - spread
    open_ = np.concatenate([[5800.0], close[:-1]])
    volume = np.full(n, 5000.0)
    df = pd.DataFrame(
        {"open": open_,
         "high": np.maximum(high, np.maximum(open_, close)),
         "low": np.minimum(low, np.minimum(open_, close)),
         "close": close,
         "volume": volume},
        index=idx,
    )
    return df


def _rth_ts(df, hour_et=11, min_idx=220):
    """Premier timestamp du df, après min_idx bougies de warmup, qui tombe
    pendant la session RTH à l'heure ET donnée."""
    et = pytz.timezone("America/New_York")
    for pos in range(min_idx, len(df)):
        ts = df.index[pos]
        if ts.astimezone(et).hour == hour_et:
            return ts
    return df.index[-1]


def test_insufficient_warmup_returns_none():
    df = _trending_h1_df(n=50)
    df = strat.add_indicators(df)
    sig = strat.evaluate(df, ts=df.index[-1].to_pydatetime())
    assert sig is None


def test_uptrend_generates_long_with_atr_based_sl_tp():
    df = _trending_h1_df(n=400, direction="up")
    df = strat.add_indicators(df)
    ts = _rth_ts(df, hour_et=11)
    i = df.index.get_loc(ts)
    sig = strat.evaluate(df.iloc[:i + 1], ts=ts.to_pydatetime())

    assert sig is not None
    assert sig["bias"] == "LONG"
    entry, sl, tp1, tp2 = sig["entry"], sig["stop_loss"], sig["take_profit1"], sig["take_profit2"]
    assert sl < entry < tp1 < tp2

    risk = entry - sl
    reward1 = tp1 - entry
    reward2 = tp2 - entry
    assert abs(reward1 / risk - strat.DEFAULTS["tp1_r"]) < 0.05
    assert abs(reward2 / risk - strat.DEFAULTS["tp2_r"]) < 0.05


def test_downtrend_generates_short():
    df = _trending_h1_df(n=400, direction="down")
    df = strat.add_indicators(df)
    ts = _rth_ts(df, hour_et=11)
    i = df.index.get_loc(ts)
    sig = strat.evaluate(df.iloc[:i + 1], ts=ts.to_pydatetime())

    assert sig is not None
    assert sig["bias"] == "SHORT"
    entry, sl, tp1, tp2 = sig["entry"], sig["stop_loss"], sig["take_profit1"], sig["take_profit2"]
    assert tp2 < tp1 < entry < sl


def test_outside_rth_session_rejected():
    df = _trending_h1_df(n=400, direction="up")
    df = strat.add_indicators(df)
    et = pytz.timezone("America/New_York")
    # cherche une bougie hors 9h30-16h ET (nuit / pré-marché)
    ts = None
    for cand in df.index[-40:]:
        hour_et = cand.astimezone(et).hour
        if hour_et < 9 or hour_et >= 16:
            ts = cand
            break
    assert ts is not None
    i = df.index.get_loc(ts)
    sig = strat.evaluate(df.iloc[:i + 1], ts=ts.to_pydatetime())
    assert sig is None


def test_flatten_window_blocks_late_entry():
    df = _trending_h1_df(n=400, direction="up")
    df = strat.add_indicators(df)
    ts = _rth_ts(df, hour_et=15)  # dans la dernière heure avant clôture (16h ET)
    i = df.index.get_loc(ts)
    sig = strat.evaluate(df.iloc[:i + 1], ts=ts.to_pydatetime(),
                          params={"flatten_before_close_h1": 1})
    assert sig is None


def test_bad_hour_blocks_entry():
    df = _trending_h1_df(n=400, direction="up")
    df = strat.add_indicators(df)
    ts = _rth_ts(df, hour_et=11)
    i = df.index.get_loc(ts)
    sig = strat.evaluate(df.iloc[:i + 1], ts=ts.to_pydatetime(), params={"bad_hours_et": [11]})
    assert sig is None
