"""
strategy.py
===========
Multi-timeframe XAU/USD scalping strategy.

Pipeline
--------
H1  : daily bias       -> price vs EMA200 (and EMA50 confusion zone)
M15 : confirmation     -> EMA9/EMA21 cross in bias direction,
                          RSI(14) in 45-55, volume > 20-period average
M5  : entry trigger    -> engulfing candle / EMA9 pullback bounce /
                          micro-consolidation breakout, ATR(14) > 0.8

Trade management parameters are produced with the signal:
    - stop loss at last M5 swing (capped at 1.2x ATR)
    - TP1 = 1R (close 60%), TP2 = 2R (close 40%)
    - max duration 45 minutes

Sessions: London (08-12 CET) and New York (14-18 CET) only.

The module is pure / stateless: feed it indicator-ready DataFrames and it
returns a `Signal` (or None).  Both the live engine and the backtester use
the exact same functions, guaranteeing parity.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, time, timezone, timedelta
from typing import Optional, Dict, Any, List

import numpy as np
import pandas as pd
import pytz

# --------------------------------------------------------------------------- #
# Tunables
# --------------------------------------------------------------------------- #
EMA_FAST = 9
EMA_MID = 21
EMA_SLOW = 200
EMA_50 = 50
RSI_PERIOD = 14
ATR_PERIOD = 14
VOL_AVG_PERIOD = 20

RSI_LOW = 45.0
RSI_HIGH = 55.0
ATR_MIN = 0.8
SL_ATR_MULT = 1.2
SWING_LOOKBACK = 5          # bars each side for swing detection
MICRO_RANGE_BARS = 3        # micro-consolidation length
MAX_TRADE_MINUTES = 45

CET = pytz.timezone("Europe/Paris")  # CET/CEST

# Sessions in CET local time
LONDON = (time(8, 0), time(12, 0))
NEWYORK = (time(14, 0), time(18, 0))


# --------------------------------------------------------------------------- #
# Indicators
# --------------------------------------------------------------------------- #
def ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def rsi(series: pd.Series, period: int = RSI_PERIOD) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1.0 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    out = 100.0 - (100.0 / (1.0 + rs))
    return out.fillna(50.0)


def atr(df: pd.DataFrame, period: int = ATR_PERIOD) -> pd.Series:
    high, low, close = df["high"], df["low"], df["close"]
    prev_close = close.shift(1)
    tr = pd.concat([
        (high - low),
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1.0 / period, adjust=False).mean()


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Attach EMA/RSI/ATR/volume-avg columns.  Expects OHLCV with a tz-aware
    DatetimeIndex (UTC) and columns: open, high, low, close, volume."""
    out = df.copy()
    out["ema9"] = ema(out["close"], EMA_FAST)
    out["ema21"] = ema(out["close"], EMA_MID)
    out["ema50"] = ema(out["close"], EMA_50)
    out["ema200"] = ema(out["close"], EMA_SLOW)
    out["rsi"] = rsi(out["close"], RSI_PERIOD)
    out["atr"] = atr(out, ATR_PERIOD)
    if "volume" in out.columns:
        out["vol_avg"] = out["volume"].rolling(VOL_AVG_PERIOD, min_periods=1).mean()
    else:
        out["volume"] = 0.0
        out["vol_avg"] = 0.0
    return out


# --------------------------------------------------------------------------- #
# Sessions
# --------------------------------------------------------------------------- #
def _in_window(t: time, window) -> bool:
    return window[0] <= t < window[1]


def active_session(ts_utc: datetime) -> Optional[str]:
    """Return 'London' / 'NewYork' / None for a UTC timestamp."""
    if ts_utc.tzinfo is None:
        ts_utc = ts_utc.replace(tzinfo=timezone.utc)
    local = ts_utc.astimezone(CET).time()
    if _in_window(local, LONDON):
        return "London"
    if _in_window(local, NEWYORK):
        return "NewYork"
    return None


# --------------------------------------------------------------------------- #
# Swings / structure
# --------------------------------------------------------------------------- #
def swing_levels(df: pd.DataFrame, lookback: int = 50) -> Dict[str, List[float]]:
    """Auto-detect support/resistance from swing highs/lows of last `lookback`."""
    sub = df.tail(lookback)
    highs, lows = [], []
    h = sub["high"].values
    l = sub["low"].values
    n = len(sub)
    k = SWING_LOOKBACK
    for i in range(k, n - k):
        if h[i] == max(h[i - k:i + k + 1]):
            highs.append(float(h[i]))
        if l[i] == min(l[i - k:i + k + 1]):
            lows.append(float(l[i]))
    # de-duplicate close levels
    def _dedupe(levels: List[float], tol: float) -> List[float]:
        levels = sorted(levels)
        result: List[float] = []
        for lv in levels:
            if not result or abs(lv - result[-1]) > tol:
                result.append(lv)
        return result

    tol = float(sub["close"].iloc[-1]) * 0.0008 if n else 0.0
    return {
        "resistance": _dedupe(highs, tol),
        "support": _dedupe(lows, tol),
    }


def last_swing_low(df: pd.DataFrame, lookback: int = 10) -> float:
    return float(df["low"].tail(lookback).min())


def last_swing_high(df: pd.DataFrame, lookback: int = 10) -> float:
    return float(df["high"].tail(lookback).max())


# --------------------------------------------------------------------------- #
# Bias / confirmation / entry primitives
# --------------------------------------------------------------------------- #
def compute_bias(h1: pd.DataFrame) -> str:
    """LONG / SHORT / NEUTRE from H1 EMA structure."""
    if len(h1) < 1:
        return "NEUTRE"
    row = h1.iloc[-1]
    price = row["close"]
    ema200 = row["ema200"]
    ema50 = row["ema50"]
    if any(pd.isna(x) for x in (price, ema200, ema50)):
        return "NEUTRE"
    lo, hi = min(ema50, ema200), max(ema50, ema200)
    if lo <= price <= hi:
        return "NEUTRE"            # confusion zone between EMA50 & EMA200
    if price > ema200:
        return "LONG"
    if price < ema200:
        return "SHORT"
    return "NEUTRE"


def confirm_m15(m15: pd.DataFrame, bias: str) -> bool:
    if bias not in ("LONG", "SHORT") or len(m15) < 2:
        return False
    cur, prev = m15.iloc[-1], m15.iloc[-2]
    if any(pd.isna(cur[c]) for c in ("ema9", "ema21", "rsi")):
        return False

    rsi_ok = RSI_LOW <= cur["rsi"] <= RSI_HIGH
    vol_ok = cur.get("volume", 0) > cur.get("vol_avg", 0) or cur.get("vol_avg", 0) == 0

    if bias == "LONG":
        crossed = prev["ema9"] <= prev["ema21"] and cur["ema9"] > cur["ema21"]
        aligned = cur["ema9"] > cur["ema21"]
        return (crossed or aligned) and rsi_ok and vol_ok
    else:  # SHORT
        crossed = prev["ema9"] >= prev["ema21"] and cur["ema9"] < cur["ema21"]
        aligned = cur["ema9"] < cur["ema21"]
        return (crossed or aligned) and rsi_ok and vol_ok


def is_bullish_engulfing(prev, cur) -> bool:
    return (
        prev["close"] < prev["open"] and          # prev bearish
        cur["close"] > cur["open"] and            # cur bullish
        cur["close"] >= prev["open"] and
        cur["open"] <= prev["close"]
    )


def is_bearish_engulfing(prev, cur) -> bool:
    return (
        prev["close"] > prev["open"] and
        cur["close"] < cur["open"] and
        cur["close"] <= prev["open"] and
        cur["open"] >= prev["close"]
    )


def ema9_pullback_bounce(m5: pd.DataFrame, bias: str) -> bool:
    """Pullback to EMA9 then rejection in bias direction."""
    if len(m5) < 3:
        return False
    cur, prev = m5.iloc[-1], m5.iloc[-2]
    if bias == "LONG":
        touched = prev["low"] <= prev["ema9"]
        bounce = cur["close"] > cur["ema9"] and cur["close"] > cur["open"]
        return touched and bounce
    else:
        touched = prev["high"] >= prev["ema9"]
        bounce = cur["close"] < cur["ema9"] and cur["close"] < cur["open"]
        return touched and bounce


def micro_breakout(m5: pd.DataFrame, bias: str) -> bool:
    """Breakout of a 3+ bar micro-consolidation."""
    if len(m5) < MICRO_RANGE_BARS + 1:
        return False
    window = m5.iloc[-(MICRO_RANGE_BARS + 1):-1]
    cur = m5.iloc[-1]
    rng_high = float(window["high"].max())
    rng_low = float(window["low"].min())
    # require a genuinely tight range relative to ATR
    atr_val = cur.get("atr", 0) or 0
    if atr_val <= 0:
        return False
    tight = (rng_high - rng_low) <= 1.5 * atr_val
    if not tight:
        return False
    if bias == "LONG":
        return cur["close"] > rng_high
    else:
        return cur["close"] < rng_low


# --------------------------------------------------------------------------- #
# Signal object
# --------------------------------------------------------------------------- #
@dataclass
class Signal:
    direction: str          # 'long' | 'short'
    bias: str
    session: str
    entry: float
    stop_loss: float
    take_profit1: float
    take_profit2: float
    atr: float
    reason: str
    risk_distance: float
    timestamp: datetime
    max_duration_min: int = MAX_TRADE_MINUTES
    meta: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        d = {
            "direction": self.direction,
            "bias": self.bias,
            "session": self.session,
            "entry": round(self.entry, 3),
            "stop_loss": round(self.stop_loss, 3),
            "take_profit1": round(self.take_profit1, 3),
            "take_profit2": round(self.take_profit2, 3),
            "atr": round(self.atr, 3),
            "reason": self.reason,
            "risk_distance": round(self.risk_distance, 3),
            "timestamp": self.timestamp.isoformat(),
            "max_duration_min": self.max_duration_min,
            "meta": self.meta,
        }
        return d


# --------------------------------------------------------------------------- #
# Master evaluation
# --------------------------------------------------------------------------- #
def evaluate(
    m5: pd.DataFrame,
    m15: pd.DataFrame,
    h1: pd.DataFrame,
    now: Optional[datetime] = None,
    check_session: bool = True,
) -> Optional[Signal]:
    """
    Evaluate the full multi-timeframe stack on the *last closed* M5 bar.
    Returns a Signal or None.

    DataFrames must already contain indicators (call add_indicators).
    """
    if len(m5) < max(EMA_SLOW, 30) or len(m15) < 3 or len(h1) < 1:
        return None

    cur = m5.iloc[-1]
    prev = m5.iloc[-2]
    ts = now or cur.name.to_pydatetime()
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)

    # 1) Session gate
    session = active_session(ts)
    if check_session and session is None:
        return None
    session = session or "London"

    # 2) H1 bias
    bias = compute_bias(h1)
    if bias == "NEUTRE":
        return None

    # 3) M15 confirmation
    if not confirm_m15(m15, bias):
        return None

    # 4) M5 volatility floor
    atr_val = float(cur["atr"]) if not pd.isna(cur["atr"]) else 0.0
    if atr_val < ATR_MIN:
        return None

    # 5) M5 entry trigger
    triggers = []
    if bias == "LONG":
        if is_bullish_engulfing(prev, cur):
            triggers.append("bullish_engulfing")
        if ema9_pullback_bounce(m5, bias):
            triggers.append("ema9_pullback")
        if micro_breakout(m5, bias):
            triggers.append("micro_breakout")
    else:
        if is_bearish_engulfing(prev, cur):
            triggers.append("bearish_engulfing")
        if ema9_pullback_bounce(m5, bias):
            triggers.append("ema9_pullback")
        if micro_breakout(m5, bias):
            triggers.append("micro_breakout")

    if not triggers:
        return None

    # 6) Build trade levels
    entry = float(cur["close"])
    if bias == "LONG":
        swing = last_swing_low(m5, lookback=10)
        raw_sl = min(swing, entry - 1e-6)
        sl = max(raw_sl, entry - SL_ATR_MULT * atr_val)  # cap at 1.2 ATR
        direction = "long"
    else:
        swing = last_swing_high(m5, lookback=10)
        raw_sl = max(swing, entry + 1e-6)
        sl = min(raw_sl, entry + SL_ATR_MULT * atr_val)
        direction = "short"

    risk = abs(entry - sl)
    if risk <= 0:
        return None

    if direction == "long":
        tp1 = entry + risk
        tp2 = entry + 2 * risk
    else:
        tp1 = entry - risk
        tp2 = entry - 2 * risk

    return Signal(
        direction=direction,
        bias=bias,
        session=session,
        entry=entry,
        stop_loss=sl,
        take_profit1=tp1,
        take_profit2=tp2,
        atr=atr_val,
        reason="+".join(triggers),
        risk_distance=risk,
        timestamp=ts,
        meta={
            "rsi_m5": round(float(cur["rsi"]), 1),
            "rsi_m15": round(float(m15.iloc[-1]["rsi"]), 1),
            "triggers": triggers,
        },
    )


def snapshot(m5: pd.DataFrame, m15: pd.DataFrame, h1: pd.DataFrame,
             now: Optional[datetime] = None) -> Dict[str, Any]:
    """Lightweight market snapshot for the dashboard (bias, session, RSI...)."""
    ts = now or datetime.now(timezone.utc)
    bias = compute_bias(h1) if len(h1) else "NEUTRE"
    session = active_session(ts)
    cur5 = m5.iloc[-1] if len(m5) else None
    cur15 = m15.iloc[-1] if len(m15) else None
    return {
        "bias": bias,
        "session": session or "Hors session",
        "rsi_m5": round(float(cur5["rsi"]), 1) if cur5 is not None and not pd.isna(cur5["rsi"]) else None,
        "rsi_m15": round(float(cur15["rsi"]), 1) if cur15 is not None and not pd.isna(cur15["rsi"]) else None,
        "atr_m5": round(float(cur5["atr"]), 3) if cur5 is not None and not pd.isna(cur5["atr"]) else None,
        "atr_avg": round(float(m5["atr"].tail(50).mean()), 3) if len(m5) else None,
        "price": round(float(cur5["close"]), 3) if cur5 is not None else None,
        "atr_min": ATR_MIN,
    }
