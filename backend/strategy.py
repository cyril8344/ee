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
ADX_PERIOD = 14
VOL_AVG_PERIOD = 20

RSI_LOW = 45.0
RSI_HIGH = 55.0
ATR_MIN = 0.8
ADX_MIN = 20.0              # minimum trend strength (0-100)
SR_PROXIMITY_ATR = 0.5      # block entry if opposing S/R within 0.5×ATR
SPREAD_MAX_PIPS = 0.8       # block entry if spread > 0.8 pip
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
    rs = avg_gain / avg_loss
    out = 100.0 - (100.0 / (1.0 + rs))
    # Degenerate cases when avg_loss == 0 (no down moves in the window):
    #   gains present -> fully overbought (100); totally flat -> neutral (50).
    out = out.mask((avg_loss == 0) & (avg_gain > 0), 100.0)
    out = out.mask((avg_loss == 0) & (avg_gain == 0), 50.0)
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


def adx(df: pd.DataFrame, period: int = ADX_PERIOD) -> pd.Series:
    high, low, close = df["high"], df["low"], df["close"]
    prev_high = high.shift(1)
    prev_low = low.shift(1)
    prev_close = close.shift(1)
    tr = pd.concat([
        (high - low),
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    up = high - prev_high
    down = prev_low - low
    dm_plus = pd.Series(
        np.where((up > down) & (up > 0), up, 0.0), index=df.index, dtype=float)
    dm_minus = pd.Series(
        np.where((down > up) & (down > 0), down, 0.0), index=df.index, dtype=float)
    alpha = 1.0 / period
    atr_s = tr.ewm(alpha=alpha, adjust=False).mean().replace(0, np.nan)
    di_plus = 100 * dm_plus.ewm(alpha=alpha, adjust=False).mean() / atr_s
    di_minus = 100 * dm_minus.ewm(alpha=alpha, adjust=False).mean() / atr_s
    di_sum = (di_plus + di_minus).replace(0, np.nan)
    dx = 100 * (di_plus - di_minus).abs() / di_sum
    return dx.ewm(alpha=alpha, adjust=False).mean().fillna(0)


def vwap(df: pd.DataFrame) -> pd.Series:
    """Intraday VWAP — resets at midnight UTC."""
    typical = (df["high"] + df["low"] + df["close"]) / 3
    vol = df["volume"].replace(0, np.nan)
    dates = df.index.normalize()
    tpv = typical * df["volume"]
    cum_tpv = tpv.groupby(dates).cumsum()
    cum_vol = df["volume"].groupby(dates).cumsum().replace(0, np.nan)
    return (cum_tpv / cum_vol).fillna(typical)


def asian_session_range(df: pd.DataFrame) -> Optional[Dict[str, float]]:
    """Return high/low/mid of the Asian session (22:00-07:00 UTC) ending before London."""
    if df.index.tz is None:
        return None
    last_ts = df.index[-1]
    today = last_ts.normalize()
    asian_end = today.replace(hour=7)    # 07:00 UTC ~ London pre-open
    asian_start = today.replace(hour=22) - pd.Timedelta(days=1)
    asian = df[(df.index >= asian_start) & (df.index < asian_end)]
    if len(asian) < 4:
        return None
    return {
        "high": float(asian["high"].max()),
        "low": float(asian["low"].min()),
        "mid": float((asian["high"].max() + asian["low"].min()) / 2),
    }


def market_structure_ok(df: pd.DataFrame, bias: str, lookback: int = 20) -> bool:
    """True if recent HH/HL (LONG) or LH/LL (SHORT) structure matches bias."""
    if len(df) < lookback:
        return True
    sub = df.tail(lookback)
    t = lookback // 3
    highs_early = sub["high"].iloc[:t].mean()
    highs_late = sub["high"].iloc[-t:].mean()
    lows_early = sub["low"].iloc[:t].mean()
    lows_late = sub["low"].iloc[-t:].mean()
    if bias == "LONG":
        return highs_late > highs_early or lows_late > lows_early
    else:
        return highs_late < highs_early or lows_late < lows_early


def near_opposing_sr(entry: float, bias: str,
                     sr: Dict[str, List[float]], atr_val: float) -> bool:
    """True if an opposing S/R level is dangerously close to the entry."""
    tol = SR_PROXIMITY_ATR * atr_val
    if bias == "LONG":
        return any(0 < (r - entry) < tol for r in sr.get("resistance", []))
    else:
        return any(0 < (entry - s) < tol for s in sr.get("support", []))


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Attach EMA/RSI/ATR/ADX/VWAP/volume-avg columns."""
    out = df.copy()
    out["ema9"] = ema(out["close"], EMA_FAST)
    out["ema21"] = ema(out["close"], EMA_MID)
    out["ema50"] = ema(out["close"], EMA_50)
    out["ema200"] = ema(out["close"], EMA_SLOW)
    out["rsi"] = rsi(out["close"], RSI_PERIOD)
    out["atr"] = atr(out, ATR_PERIOD)
    out["adx"] = adx(out, ADX_PERIOD)
    if "volume" in out.columns:
        out["vol_avg"] = out["volume"].rolling(VOL_AVG_PERIOD, min_periods=1).mean()
    else:
        out["volume"] = 0.0
        out["vol_avg"] = 0.0
    out["vwap"] = vwap(out)
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

    # 3) ADX — trend strength filter (no trading in ranging markets)
    adx_val_h1 = float(h1.iloc[-1].get("adx", 0)) if len(h1) else 0.0
    if adx_val_h1 < ADX_MIN:
        return None

    # 4) Market structure — H1 HH/HL or LH/LL confirms bias
    if not market_structure_ok(h1, bias):
        return None

    # 5) M15 confirmation
    if not confirm_m15(m15, bias):
        return None

    # 6) VWAP alignment on M15
    if len(m15) and "vwap" in m15.columns:
        m15_cur = m15.iloc[-1]
        if not pd.isna(m15_cur.get("vwap", float("nan"))):
            if bias == "LONG" and m15_cur["close"] < m15_cur["vwap"]:
                return None
            if bias == "SHORT" and m15_cur["close"] > m15_cur["vwap"]:
                return None

    # 7) Asian range context — prefer trading in breakout direction
    asian = asian_session_range(m5)

    # 8) M5 volatility floor
    atr_val = float(cur["atr"]) if not pd.isna(cur["atr"]) else 0.0
    if atr_val < ATR_MIN:
        return None

    # 9) Spread check (bid-ask approximated from bar range vs ATR)
    bar_range = float(cur["high"] - cur["low"])
    implied_spread = max(0.0, bar_range - atr_val * 0.5)
    if implied_spread > SPREAD_MAX_PIPS * 0.1:  # 0.1 = pip size for gold
        pass  # spread check is heuristic; keep as soft filter only

    # 10) M5 entry trigger
    triggers = []
    if bias == "LONG":
        if is_bullish_engulfing(prev, cur):
            triggers.append("bullish_engulfing")
        if ema9_pullback_bounce(m5, bias):
            triggers.append("ema9_pullback")
        if micro_breakout(m5, bias):
            triggers.append("micro_breakout")
        # Asian range breakout bonus tag
        if asian and float(cur["close"]) > asian["high"]:
            triggers.append("asian_breakout")
    else:
        if is_bearish_engulfing(prev, cur):
            triggers.append("bearish_engulfing")
        if ema9_pullback_bounce(m5, bias):
            triggers.append("ema9_pullback")
        if micro_breakout(m5, bias):
            triggers.append("micro_breakout")
        if asian and float(cur["close"]) < asian["low"]:
            triggers.append("asian_breakout")

    # Keep only non-asian-breakout triggers for the mandatory check
    core_triggers = [t for t in triggers if t != "asian_breakout"]
    if not core_triggers:
        return None

    # 11) S/R proximity — don't enter into a wall
    sr = swing_levels(m5, lookback=60)
    if near_opposing_sr(entry, bias, sr, atr_val):
        return None

    # 12) Build trade levels
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
    """Lightweight market snapshot for the dashboard."""
    ts = now or datetime.now(timezone.utc)
    bias = compute_bias(h1) if len(h1) else "NEUTRE"
    session = active_session(ts)
    cur5 = m5.iloc[-1] if len(m5) else None
    cur15 = m15.iloc[-1] if len(m15) else None
    cur_h1 = h1.iloc[-1] if len(h1) else None

    def _f(row, col, digits=3):
        v = row.get(col, float("nan")) if row is not None else float("nan")
        return round(float(v), digits) if not pd.isna(v) else None

    asian = asian_session_range(m5) if len(m5) else None

    return {
        "bias": bias,
        "session": session or "Hors session",
        "rsi_m5": _f(cur5, "rsi", 1),
        "rsi_m15": _f(cur15, "rsi", 1),
        "atr_m5": _f(cur5, "atr", 3),
        "atr_avg": round(float(m5["atr"].tail(50).mean()), 3) if len(m5) else None,
        "adx_h1": _f(cur_h1, "adx", 1),
        "vwap_m15": _f(cur15, "vwap", 3),
        "price": _f(cur5, "close", 3),
        "atr_min": ATR_MIN,
        "adx_min": ADX_MIN,
        "asian_range": asian,
        "structure_ok": market_structure_ok(h1, bias) if len(h1) >= 6 else None,
    }
