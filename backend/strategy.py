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
    - TP1 = 0.5R (close 50%), TP2 = 1.0R (close 50%), SL breakeven après TP1
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

RSI_LOW = 38.0
RSI_HIGH = 62.0
ATR_MIN = 2.5
ATR_MAX = 7.0   # plafond ATR M5 — au-delà = whipsaw → SL direct (SL dir avg 7.44 vs TP2 avg 5.99)
ADX_MIN = 20.0
SR_PROXIMITY_ATR = 0.7
SPREAD_MAX_PIPS = 0.8       # block entry if spread > 0.8 pip
SL_ATR_MULT = 1.4
SWING_LOOKBACK = 5          # bars each side for swing detection

# SMC parameters (optimised by Agent IA)
OB_LOOKBACK       = 40      # bougies analysées pour détecter les Order Blocks
OB_PROXIMITY_ATR  = 0.4     # tolérance de proximité OB en multiples d'ATR
FVG_MIN_SIZE_ATR  = 0.3     # taille minimale d'un FVG pour être valide
MICRO_RANGE_BARS = 3        # micro-consolidation length
MAX_TRADE_MINUTES = 45
TREND_BIAS_DISTANCE   = 0.5  # multiples d'ATR H1 — bloque SHORT si prix > EMA200 + 0.5 ATR
EMA200_MIN_DIST_LONG  = 0.3  # LONG doit être à ≥ 0.3×ATR au-dessus de EMA200
EMA200_MIN_DIST_SHORT = 0.6  # SHORT doit être à ≥ 0.6×ATR en-dessous de EMA200 (XAUUSD uptrend)
BAD_HOURS_CET         = {10}     # 14h débloquée pour amorcer l'apprentissage — LiveAdaptiveAgent ajustera
ATR_REGIME_MIN_RATIO  = 0.65     # assoupli 0.75→0.65 pour amorcer l'apprentissage — LiveAdaptiveAgent ajustera
RSI_M5_LONG_MIN       = 45.0    # momentum M5 minimum pour LONG (ajustable par LiveAdaptiveAgent)
RSI_M5_SHORT_MAX      = 55.0    # momentum M5 maximum pour SHORT (ajustable par LiveAdaptiveAgent)
PATTERN_FLOOR = 0.67        # exclut les patterns avec WR historique < 67%
MIN_WEIGHT_SUM_LONG = 1.0   # confluence minimale côté LONG (SHORT reste à 1.5)

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


def is_bad_timing(ts_utc: datetime) -> bool:
    """Return True during high-uncertainty windows we want to avoid.

    - Monday before 10:00 CET  (uncertain market open)
    - Friday after 16:00 CET   (volatile weekly close)
    """
    if ts_utc.tzinfo is None:
        ts_utc = ts_utc.replace(tzinfo=timezone.utc)
    local = ts_utc.astimezone(CET)
    weekday = local.weekday()   # 0=Monday, 4=Friday
    t = local.time()
    if weekday == 0 and t < time(10, 0):
        return True
    if weekday == 4 and t >= time(16, 0):
        return True
    if local.hour in BAD_HOURS_CET:
        return True
    return False


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
# SMC — Smart Money Concepts
# --------------------------------------------------------------------------- #
def find_fair_value_gaps(df: pd.DataFrame, lookback: int = 40) -> List[Dict[str, Any]]:
    """Detect unfilled Fair Value Gaps (3-candle imbalance zones)."""
    sub = df.tail(lookback)
    fvgs: List[Dict[str, Any]] = []
    for i in range(2, len(sub)):
        c0, c2 = sub.iloc[i - 2], sub.iloc[i]
        # Bullish FVG: gap between candle-1 high and candle+1 low
        if c0["high"] < c2["low"]:
            gap_low, gap_high = float(c0["high"]), float(c2["low"])
            # Unfilled = price hasn't traded back into the gap
            if sub["low"].iloc[i:].min() > gap_low:
                fvgs.append({"type": "bullish", "low": gap_low, "high": gap_high})
        # Bearish FVG: gap between candle-1 low and candle+1 high
        if c0["low"] > c2["high"]:
            gap_low, gap_high = float(c2["high"]), float(c0["low"])
            if sub["high"].iloc[i:].max() < gap_high:
                fvgs.append({"type": "bearish", "low": gap_low, "high": gap_high})
    return fvgs


def find_order_blocks(df: pd.DataFrame, lookback: int = None) -> List[Dict[str, Any]]:
    """Last bearish candle before a bullish impulse (bullish OB) and vice-versa."""
    if lookback is None:
        lookback = OB_LOOKBACK
    sub = df.tail(lookback)
    atr_val = float(sub["atr"].iloc[-1]) if "atr" in sub.columns else 1.0
    obs: List[Dict[str, Any]] = []
    for i in range(1, len(sub) - 1):
        c, nxt = sub.iloc[i], sub.iloc[i + 1]
        impulse = abs(nxt["close"] - nxt["open"])
        if impulse < 0.5 * atr_val:
            continue
        if c["close"] < c["open"] and nxt["close"] > nxt["open"]:
            obs.append({"type": "bullish",
                        "low": float(c["close"]), "high": float(c["open"])})
        elif c["close"] > c["open"] and nxt["close"] < nxt["open"]:
            obs.append({"type": "bearish",
                        "low": float(c["open"]), "high": float(c["close"])})
    return obs


def find_fvgs(df: pd.DataFrame, lookback: int = None) -> List[Dict[str, Any]]:
    """
    Détecte les Fair Value Gaps (imbalances à 3 bougies).
    Filtre les FVG trop petits (< FVG_MIN_SIZE_ATR * ATR).

    Bullish FVG : lows[i] > highs[i-2]
    Bearish FVG : highs[i] < lows[i-2]
    """
    if lookback is None:
        lookback = OB_LOOKBACK
    sub = df.tail(lookback)
    if len(sub) < 3:
        return []
    atr_val = float(sub["atr"].iloc[-1]) if "atr" in sub.columns else 1.0
    min_size = FVG_MIN_SIZE_ATR * atr_val
    fvgs: List[Dict[str, Any]] = []
    for i in range(2, len(sub)):
        hi_prev2 = float(sub["high"].iloc[i - 2])
        lo_prev2 = float(sub["low"].iloc[i - 2])
        hi_cur   = float(sub["high"].iloc[i])
        lo_cur   = float(sub["low"].iloc[i])
        # Bullish FVG : gap entre le haut de i-2 et le bas de i
        if lo_cur > hi_prev2 and (lo_cur - hi_prev2) >= min_size:
            mid = (hi_prev2 + lo_cur) / 2
            fvgs.append({"type": "bullish", "low": hi_prev2, "high": lo_cur,
                         "midpoint": mid, "pct50": mid, "pct100": lo_cur})
        # Bearish FVG : gap entre le bas de i-2 et le haut de i
        elif hi_cur < lo_prev2 and (lo_prev2 - hi_cur) >= min_size:
            mid = (hi_cur + lo_prev2) / 2
            fvgs.append({"type": "bearish", "low": hi_cur, "high": lo_prev2,
                         "midpoint": mid, "pct50": mid, "pct100": hi_cur})
    return fvgs


def liquidity_swept(df: pd.DataFrame, bias: str, lookback: int = 20) -> bool:
    """True if sell-side (LONG) or buy-side (SHORT) liquidity was recently swept then recovered."""
    sub = df.tail(lookback)
    if len(sub) < 6:
        return False
    prior, recent = sub.iloc[:-5], sub.iloc[-5:]
    if bias == "LONG":
        swept = float(recent["low"].min()) < float(prior["low"].min())
        recovered = float(sub["close"].iloc[-1]) > float(prior["low"].min())
        return swept and recovered
    else:
        swept = float(recent["high"].max()) > float(prior["high"].max())
        recovered = float(sub["close"].iloc[-1]) < float(prior["high"].max())
        return swept and recovered


def near_orderblock(price: float, bias: str,
                    obs: List[Dict[str, Any]], atr_val: float) -> bool:
    """True if price is inside or within OB_PROXIMITY_ATR of a matching order block."""
    tol = OB_PROXIMITY_ATR * atr_val
    for ob in obs:
        match = (ob["type"] == "bullish" and bias == "LONG") or \
                (ob["type"] == "bearish" and bias == "SHORT")
        if match and ob["low"] - tol <= price <= ob["high"] + tol:
            return True
    return False


def near_fvg(price: float, bias: str, fvgs: List[Dict[str, Any]]) -> bool:
    """True if price is inside a matching Fair Value Gap."""
    for fvg in fvgs:
        match = (fvg["type"] == "bullish" and bias == "LONG") or \
                (fvg["type"] == "bearish" and bias == "SHORT")
        if match and fvg["low"] <= price <= fvg["high"]:
            return True
    return False


# --------------------------------------------------------------------------- #
# Bias / confirmation / entry primitives
# --------------------------------------------------------------------------- #
def compute_bias(h1: pd.DataFrame) -> str:
    """LONG / SHORT : EMA50 donne la direction, EMA200 bloque si contradictoire.
    On ne trade QUE dans le sens du flux majeur (price > EMA200 → LONG autorisé)."""
    if len(h1) < 1:
        return "NEUTRE"
    row = h1.iloc[-1]
    price  = row["close"]
    ema50  = row.get("ema50",  float("nan"))
    ema200 = row.get("ema200", float("nan"))

    if pd.isna(ema50):
        return "NEUTRE"

    bias = "LONG" if price > ema50 else "SHORT"

    # Filtre dur EMA200 : interdit de trader contre le flux majeur
    if not pd.isna(ema200):
        if bias == "LONG"  and price < ema200:
            return "NEUTRE"
        if bias == "SHORT" and price > ema200:
            return "NEUTRE"

    return bias


def confirm_m15(m15: pd.DataFrame, bias: str, ema_mult: float = 0.3) -> bool:
    if bias not in ("LONG", "SHORT") or len(m15) < 1:
        return False
    cur = m15.iloc[-1]
    if any(pd.isna(cur[c]) for c in ("ema9", "ema21", "rsi")):
        return False
    rsi_ok = RSI_LOW <= float(cur["rsi"]) <= RSI_HIGH
    atr_m15 = float(cur.get("atr", 0) or 0)
    price = float(cur.get("close", 1) or 1)
    pip_floor = price * 0.0001
    tol = max(ema_mult * atr_m15, pip_floor)
    if bias == "LONG":
        ema_ok = cur["ema9"] >= cur["ema21"] - tol
    else:
        ema_ok = cur["ema9"] <= cur["ema21"] + tol
    return ema_ok and rsi_ok


def _body(c) -> float:
    return abs(c["close"] - c["open"])

def _upper_wick(c) -> float:
    return c["high"] - max(c["open"], c["close"])

def _lower_wick(c) -> float:
    return min(c["open"], c["close"]) - c["low"]

def _range(c) -> float:
    return c["high"] - c["low"]


def is_bullish_engulfing(prev, cur, atr_val: float = 0.0) -> bool:
    body = cur["close"] - cur["open"]
    body_ok = atr_val <= 0 or body >= 0.4 * atr_val
    return (
        prev["close"] < prev["open"] and
        cur["close"] > cur["open"] and
        cur["close"] >= prev["open"] and
        cur["open"] <= prev["close"] and
        body_ok
    )


def is_bearish_engulfing(prev, cur, atr_val: float = 0.0) -> bool:
    body = cur["open"] - cur["close"]
    body_ok = atr_val <= 0 or body >= 0.4 * atr_val
    return (
        prev["close"] > prev["open"] and
        cur["close"] < cur["open"] and
        cur["close"] <= prev["open"] and
        cur["open"] >= prev["close"] and
        body_ok
    )


def is_hammer(cur, atr_val: float = 0.0) -> bool:
    """Hammer : longue mèche basse >= 2× corps, petite mèche haute."""
    b = _body(cur)
    lw = _lower_wick(cur)
    uw = _upper_wick(cur)
    if b <= 0:
        return False
    return lw >= 2.0 * b and uw <= 0.5 * b and (atr_val <= 0 or _range(cur) >= 0.3 * atr_val)


def is_shooting_star(cur, atr_val: float = 0.0) -> bool:
    """Shooting Star : longue mèche haute >= 2× corps, petite mèche basse."""
    b = _body(cur)
    uw = _upper_wick(cur)
    lw = _lower_wick(cur)
    if b <= 0:
        return False
    return uw >= 2.0 * b and lw <= 0.5 * b and (atr_val <= 0 or _range(cur) >= 0.3 * atr_val)


def is_pin_bar_bullish(cur, atr_val: float = 0.0) -> bool:
    """Pin bar haussier : mèche basse >= 2/3 de la bougie totale."""
    rng = _range(cur)
    if rng <= 0:
        return False
    lw = _lower_wick(cur)
    return lw >= 0.66 * rng and (atr_val <= 0 or rng >= 0.5 * atr_val)


def is_pin_bar_bearish(cur, atr_val: float = 0.0) -> bool:
    """Pin bar baissier : mèche haute >= 2/3 de la bougie totale."""
    rng = _range(cur)
    if rng <= 0:
        return False
    uw = _upper_wick(cur)
    return uw >= 0.66 * rng and (atr_val <= 0 or rng >= 0.5 * atr_val)


def is_doji(cur, atr_val: float = 0.0) -> bool:
    """Doji : corps < 10% de la range totale (indécision)."""
    rng = _range(cur)
    if rng <= 0:
        return False
    return _body(cur) <= 0.1 * rng


def is_marubozu_bullish(cur, atr_val: float = 0.0) -> bool:
    """Marubozu haussier : bougie verte sans mèches (momentum fort)."""
    rng = _range(cur)
    if rng <= 0 or cur["close"] <= cur["open"]:
        return False
    return _upper_wick(cur) <= 0.05 * rng and _lower_wick(cur) <= 0.05 * rng


def is_marubozu_bearish(cur, atr_val: float = 0.0) -> bool:
    """Marubozu baissier : bougie rouge sans mèches (momentum fort)."""
    rng = _range(cur)
    if rng <= 0 or cur["close"] >= cur["open"]:
        return False
    return _upper_wick(cur) <= 0.05 * rng and _lower_wick(cur) <= 0.05 * rng


def is_morning_star(df, atr_val: float = 0.0) -> bool:
    """Morning Star : bougie baissière + doji + bougie haussière."""
    if len(df) < 3:
        return False
    c1, c2, c3 = df.iloc[-3], df.iloc[-2], df.iloc[-1]
    return (c1["close"] < c1["open"] and
            is_doji(c2) and
            c3["close"] > c3["open"] and
            c3["close"] > (c1["open"] + c1["close"]) / 2)


def is_evening_star(df, atr_val: float = 0.0) -> bool:
    """Evening Star : bougie haussière + doji + bougie baissière."""
    if len(df) < 3:
        return False
    c1, c2, c3 = df.iloc[-3], df.iloc[-2], df.iloc[-1]
    return (c1["close"] > c1["open"] and
            is_doji(c2) and
            c3["close"] < c3["open"] and
            c3["close"] < (c1["open"] + c1["close"]) / 2)


def is_bullish_harami(prev, cur) -> bool:
    """Harami haussier : petite bougie verte à l'intérieur d'une grande rouge."""
    return (prev["close"] < prev["open"] and
            cur["close"] > cur["open"] and
            cur["open"] >= prev["close"] and
            cur["close"] <= prev["open"])


def is_bearish_harami(prev, cur) -> bool:
    """Harami baissier : petite bougie rouge à l'intérieur d'une grande verte."""
    return (prev["close"] > prev["open"] and
            cur["close"] < cur["open"] and
            cur["open"] <= prev["close"] and
            cur["close"] >= prev["open"])


def is_three_white_soldiers(df, atr_val: float = 0.0) -> bool:
    """Trois soldats blancs : 3 bougies vertes consécutives en escalier (continuation forte)."""
    if len(df) < 3:
        return False
    c1, c2, c3 = df.iloc[-3], df.iloc[-2], df.iloc[-1]
    bulls = all(c["close"] > c["open"] for c in (c1, c2, c3))
    rising = c2["close"] > c1["close"] and c3["close"] > c2["close"]
    bodies_ok = atr_val <= 0 or all(_body(c) >= 0.3 * atr_val for c in (c1, c2, c3))
    return bulls and rising and bodies_ok


def is_three_black_crows(df, atr_val: float = 0.0) -> bool:
    """Trois corbeaux noirs : 3 bougies rouges consécutives en escalier (continuation forte)."""
    if len(df) < 3:
        return False
    c1, c2, c3 = df.iloc[-3], df.iloc[-2], df.iloc[-1]
    bears = all(c["close"] < c["open"] for c in (c1, c2, c3))
    falling = c2["close"] < c1["close"] and c3["close"] < c2["close"]
    bodies_ok = atr_val <= 0 or all(_body(c) >= 0.3 * atr_val for c in (c1, c2, c3))
    return bears and falling and bodies_ok


def is_tweezer_bottom(prev, cur, atr_val: float = 0.0) -> bool:
    """Pincettes bas : deux bougies avec des plus-bas quasi identiques (support, retournement haussier)."""
    rng = max(_range(prev), _range(cur))
    if rng <= 0:
        return False
    same_low = abs(prev["low"] - cur["low"]) <= 0.1 * rng
    return (prev["close"] < prev["open"] and
            cur["close"] > cur["open"] and
            same_low)


def is_tweezer_top(prev, cur, atr_val: float = 0.0) -> bool:
    """Pincettes haut : deux bougies avec des plus-hauts quasi identiques (résistance, retournement baissier)."""
    rng = max(_range(prev), _range(cur))
    if rng <= 0:
        return False
    same_high = abs(prev["high"] - cur["high"]) <= 0.1 * rng
    return (prev["close"] > prev["open"] and
            cur["close"] < cur["open"] and
            same_high)


def is_piercing_line(prev, cur, atr_val: float = 0.0) -> bool:
    """Ligne perçante : rouge puis verte qui clôture au-dessus du milieu de la précédente."""
    if not (prev["close"] < prev["open"] and cur["close"] > cur["open"]):
        return False
    mid = (prev["open"] + prev["close"]) / 2
    return (cur["open"] < prev["close"] and
            cur["close"] > mid and
            cur["close"] < prev["open"])


def is_dark_cloud_cover(prev, cur, atr_val: float = 0.0) -> bool:
    """Couverture en nuage noir : verte puis rouge qui clôture sous le milieu de la précédente."""
    if not (prev["close"] > prev["open"] and cur["close"] < cur["open"]):
        return False
    mid = (prev["open"] + prev["close"]) / 2
    return (cur["open"] > prev["close"] and
            cur["close"] < mid and
            cur["close"] > prev["open"])


def ema9_pullback_bounce(m5: pd.DataFrame, bias: str, min_pullback_atr: float = 0.5) -> bool:
    """Pullback to EMA9 then rejection in bias direction.

    Exige que le prix ait fait ≥ min_pullback_atr×ATR de pullback depuis EMA9
    dans les 6 bougies précédentes — évite les effleurements sans conviction.
    """
    if len(m5) < 5:
        return False
    cur, prev = m5.iloc[-1], m5.iloc[-2]
    atr_val = float(cur.get("atr", 0) or 0)

    if bias == "LONG":
        touched = prev["low"] <= prev["ema9"]
        bounce  = cur["close"] > cur["ema9"] and cur["close"] > cur["open"]
        if not (touched and bounce):
            return False
        if atr_val > 0:
            lookback = m5.iloc[-8:-2]
            if len(lookback) > 0:
                recent_high = float(lookback["high"].max())
                if recent_high - float(prev["ema9"]) < min_pullback_atr * atr_val:
                    return False
        return True
    else:
        touched = prev["high"] >= prev["ema9"]
        bounce  = cur["close"] < cur["ema9"] and cur["close"] < cur["open"]
        if not (touched and bounce):
            return False
        if atr_val > 0:
            lookback = m5.iloc[-8:-2]
            if len(lookback) > 0:
                recent_low = float(lookback["low"].min())
                if float(prev["ema9"]) - recent_low < min_pullback_atr * atr_val:
                    return False
        return True


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
def _rej(log: Optional[Dict], stage: str) -> None:
    if log is not None:
        log[stage] = log.get(stage, 0) + 1


def evaluate(
    m5: pd.DataFrame,
    m15: pd.DataFrame,
    h1: pd.DataFrame,
    h4: Optional[pd.DataFrame] = None,
    now: Optional[datetime] = None,
    check_session: bool = True,
    atr_min: float = ATR_MIN,
    pattern_weights: Optional[Dict[str, float]] = None,
    ml_gate=None,
    adaptive_thresholds=None,
    _reject_log: Optional[Dict] = None,
) -> Optional[Signal]:
    """
    Evaluate the full multi-timeframe stack on the *last closed* M5 bar.
    Returns a Signal or None.

    DataFrames must already contain indicators (call add_indicators).
    """
    if len(m5) < max(EMA_SLOW, 30) or len(m15) < 1 or len(h1) < 1:
        return None

    cur = m5.iloc[-1]
    prev = m5.iloc[-2]
    ts = now or cur.name.to_pydatetime()
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)

    # 0) Bad timing (Monday open / Friday close)
    if is_bad_timing(ts):
        _rej(_reject_log, "timing"); return None

    # 1) Session gate
    session = active_session(ts)
    if check_session and session is None:
        _rej(_reject_log, "session"); return None
    session = session or "London"

    # 2) H1 EMA200 bias
    bias = compute_bias(h1)
    if bias == "NEUTRE":
        _rej(_reject_log, "h1_neutre"); return None

    # 2b) Distance EMA200 — bloquer SHORT si prix trop haut au-dessus EMA200 (uptrend fort)
    if len(h1) > 0:
        h1_last = h1.iloc[-1]
        h1_ema200_val = float(h1_last.get("ema200", float("nan")))
        h1_atr_val = float(h1_last.get("atr", 0) or 0)
        if not pd.isna(h1_ema200_val) and h1_atr_val > 0:
            price_vs_ema200 = (float(h1_last["close"]) - h1_ema200_val) / h1_atr_val
            if price_vs_ema200 > TREND_BIAS_DISTANCE and bias == "SHORT":
                _rej(_reject_log, "h1_ema200"); return None
            if price_vs_ema200 < -TREND_BIAS_DISTANCE and bias == "LONG":
                _rej(_reject_log, "h1_ema200"); return None

    # H4 EMA200 bias filter supprimé — H1 suffit pour le biais directionnel

    # Seuils adaptatifs (si disponibles et entraînés)
    _adapt = adaptive_thresholds
    _adapt_ready = _adapt is not None and _adapt.is_ready
    effective_atr_min  = _adapt.atr_min   if _adapt_ready else atr_min
    effective_ema9_mult = _adapt.ema9_mult if _adapt_ready else 0.5
    effective_m15_mult  = _adapt.m15_mult  if _adapt_ready else 0.5

    # 3) M15 EMA9/21 + RSI confirmation
    if not confirm_m15(m15, bias, ema_mult=effective_m15_mult):
        _rej(_reject_log, "m15"); return None

    # 4) M5 volatility gate — plancher ET plafond
    atr_val = float(cur["atr"]) if not pd.isna(cur["atr"]) else 0.0
    if atr_val < effective_atr_min:
        _rej(_reject_log, "atr_min"); return None
    if atr_val > ATR_MAX:
        _rej(_reject_log, "atr_max"); return None  # trop volatile → whipsaw → SL direct

    # 4c) Régime volatilité — ATR actuel vs moyenne 20 bougies (filtre marché range)
    if len(m5) >= 20:
        atr_avg = float(m5["atr"].iloc[-20:].mean())
        if atr_avg > 0 and atr_val / atr_avg < ATR_REGIME_MIN_RATIO:
            return None

    # 4b) H1 ADX trend strength — ne trader qu'en vraie tendance
    h1_adx = float(h1.iloc[-1].get("adx", 0)) if len(h1) else 0.0
    adx_required = ADX_MIN  # même seuil LONG et SHORT
    if h1_adx < adx_required:
        _rej(_reject_log, "adx"); return None

    # 4d) ADX pente — évite les entrées sur momentum épuisé
    # ADX doit être en hausse sur la dernière bougie H1 (1 bougie — amorçage apprentissage)
    if len(h1) >= 2:
        adx_prev1 = float(h1.iloc[-2].get("adx", h1_adx))
        if h1_adx < adx_prev1:
            _rej(_reject_log, "adx_slope"); return None

    # 5) M5 EMA9 alignment — tolérance adaptative (défaut 0.5 ATR)
    ema9_tolerance = atr_val * effective_ema9_mult
    if bias == "LONG" and cur["close"] < cur["ema9"] - ema9_tolerance:
        _rej(_reject_log, "ema9"); return None
    if bias == "SHORT" and cur["close"] > cur["ema9"] + ema9_tolerance:
        _rej(_reject_log, "ema9"); return None

    # 5b) M5 RSI momentum confirmation — seuils ajustables par LiveAdaptiveAgent
    rsi_m5 = float(cur.get("rsi", 50) or 50)
    if bias == "LONG"  and rsi_m5 < RSI_M5_LONG_MIN:
        _rej(_reject_log, "rsi_m5"); return None
    if bias == "SHORT" and rsi_m5 > RSI_M5_SHORT_MAX:
        _rej(_reject_log, "rsi_m5"); return None

    # 5c) VWAP alignment — close du bon côté du VWAP
    vwap_val = float(cur.get("vwap", float("nan")) or float("nan"))
    if not pd.isna(vwap_val):
        if bias == "LONG"  and float(cur["close"]) < vwap_val:
            _rej(_reject_log, "vwap"); return None
        if bias == "SHORT" and float(cur["close"]) > vwap_val:
            _rej(_reject_log, "vwap"); return None

    # 6) Candlestick pattern trigger (any single pattern is enough)
    entry = float(cur["close"])
    triggers = []

    if bias == "LONG":
        if is_bullish_engulfing(prev, cur, atr_val):  triggers.append("bullish_engulfing")
        # hammer exclu (WR 20% sur données historiques)
        if is_pin_bar_bullish(cur, atr_val):          triggers.append("pin_bar")
        if is_marubozu_bullish(cur, atr_val):         triggers.append("marubozu")
        # morning_star exclu (WR 42.9% sur données historiques)
        if is_bullish_harami(prev, cur):              triggers.append("harami")

        if is_three_white_soldiers(m5.iloc[-3:], atr_val): triggers.append("three_white_soldiers")
        if is_tweezer_bottom(prev, cur, atr_val):     triggers.append("tweezer_bottom")
        if is_piercing_line(prev, cur, atr_val):      triggers.append("piercing_line")
        if ema9_pullback_bounce(m5, bias):            triggers.append("ema9_pullback")
        if micro_breakout(m5, bias):                  triggers.append("micro_breakout")
        if is_doji(prev):                             triggers.append("doji_reversal")
    else:
        if is_bearish_engulfing(prev, cur, atr_val):  triggers.append("bearish_engulfing")
        if is_shooting_star(cur, atr_val):            triggers.append("shooting_star")
        if is_pin_bar_bearish(cur, atr_val):          triggers.append("pin_bar")
        if is_marubozu_bearish(cur, atr_val):         triggers.append("marubozu")
        if is_evening_star(m5.iloc[-3:], atr_val):   triggers.append("evening_star")
        if is_bearish_harami(prev, cur):              triggers.append("bearish_harami")
        if is_three_black_crows(m5.iloc[-3:], atr_val): triggers.append("three_black_crows")
        if is_tweezer_top(prev, cur, atr_val):        triggers.append("tweezer_top")
        if is_dark_cloud_cover(prev, cur, atr_val):   triggers.append("dark_cloud_cover")
        if ema9_pullback_bounce(m5, bias):            triggers.append("ema9_pullback")
        if micro_breakout(m5, bias):                  triggers.append("micro_breakout")
        if is_doji(prev):                             triggers.append("doji_reversal")

    # Order block proximity — confluence only, never a standalone trigger
    obs = find_order_blocks(m5)
    if triggers and near_orderblock(entry, bias, obs, atr_val):
        triggers.append("near_order_block")

    # FVG confluence — add weight if price is inside a matching Fair Value Gap
    fvgs = find_fvgs(m5)
    if triggers and near_fvg(entry, bias, fvgs):
        triggers.append("near_fvg")

    # Entry gating: sum of weights >= 1.0 AND average weight >= 0.45
    def _w(t: str) -> float:
        if pattern_weights is None:
            return 1.0
        info = pattern_weights.get(t)
        return info["weight"] if isinstance(info, dict) else float(info) if info else 1.0

    # Exclure les patterns sous le seuil de qualité
    triggers = [t for t in triggers if _w(t) >= PATTERN_FLOOR]

    weights = [_w(t) for t in triggers]
    weight_total = sum(weights)
    min_weight_sum = MIN_WEIGHT_SUM_LONG if bias == "LONG" else 1.5

    ANCHOR_PATTERNS = {"ema9_pullback", "micro_breakout"}
    has_anchor = bool(set(triggers) & ANCHOR_PATTERNS)

    # Passage : soit 1 pattern fort (ancre obligatoire, weight ≥ 0.85)
    #           soit 2+ patterns avec confluence suffisante (ancre toujours requise)
    single_strong = (len(triggers) == 1 and weight_total >= 0.85 and has_anchor)
    multi_ok      = (len(triggers) >= 2 and weight_total >= min_weight_sum and has_anchor)

    if not (single_strong or multi_ok):
        _rej(_reject_log, "patterns"); return None

    # Filtre corps de bougie : rejette les bougies indécises (corps < 40% de la range)
    # Exempt pour les patterns conçus avec petite bougie (hammer, pin_bar, doji, tweezer)
    SMALL_BODY_EXEMPT = {
        "hammer", "pin_bar", "doji_reversal", "shooting_star",
        "tweezer_bottom", "tweezer_top", "piercing_line", "dark_cloud_cover",
    }
    if not set(triggers) & SMALL_BODY_EXEMPT:
        bar_range = float(cur["high"]) - float(cur["low"])
        bar_body  = abs(float(cur["close"]) - float(cur["open"]))
        if bar_range > 0 and bar_body / bar_range < 0.4:
            _rej(_reject_log, "body"); return None

    # 7) Build trade levels
    weight_sum = weight_total
    sl_mult = SL_ATR_MULT

    if bias == "LONG":
        swing = last_swing_low(m5, lookback=10)
        raw_sl = min(swing, entry - 1e-6)
        sl = max(raw_sl, entry - sl_mult * atr_val)
        direction = "long"
    else:
        swing = last_swing_high(m5, lookback=10)
        raw_sl = max(swing, entry + 1e-6)
        sl = min(raw_sl, entry + sl_mult * atr_val)
        direction = "short"

    risk = abs(entry - sl)
    if risk <= 0:
        return None

    if direction == "long":
        tp1 = entry + 0.7 * risk
        tp2 = entry + 1.8 * risk
    else:
        tp1 = entry - 0.7 * risk
        tp2 = entry - 1.8 * risk

    # Extraction des features ML — toujours calculées (gate live + pré-entraînement)
    ml_prob: float = -1.0
    ml_features = None
    try:
        from ml_gate import extract_features

        # Fraction horaire dans la session (0=début, 1=fin)
        _cet_ts = ts.astimezone(CET)
        _cet_h = _cet_ts.hour + _cet_ts.minute / 60.0
        if session == "London":
            _sess_frac = (_cet_h - LONDON[0].hour) / 4.0
        elif session == "NewYork":
            _sess_frac = (_cet_h - NEWYORK[0].hour) / 4.0
        else:
            _sess_frac = 0.5
        _sess_frac = max(0.0, min(1.0, _sess_frac))

        h1_rsi_val = float(h1.iloc[-1].get("rsi", 50) or 50) if len(h1) > 0 else 50.0

        weight_sum = sum([_w(t) for t in triggers])
        ml_features = extract_features(
            m5, m15, bias, session, weight_sum, ts,
            h1_adx=h1_adx, h1_rsi=h1_rsi_val,
            n_patterns=len(triggers), session_hour_frac=_sess_frac,
        )
        if ml_gate is not None:
            allowed, ml_prob = ml_gate.gate(ml_features)
            if not allowed:
                return None
    except Exception:
        pass

    meta: Dict[str, Any] = {
        "rsi_m5":    round(float(cur["rsi"]), 1),
        "rsi_m15":   round(float(m15.iloc[-1]["rsi"]), 1),
        "triggers":  triggers,
        "ml_prob":   round(ml_prob, 3) if ml_prob >= 0 else None,
    }
    if ml_features is not None:
        meta["ml_features"] = ml_features

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
        meta=meta,
    )


def evaluate_eurusd(
    m5: pd.DataFrame,
    m15: pd.DataFrame,
    h1: pd.DataFrame,
    now: Optional[datetime] = None,
    check_session: bool = True,
    atr_min: float = 0.00030,
    pattern_weights: Optional[Dict[str, float]] = None,
    ml_gate=None,
    _reject_log: Optional[Dict] = None,
) -> Optional[Signal]:
    """
    Stratégie EUR/USD simplifiée : H1 EMA200 bias + ancre EMA9/OB + patterns.
    Pas de filtre M15, ADX, VWAP, RSI. ATR en valeur absolue (EUR/USD natif).
    """
    if len(m5) < 3 or len(h1) < 1:
        return None

    cur  = m5.iloc[-1]
    prev = m5.iloc[-2]
    if hasattr(cur.name, "tzinfo") and cur.name.tzinfo is None:
        ts = now or pd.Timestamp(cur.name).tz_localize("UTC")
    else:
        ts = now or pd.Timestamp(cur.name)

    # 1) Bad timing (lundi matin, vendredi soir, heures bloquées)
    if is_bad_timing(ts):
        _rej(_reject_log, "timing"); return None

    # 2) Session gate
    local = ts.astimezone(CET)
    if LONDON[0] <= local.time() < LONDON[1]:
        session = "London"
    elif NEWYORK[0] <= local.time() < NEWYORK[1]:
        session = "NewYork"
    else:
        if check_session:
            _rej(_reject_log, "session"); return None
        session = "London"

    # 3) H1 EMA200 bias
    bias = compute_bias(h1)
    if bias == "NEUTRE":
        _rej(_reject_log, "h1_neutre"); return None

    # 4) ATR gate
    atr_val = float(cur.get("atr", 0) or 0)
    if atr_val < atr_min:
        _rej(_reject_log, "atr_min"); return None

    entry = float(cur["close"])

    # 5) Patterns — bons patterns uniquement, pas de morning_star/hammer/doji/shooting_star
    triggers: List[str] = []
    if bias == "LONG":
        if is_bullish_engulfing(prev, cur, atr_val):        triggers.append("bullish_engulfing")
        if is_pin_bar_bullish(cur, atr_val):                triggers.append("pin_bar")
        if is_bullish_harami(prev, cur):                    triggers.append("harami")
        if is_three_white_soldiers(m5.iloc[-3:], atr_val): triggers.append("three_white_soldiers")
        if is_tweezer_bottom(prev, cur, atr_val):           triggers.append("tweezer_bottom")
        if is_marubozu_bullish(cur, atr_val):               triggers.append("marubozu")
        if ema9_pullback_bounce(m5, bias):                  triggers.append("ema9_pullback")
        obs = find_order_blocks(m5)
        if near_orderblock(entry, bias, obs, atr_val):      triggers.append("near_order_block")
    else:
        if is_bearish_engulfing(prev, cur, atr_val):        triggers.append("bearish_engulfing")
        if is_pin_bar_bearish(cur, atr_val):                triggers.append("pin_bar")
        if is_bearish_harami(prev, cur):                    triggers.append("bearish_harami")
        if is_three_black_crows(m5.iloc[-3:], atr_val):    triggers.append("three_black_crows")
        if is_tweezer_top(prev, cur, atr_val):              triggers.append("tweezer_top")
        if is_marubozu_bearish(cur, atr_val):               triggers.append("marubozu")
        if ema9_pullback_bounce(m5, bias):                  triggers.append("ema9_pullback")
        obs = find_order_blocks(m5)
        if near_orderblock(entry, bias, obs, atr_val):      triggers.append("near_order_block")

    # Filtre qualité (PATTERN_FLOOR)
    def _w(t: str) -> float:
        if pattern_weights is None:
            return 1.0
        info = pattern_weights.get(t)
        return info["weight"] if isinstance(info, dict) else float(info) if info else 1.0

    triggers = [t for t in triggers if _w(t) >= PATTERN_FLOOR]

    # Ancre obligatoire : EMA9 pullback OU Order Block
    if not set(triggers) & {"ema9_pullback", "near_order_block"}:
        _rej(_reject_log, "no_anchor"); return None

    weights = [_w(t) for t in triggers]
    if not triggers or sum(weights) < 1.0:
        _rej(_reject_log, "patterns"); return None

    # 6) Niveaux de trade — SL sous le low/high de la bougie signal (serré vs swing lookback XAU)
    if bias == "LONG":
        sl = min(float(prev["low"]), float(cur["low"])) - atr_val * 0.1
        sl = max(sl, entry - SL_ATR_MULT * atr_val)  # plafond : jamais plus large que 1.4×ATR
        direction = "long"
    else:
        sl = max(float(prev["high"]), float(cur["high"])) + atr_val * 0.1
        sl = min(sl, entry + SL_ATR_MULT * atr_val)
        direction = "short"

    risk = abs(entry - sl)
    if risk <= 0:
        return None

    tp1 = entry + 0.7 * risk if direction == "long" else entry - 0.7 * risk
    tp2 = entry + 1.2 * risk if direction == "long" else entry - 1.2 * risk  # EUR/USD : TP2 réduit vs XAU (1.8R)

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
        meta={"triggers": triggers, "bias": bias, "strategy": "eurusd_simple"},
    )


def batch_signals(
    m5: pd.DataFrame,
    m15: pd.DataFrame,
    h1: pd.DataFrame,
    h4: Optional[pd.DataFrame] = None,
    check_session: bool = True,
    atr_min: float = ATR_MIN,
) -> pd.Series:
    """
    Vectorised signal generation for the full backtest period.

    Instead of calling evaluate() O(n) times (each O(n) → O(n²) total),
    this function computes ALL conditions as pandas column operations in
    a single forward pass → O(n).

    Returns a Series indexed like m5 with values:
        "long"  — enter long at this bar
        "short" — enter short at this bar
        None    — no signal

    Covers ~90% of evaluate()'s logic. Complex per-bar checks (SMC order
    blocks, FVG, S/R proximity) are deliberately omitted here to keep
    vectorisation clean; they are still applied in the live engine via
    evaluate().
    """
    if len(m5) < EMA_SLOW + 10:
        return pd.Series(None, index=m5.index, dtype=object)

    c = m5["close"]
    o = m5["open"]
    h = m5["high"]
    lo = m5["low"]

    # ── H1 bias (forward-filled onto M5 timestamps) ──────────────────────────
    h1_close  = h1["close"].reindex(m5.index, method="ffill")
    h1_ema200 = h1["ema200"].reindex(m5.index, method="ffill") if "ema200" in h1.columns else None
    h1_adx    = h1["adx"].reindex(m5.index, method="ffill")   if "adx"    in h1.columns else None

    if h1_ema200 is not None:
        bias_long  = h1_close > h1_ema200
        bias_short = h1_close < h1_ema200
    else:
        bias_long  = pd.Series(False, index=m5.index)
        bias_short = pd.Series(False, index=m5.index)

    adx_ok = (h1_adx >= ADX_MIN).reindex(m5.index, method="ffill").fillna(False) if h1_adx is not None else pd.Series(True, index=m5.index)

    # ── M15 confirmation (forward-filled) ────────────────────────────────────
    m15_ema9  = m15["ema9"].reindex(m5.index,  method="ffill") if "ema9"  in m15.columns else None
    m15_ema21 = m15["ema21"].reindex(m5.index, method="ffill") if "ema21" in m15.columns else None
    m15_rsi   = m15["rsi"].reindex(m5.index,   method="ffill") if "rsi"   in m15.columns else None

    if m15_ema9 is not None and m15_ema21 is not None:
        m15_bull = m15_ema9 > m15_ema21
        m15_bear = m15_ema9 < m15_ema21
    else:
        m15_bull = pd.Series(True, index=m5.index)
        m15_bear = pd.Series(True, index=m5.index)

    rsi_ok = pd.Series(True, index=m5.index)
    if m15_rsi is not None:
        rsi_ok = (m15_rsi >= RSI_LOW) & (m15_rsi <= RSI_HIGH)

    # ── M5 indicators ────────────────────────────────────────────────────────
    atr   = m5["atr"]   if "atr"   in m5.columns else pd.Series(1.0, index=m5.index)
    ema9  = m5["ema9"]  if "ema9"  in m5.columns else c
    ema21 = m5["ema21"] if "ema21" in m5.columns else c

    atr_ok    = atr >= atr_min
    ema9_long = c > ema9
    ema9_sht  = c < ema9

    # ── Session gate ─────────────────────────────────────────────────────────
    if check_session:
        cet_idx = m5.index.tz_convert(CET)
        hour    = pd.Series(cet_idx.hour, index=m5.index)
        in_london  = (hour >= 8)  & (hour < 12)
        in_newyork = (hour >= 14) & (hour < 18)
        session_ok = in_london | in_newyork
    else:
        session_ok = pd.Series(True, index=m5.index)

    # ── Candlestick patterns (vectorised) ────────────────────────────────────
    body      = (c - o).abs()
    rng       = (h - lo).clip(lower=1e-9)
    prev_c    = c.shift(1)
    prev_o    = o.shift(1)
    prev_body = (prev_c - prev_o).abs()

    # Bullish engulfing
    bull_eng = (prev_c < prev_o) & (c > o) & (c > prev_o) & (o < prev_c) & (body >= 0.4 * atr)
    # Bearish engulfing
    bear_eng = (prev_c > prev_o) & (c < o) & (c < prev_o) & (o > prev_c) & (body >= 0.4 * atr)
    # Hammer
    lower_wick = (o.where(c >= o, c) - lo)
    hammer     = (lower_wick >= 2 * body) & (body >= 0.1 * rng) & (c >= o)
    # Shooting star
    upper_wick = (h - c.where(c >= o, o))
    shooting   = (upper_wick >= 2 * body) & (body >= 0.1 * rng) & (c < o)
    # Pin bar bullish
    pin_bull   = (lower_wick >= 0.6 * rng) & (body <= 0.3 * rng)
    # Pin bar bearish
    pin_bear   = (upper_wick >= 0.6 * rng) & (body <= 0.3 * rng)

    # Any bullish / bearish pattern
    bull_pattern = bull_eng | hammer | pin_bull
    bear_pattern = bear_eng | shooting | pin_bear

    # ── H4 bias confirmation ──────────────────────────────────────────────────
    if h4 is not None and len(h4) > 0 and "ema200" in h4.columns:
        h4_close  = h4["close"].reindex(m5.index, method="ffill")
        h4_ema200 = h4["ema200"].reindex(m5.index, method="ffill")
        h4_long_ok  = h4_close > h4_ema200
        h4_short_ok = h4_close < h4_ema200
    else:
        h4_long_ok  = pd.Series(True, index=m5.index)
        h4_short_ok = pd.Series(True, index=m5.index)

    # ── Warmup mask ───────────────────────────────────────────────────────────
    warmup = pd.Series(False, index=m5.index)
    warmup.iloc[:EMA_SLOW + 10] = True

    # ── Combine ───────────────────────────────────────────────────────────────
    long_signal  = (
        ~warmup & bias_long & adx_ok & m15_bull & rsi_ok
        & atr_ok & ema9_long & session_ok & bull_pattern & h4_long_ok
    )
    short_signal = (
        ~warmup & bias_short & adx_ok & m15_bear & rsi_ok
        & atr_ok & ema9_sht  & session_ok & bear_pattern & h4_short_ok
    )

    out = pd.Series(None, index=m5.index, dtype=object)
    out[long_signal]  = "long"
    out[short_signal] = "short"
    # resolve conflicts: skip if both fire on same bar
    out[long_signal & short_signal] = None
    return out


def snapshot(m5: pd.DataFrame, m15: pd.DataFrame, h1: pd.DataFrame,
             now: Optional[datetime] = None,
             atr_min_override: float = ATR_MIN,
             pattern_weights: Optional[Dict[str, Any]] = None,
             ml_gate=None,
             adaptive_thresholds=None) -> Dict[str, Any]:
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

    # Seuils adaptatifs pour le diagnostic
    _adapt = adaptive_thresholds
    _adapt_ready = _adapt is not None and _adapt.is_ready
    snap_atr_min   = _adapt.atr_min    if _adapt_ready else atr_min_override
    snap_ema9_mult = _adapt.ema9_mult  if _adapt_ready else 0.5
    snap_m15_mult  = _adapt.m15_mult   if _adapt_ready else 0.3

    # --- condition diagnostics ---
    m15_confirmed = confirm_m15(m15, bias, ema_mult=snap_m15_mult) if (len(m15) >= 1 and bias != "NEUTRE") else False

    # Diagnostic fin de la M15 : on sépare l'alignement EMA9/21 de la bande RSI
    # pour savoir laquelle des deux sous-conditions bloque réellement.
    m15_ema_aligned = None
    m15_rsi_ok = None
    m15_ema9 = None
    m15_ema21 = None
    m15_ema_gap = None
    m15_ema_tol = None
    if bias != "NEUTRE" and len(m15) >= 1:
        cur15_d = m15.iloc[-1]
        if not any(pd.isna(cur15_d.get(c, float("nan"))) for c in ("ema9", "ema21", "rsi")):
            atr_m15_d = float(cur15_d.get("atr", 0) or 0)
            price_m15 = float(cur15_d.get("close", 1) or 1)
            tol_d = max(snap_m15_mult * atr_m15_d, price_m15 * 0.0001)
            m15_ema9 = round(float(cur15_d["ema9"]), 3)
            m15_ema21 = round(float(cur15_d["ema21"]), 3)
            m15_ema_gap = round(float(cur15_d["ema9"]) - float(cur15_d["ema21"]), 3)
            m15_ema_tol = round(tol_d, 3)
            if bias == "LONG":
                m15_ema_aligned = bool(cur15_d["ema9"] >= cur15_d["ema21"] - tol_d)
            else:
                m15_ema_aligned = bool(cur15_d["ema9"] <= cur15_d["ema21"] + tol_d)
            m15_rsi_ok = bool(RSI_LOW <= float(cur15_d["rsi"]) <= RSI_HIGH)

    atr_val = float(cur5["atr"]) if (cur5 is not None and not pd.isna(cur5.get("atr", float("nan")))) else 0.0
    atr_ok = atr_val >= snap_atr_min

    ema9_aligned = False
    if bias != "NEUTRE" and cur5 is not None:
        ema9_v = cur5.get("ema9", float("nan"))
        if not pd.isna(ema9_v):
            ema9_tol = atr_val * snap_ema9_mult
            if bias == "LONG":
                ema9_aligned = cur5["close"] >= float(ema9_v) - ema9_tol
            else:
                ema9_aligned = cur5["close"] <= float(ema9_v) + ema9_tol

    # NOTE: cette liste doit rester alignée avec les triggers de evaluate()
    # pour que l'affichage du dashboard reflète exactement la logique d'entrée.
    patterns_detected: List[str] = []
    if bias != "NEUTRE" and cur5 is not None and len(m5) >= 2:
        prev5 = m5.iloc[-2]
        if bias == "LONG":
            if is_bullish_engulfing(prev5, cur5, atr_val): patterns_detected.append("bullish_engulfing")
            if is_hammer(cur5, atr_val): patterns_detected.append("hammer")
            if is_pin_bar_bullish(cur5, atr_val): patterns_detected.append("pin_bar")
            if is_marubozu_bullish(cur5, atr_val): patterns_detected.append("marubozu")
            # morning_star exclu (WR 42.9% sur données historiques)
            if is_bullish_harami(prev5, cur5): patterns_detected.append("harami")
            if is_three_white_soldiers(m5.iloc[-3:], atr_val): patterns_detected.append("three_white_soldiers")
            if is_tweezer_bottom(prev5, cur5, atr_val): patterns_detected.append("tweezer_bottom")
            if is_piercing_line(prev5, cur5, atr_val): patterns_detected.append("piercing_line")
            if ema9_pullback_bounce(m5, bias): patterns_detected.append("ema9_pullback")
            if micro_breakout(m5, bias): patterns_detected.append("micro_breakout")
            if is_doji(prev5): patterns_detected.append("doji_reversal")
        else:
            if is_bearish_engulfing(prev5, cur5, atr_val): patterns_detected.append("bearish_engulfing")
            if is_shooting_star(cur5, atr_val): patterns_detected.append("shooting_star")
            if is_pin_bar_bearish(cur5, atr_val): patterns_detected.append("pin_bar")
            if is_marubozu_bearish(cur5, atr_val): patterns_detected.append("marubozu")
            if is_evening_star(m5.iloc[-3:], atr_val): patterns_detected.append("evening_star")
            if is_bearish_harami(prev5, cur5): patterns_detected.append("bearish_harami")
            if is_three_black_crows(m5.iloc[-3:], atr_val): patterns_detected.append("three_black_crows")
            if is_tweezer_top(prev5, cur5, atr_val): patterns_detected.append("tweezer_top")
            if is_dark_cloud_cover(prev5, cur5, atr_val): patterns_detected.append("dark_cloud_cover")
            if ema9_pullback_bounce(m5, bias): patterns_detected.append("ema9_pullback")
            if micro_breakout(m5, bias): patterns_detected.append("micro_breakout")
            if is_doji(prev5): patterns_detected.append("doji_reversal")

    # Order block proximity for dashboard display
    if bias != "NEUTRE" and cur5 is not None:
        obs_snap = find_order_blocks(m5)
        if near_orderblock(float(cur5["close"]), bias, obs_snap, atr_val):
            patterns_detected.append("near_order_block")

    # Pattern weight computation (mirrors evaluate() logic)
    def _pw(t: str) -> float:
        if pattern_weights is None:
            return 1.0
        info = pattern_weights.get(t)
        return info["weight"] if isinstance(info, dict) else float(info) if info else 1.0

    pattern_weight_detail = {t: round(_pw(t), 3) for t in patterns_detected}
    pattern_weight_sum = round(sum(pattern_weight_detail.values()), 3)
    weight_gate_ok = pattern_weight_sum >= 1.0 if patterns_detected else False

    # ML gate probability (display only — no blocking in snapshot)
    ml_prob: Optional[float] = None
    ml_ready: bool = False
    if ml_gate is not None and cur5 is not None and len(m15) > 0:
        try:
            from ml_gate import extract_features
            ml_features = extract_features(m5, m15, bias, session or "London",
                                           pattern_weight_sum, ts)
            ml_ready = ml_gate.is_ready
            if ml_ready:
                ml_prob = round(ml_gate.predict(ml_features), 3)
        except Exception:
            pass

    # first failing condition for quick diagnosis
    blocking_reason = None
    if bias == "NEUTRE":
        blocking_reason = "bias_neutre"
    elif not m15_confirmed:
        # Diagnostic fin : on précise quelle sous-condition M15 bloque.
        if m15_ema_aligned is False and m15_rsi_ok is False:
            blocking_reason = "m15_ema_et_rsi"
        elif m15_ema_aligned is False:
            blocking_reason = "m15_ema_non_aligné"
        elif m15_rsi_ok is False:
            blocking_reason = "m15_rsi_hors_zone"
        else:
            blocking_reason = "m15_non_confirmé"
    elif not atr_ok:
        blocking_reason = "atr_trop_bas"
    elif not ema9_aligned:
        blocking_reason = "ema9_non_aligné"
    elif not patterns_detected:
        blocking_reason = "aucun_pattern"
    elif not weight_gate_ok:
        blocking_reason = "poids_insuffisants"

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
        "atr_min": atr_min_override,
        "adx_min": ADX_MIN,
        "asian_range": asian,
        "structure_ok": market_structure_ok(h1, bias) if len(h1) >= 6 else None,
        "conditions": {
            "h1_bias": bias,
            "m15_confirmed": m15_confirmed,
            "m15_ema_aligned": m15_ema_aligned,
            "m15_ema9": m15_ema9,
            "m15_ema21": m15_ema21,
            "m15_ema_gap": m15_ema_gap,
            "m15_ema_tol": m15_ema_tol,
            "m15_rsi_ok": m15_rsi_ok,
            "atr_ok": atr_ok,
            "ema9_aligned": ema9_aligned,
            "patterns": patterns_detected,
            "pattern_weight_sum": pattern_weight_sum,
            "pattern_weight_detail": pattern_weight_detail,
            "weight_gate_ok": weight_gate_ok,
            "blocking_reason": blocking_reason,
            "ml_prob": ml_prob,
            "ml_ready": ml_ready,
            "adaptive": _adapt.status() if _adapt is not None else None,
        },
    }