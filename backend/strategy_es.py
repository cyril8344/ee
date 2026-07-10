"""
strategy_es.py
==============
Stratégie Order Flow pour ES (E-mini S&P 500).

Signal = volume d'absorption élevé sur une bougie directionnelle,
confirmé par biais EMA200 + EMA9/21 + RSI momentum.

En live avec DOM NinjaTrader connecté via /api/es/dom,
le signal volume proxy est renforcé par la détection d'absorption réelle.

Paramètres optimisables via pretrain_es.py / walk-forward.
"""

from __future__ import annotations
import math
import pandas as pd
from typing import Optional

# ── Constantes contrat ES ─────────────────────────────────────────────────────
TICK_SIZE   = 0.25   # 1 tick = 0.25 points
TICK_VALUE  = 12.50  # 1 tick = 12.50 $ par contrat
POINT_VALUE = 50.0   # 1 point = 50 $ par contrat

# ── Paramètres par défaut ─────────────────────────────────────────────────────
DEFAULTS: dict = {
    # Filtres tendance
    "ema_fast":       9,
    "ema_slow":       21,
    "ema_trend":      200,

    # Filtres momentum
    "rsi_long":       45,
    "rsi_short":      55,

    # Volatilité
    "atr_min_pts":    2.0,     # ATR minimum en points ES

    # Volume absorption proxy
    "vol_multiplier": 2.0,     # ratio volume / moyenne pour déclencher
    "vol_lookback":   20,      # barres pour calculer la moyenne

    # Absorption bougie : close doit être dans les X% du range
    "close_pct_long": 0.60,    # close ≥ 60% du range pour LONG
    "close_pct_short": 0.40,   # close ≤ 40% du range pour SHORT

    # SL / TP en ticks
    "sl_ticks":   8,    # 2.0 pts = 100 $ par contrat
    "tp1_ticks":  12,   # 3.0 pts = 150 $
    "tp2_ticks":  24,   # 6.0 pts = 300 $

    # Session ET (Eastern Time)
    "session_open_h":  9,
    "session_open_m":  30,
    "session_close_h": 16,
    "session_close_m": 0,
}


# ── Indicateurs ───────────────────────────────────────────────────────────────

def add_indicators(df: pd.DataFrame, params: Optional[dict] = None) -> pd.DataFrame:
    """Calcule EMA, RSI, ATR, volume ratio sur un DataFrame OHLCV."""
    p = {**DEFAULTS, **(params or {})}
    df = df.copy()

    # EMA
    df["ema9"]   = df["close"].ewm(span=int(p["ema_fast"]),  adjust=False).mean()
    df["ema21"]  = df["close"].ewm(span=int(p["ema_slow"]),  adjust=False).mean()
    df["ema200"] = df["close"].ewm(span=int(p["ema_trend"]), adjust=False).mean()

    # RSI 14
    delta    = df["close"].diff()
    gain     = delta.clip(lower=0)
    loss     = (-delta).clip(lower=0)
    avg_gain = gain.ewm(span=14, adjust=False).mean()
    avg_loss = loss.ewm(span=14, adjust=False).mean()
    rs       = avg_gain / avg_loss.replace(0, float("nan"))
    df["rsi"] = 100.0 - 100.0 / (1.0 + rs)

    # ATR 14
    hl = df["high"] - df["low"]
    hc = (df["high"] - df["close"].shift(1)).abs()
    lc = (df["low"]  - df["close"].shift(1)).abs()
    df["atr"] = pd.concat([hl, hc, lc], axis=1).max(axis=1).ewm(span=14, adjust=False).mean()

    # Volume absorption proxy
    lookback        = max(int(p["vol_lookback"]), 3)
    df["vol_avg"]   = df["volume"].rolling(lookback, min_periods=3).mean()
    df["vol_ratio"] = df["volume"] / df["vol_avg"].replace(0, float("nan"))

    return df


# ── Signal principal ──────────────────────────────────────────────────────────

def evaluate(
    bars_5m:    pd.DataFrame,
    params:     Optional[dict] = None,
    ts=None,
    dom_signal: Optional[dict] = None,   # signal DOM live de NinjaTrader
) -> Optional[dict]:
    """
    Évalue la bougie courante et retourne un signal ou None.

    dom_signal (optionnel) : dict envoyé par le DOMScanner NinjaTrader
        {"side": "BUY"|"SELL", "size": 93, "price": 5839.5, "type": "ABSORPTION"}
    """
    p = {**DEFAULTS, **(params or {})}

    if len(bars_5m) < int(p["ema_trend"]) + 10:
        return None

    cur  = bars_5m.iloc[-1]
    close    = float(cur.get("close", 0) or 0)
    bar_high = float(cur.get("high",  0) or 0)
    bar_low  = float(cur.get("low",   0) or 0)
    bar_range = bar_high - bar_low

    # ── 1. Session gate (Eastern Time) ────────────────────────────────────────
    if ts is not None:
        try:
            import pytz
            et_tz     = pytz.timezone("America/New_York")
            ts_et     = ts.astimezone(et_tz)
            dec_hour  = ts_et.hour + ts_et.minute / 60.0
            open_dec  = p["session_open_h"]  + p["session_open_m"]  / 60.0
            close_dec = p["session_close_h"] + p["session_close_m"] / 60.0
            if not (open_dec <= dec_hour < close_dec):
                return None
        except Exception:
            pass

    # ── 2. ATR minimum ────────────────────────────────────────────────────────
    atr = float(cur.get("atr", 0) or 0)
    if atr < float(p["atr_min_pts"]):
        return None

    # ── 3. Biais EMA200 ───────────────────────────────────────────────────────
    ema200 = float(cur.get("ema200", 0) or 0)
    if ema200 <= 0:
        return None
    bias = "LONG" if close > ema200 else "SHORT"

    # ── 4. EMA9/21 confirmation ───────────────────────────────────────────────
    ema9  = float(cur.get("ema9",  0) or 0)
    ema21 = float(cur.get("ema21", 0) or 0)
    if bias == "LONG"  and ema9 < ema21:
        return None
    if bias == "SHORT" and ema9 > ema21:
        return None

    # ── 5. RSI momentum ───────────────────────────────────────────────────────
    rsi = float(cur.get("rsi", 50) or 50)
    if bias == "LONG"  and rsi < float(p["rsi_long"]):
        return None
    if bias == "SHORT" and rsi > float(p["rsi_short"]):
        return None

    # ── 6. Volume absorption ─────────────────────────────────────────────────
    vol_ratio = float(cur.get("vol_ratio", 0) or 0)

    if dom_signal and dom_signal.get("type") == "ABSORPTION":
        # Signal DOM live : on valide directement si le côté correspond
        dom_side = dom_signal.get("side", "")
        if (bias == "LONG"  and dom_side != "BUY"):
            return None
        if (bias == "SHORT" and dom_side != "SELL"):
            return None
    else:
        # Mode backtest : proxy volume
        if vol_ratio < float(p["vol_multiplier"]):
            return None

        # Qualité de la bougie : close dans la bonne moitié du range
        if bar_range < 0.001:
            return None
        close_pct = (close - bar_low) / bar_range
        if bias == "LONG"  and close_pct < float(p["close_pct_long"]):
            return None
        if bias == "SHORT" and close_pct > float(p["close_pct_short"]):
            return None

    # ── 7. Calcul SL / TP ────────────────────────────────────────────────────
    sl_pts  = int(p["sl_ticks"])  * TICK_SIZE
    tp1_pts = int(p["tp1_ticks"]) * TICK_SIZE
    tp2_pts = int(p["tp2_ticks"]) * TICK_SIZE

    if bias == "LONG":
        sl  = close - sl_pts
        tp1 = close + tp1_pts
        tp2 = close + tp2_pts
    else:
        sl  = close + sl_pts
        tp1 = close - tp1_pts
        tp2 = close - tp2_pts

    return {
        "bias":         bias,
        "entry":        round(close,     2),
        "stop_loss":    round(sl,        2),
        "take_profit1": round(tp1,       2),
        "take_profit2": round(tp2,       2),
        "atr":          round(atr,       2),
        "rsi":          round(rsi,       1),
        "vol_ratio":    round(vol_ratio, 2),
        "signal":       "dom_absorption" if dom_signal else "volume_absorption",
    }


# ── Taille de position ────────────────────────────────────────────────────────

def size_contracts(
    equity:    float,
    risk_pct:  float,
    sl_ticks:  int,
) -> int:
    """Nombre de contrats ES pour risquer risk_pct% de l'equity."""
    risk_dollar = equity * risk_pct / 100.0
    sl_dollar   = sl_ticks * TICK_VALUE
    if sl_dollar <= 0:
        return 1
    n = math.floor(risk_dollar / sl_dollar)
    return max(1, n)
