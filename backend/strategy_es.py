"""
strategy_es.py
==============
Stratégie Order Flow pour ES (E-mini S&P 500).

Filtres (ordre strict) :
  1.  Session gate (RTH : 9h30–16h00 ET)
  2.  Heures bloquées ET (10h par défaut)
  3.  ATR minimum (volatilité)
  4.  Biais EMA200 — M5 ou H1 si fourni
  5.  EMA9/21 confirmation M5
  6.  RSI M5 momentum
  7.  ADX minimum (force de tendance, évite le range)
  8.  VWAP alignment (intraday)
  9.  Volume absorption proxy (ou signal DOM NinjaTrader live)
      + Qualité bougie (body/ATR, close dans bon tiers du range)
 10.  Calcul SL / TP en ticks

En live avec DOM NinjaTrader connecté via /api/es/dom :
le signal volume proxy est remplacé par la détection d'absorption réelle.

Paramètres optimisables via pretrain_es.py / walk-forward.
"""

from __future__ import annotations
import math
import pandas as pd
import numpy as np
from typing import Optional

# ── Constantes contrat ES ─────────────────────────────────────────────────────
TICK_SIZE   = 0.25   # 1 tick = 0.25 points
TICK_VALUE  = 12.50  # 1 tick = 12.50 $ par contrat
POINT_VALUE = 50.0   # 1 point = 50 $ par contrat

# ── Paramètres par défaut ─────────────────────────────────────────────────────
DEFAULTS: dict = {
    # Filtres tendance M5
    "ema_fast":         9,
    "ema_slow":         21,
    "ema_trend":        200,

    # RSI M5 momentum (plus sélectif)
    "rsi_long":         48,    # LONG : RSI > 48
    "rsi_short":        52,    # SHORT : RSI < 52

    # Volatilité
    "atr_min_pts":      2.0,   # ATR minimum en points ES

    # ADX (force de tendance — filtre les ranges plats)
    "adx_min":          20,

    # VWAP filter (1=activé, 0=désactivé)
    "vwap_filter":      1,

    # H1 bias via EMA200 H1 (1=activé, 0=désactivé)
    "h1_filter":        1,

    # Heures bloquées (heure ET, liste d'entiers)
    "bad_hours_et":     [10],  # 10h ET = choppiness post-ouverture

    # Volume absorption proxy
    "vol_multiplier":   2.8,   # ratio volume / moyenne pour déclencher
    "vol_lookback":     20,    # barres pour calculer la moyenne

    # Qualité bougie d'entrée
    "close_pct_long":   0.70,  # close ≥ 70% du range pour LONG
    "close_pct_short":  0.30,  # close ≤ 30% du range pour SHORT
    "body_ratio_min":   0.20,  # |close-open| / ATR minimum

    # SL / TP en ticks
    "sl_ticks":   14,   # 3.5 pts = $175/contrat
    "tp1_ticks":  20,   # 5.0 pts = $250/contrat
    "tp2_ticks":  40,   # 10.0 pts = $500/contrat

    # Session ET (Eastern Time)
    "session_open_h":  9,
    "session_open_m":  30,
    "session_close_h": 16,
    "session_close_m": 0,
}


# ── Indicateurs ───────────────────────────────────────────────────────────────

def add_indicators(df: pd.DataFrame, params: Optional[dict] = None) -> pd.DataFrame:
    """
    Calcule EMA, RSI, ATR, ADX, VWAP, volume ratio sur un DataFrame OHLCV.
    Compatible M5 et H1.
    """
    p  = {**DEFAULTS, **(params or {})}
    df = df.copy()

    # ── EMA ──────────────────────────────────────────────────────────────────
    df["ema9"]   = df["close"].ewm(span=int(p["ema_fast"]),  adjust=False).mean()
    df["ema21"]  = df["close"].ewm(span=int(p["ema_slow"]),  adjust=False).mean()
    df["ema200"] = df["close"].ewm(span=int(p["ema_trend"]), adjust=False).mean()

    # ── RSI 14 ───────────────────────────────────────────────────────────────
    delta    = df["close"].diff()
    gain     = delta.clip(lower=0)
    loss     = (-delta).clip(lower=0)
    avg_gain = gain.ewm(span=14, adjust=False).mean()
    avg_loss = loss.ewm(span=14, adjust=False).mean()
    rs        = avg_gain / avg_loss.replace(0, float("nan"))
    df["rsi"] = 100.0 - 100.0 / (1.0 + rs)

    # ── ATR 14 ───────────────────────────────────────────────────────────────
    hl  = df["high"] - df["low"]
    hc  = (df["high"] - df["close"].shift(1)).abs()
    lc  = (df["low"]  - df["close"].shift(1)).abs()
    tr  = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    df["atr"] = tr.ewm(span=14, adjust=False).mean()

    # ── ADX 14 ───────────────────────────────────────────────────────────────
    h_diff   = df["high"].diff()
    l_diff   = df["low"].diff()
    dm_plus  = h_diff.clip(lower=0).where(h_diff > -l_diff,  0.0)
    dm_minus = (-l_diff).clip(lower=0).where(-l_diff > h_diff, 0.0)
    atr_s    = df["atr"].replace(0, float("nan"))
    di_plus  = 100 * dm_plus.ewm(span=14,  adjust=False).mean() / atr_s
    di_minus = 100 * dm_minus.ewm(span=14, adjust=False).mean() / atr_s
    denom    = (di_plus + di_minus).replace(0, float("nan"))
    dx       = 100 * (di_plus - di_minus).abs() / denom
    df["adx"] = dx.ewm(span=14, adjust=False).mean()

    # ── VWAP intraday (reset quotidien) ──────────────────────────────────────
    typical = (df["high"] + df["low"] + df["close"]) / 3
    if hasattr(df.index, "normalize"):
        day_key = df.index.normalize()
    else:
        day_key = pd.DatetimeIndex([d.date() for d in df.index])
    cum_tp_vol = (typical * df["volume"]).groupby(day_key).cumsum()
    cum_vol    = df["volume"].groupby(day_key).cumsum().replace(0, float("nan"))
    df["vwap"] = cum_tp_vol / cum_vol
    # Fallback pour données sans volume (synthétique) → typical price
    zero_mask = df["volume"] == 0
    if zero_mask.any():
        df.loc[zero_mask, "vwap"] = typical[zero_mask]

    # ── Volume absorption proxy ───────────────────────────────────────────────
    lookback       = max(int(p["vol_lookback"]), 3)
    df["vol_avg"]  = df["volume"].rolling(lookback, min_periods=3).mean()
    df["vol_ratio"] = df["volume"] / df["vol_avg"].replace(0, float("nan"))

    return df


# ── Signal principal ──────────────────────────────────────────────────────────

def evaluate(
    bars_5m:    pd.DataFrame,
    params:     Optional[dict] = None,
    ts=None,
    dom_signal: Optional[dict] = None,   # signal DOM live NinjaTrader
    h1:         Optional[pd.DataFrame] = None,  # H1 pré-calculé (facultatif)
) -> Optional[dict]:
    """
    Évalue la bougie courante (dernière ligne de bars_5m) et retourne un signal ou None.

    dom_signal : {"side": "BUY"|"SELL", "size": 93, "price": 5839.5, "type": "ABSORPTION"}
    h1         : DataFrame H1 avec indicateurs. Si fourni + h1_filter=1, utilise H1
                 EMA200 pour le biais et H1 ADX pour le filtre de tendance.
    """
    p = {**DEFAULTS, **(params or {})}

    if len(bars_5m) < int(p["ema_trend"]) + 10:
        return None

    cur       = bars_5m.iloc[-1]
    close     = float(cur.get("close", 0) or 0)
    bar_open  = float(cur.get("open",  0) or 0)
    bar_high  = float(cur.get("high",  0) or 0)
    bar_low   = float(cur.get("low",   0) or 0)
    bar_range = bar_high - bar_low

    # ── 1. Session gate (Eastern Time) ───────────────────────────────────────
    ts_et = None
    if ts is not None:
        try:
            import pytz
            et_tz    = pytz.timezone("America/New_York")
            ts_et    = ts.astimezone(et_tz)
            dec_hour = ts_et.hour + ts_et.minute / 60.0
            open_dec = p["session_open_h"]  + p["session_open_m"]  / 60.0
            cls_dec  = p["session_close_h"] + p["session_close_m"] / 60.0
            if not (open_dec <= dec_hour < cls_dec):
                return None
        except Exception:
            pass

    # ── 2. Heures bloquées ET ────────────────────────────────────────────────
    if ts_et is not None:
        bad = p.get("bad_hours_et", [])
        if not isinstance(bad, (list, set, tuple)):
            try:
                bad = [int(x) for x in str(bad).split(",") if x.strip()]
            except Exception:
                bad = []
        if ts_et.hour in [int(h) for h in bad]:
            return None

    # ── 3. ATR minimum ───────────────────────────────────────────────────────
    atr = float(cur.get("atr", 0) or 0)
    if atr < float(p["atr_min_pts"]):
        return None

    # ── 4. Biais EMA200 (H1 prioritaire si h1_filter activé) ─────────────────
    use_h1   = bool(p.get("h1_filter", 1)) and h1 is not None and len(h1) >= 1
    h1_cur   = h1.iloc[-1] if use_h1 else None
    ema200_ref = float(h1_cur.get("ema200", 0) or 0) if use_h1 else float(cur.get("ema200", 0) or 0)

    if ema200_ref <= 0:
        return None
    bias = "LONG" if close > ema200_ref else "SHORT"

    # ── 5. EMA9/21 confirmation M5 ───────────────────────────────────────────
    ema9  = float(cur.get("ema9",  0) or 0)
    ema21 = float(cur.get("ema21", 0) or 0)
    if bias == "LONG"  and ema9 < ema21:
        return None
    if bias == "SHORT" and ema9 > ema21:
        return None

    # ── 6. RSI M5 momentum ───────────────────────────────────────────────────
    rsi = float(cur.get("rsi", 50) or 50)
    if bias == "LONG"  and rsi < float(p["rsi_long"]):
        return None
    if bias == "SHORT" and rsi > float(p["rsi_short"]):
        return None

    # ── 7. ADX minimum ───────────────────────────────────────────────────────
    adx_min = float(p.get("adx_min", 20))
    if adx_min > 0:
        adx_val = float(h1_cur.get("adx", 0) or 0) if use_h1 else float(cur.get("adx", 0) or 0)
        if 0 < adx_val < adx_min:
            return None

    # ── 8. VWAP alignment ────────────────────────────────────────────────────
    if bool(p.get("vwap_filter", 1)):
        vwap_val = float(cur.get("vwap", float("nan")) or float("nan"))
        if vwap_val == vwap_val and vwap_val > 0:   # not NaN
            if bias == "LONG"  and close < vwap_val:
                return None
            if bias == "SHORT" and close > vwap_val:
                return None

    # ── 9. Volume absorption ─────────────────────────────────────────────────
    vol_ratio = float(cur.get("vol_ratio", 0) or 0)

    if dom_signal and dom_signal.get("type") == "ABSORPTION":
        dom_side = dom_signal.get("side", "")
        if bias == "LONG"  and dom_side != "BUY":
            return None
        if bias == "SHORT" and dom_side != "SELL":
            return None
    else:
        # Proxy volume backtest
        if vol_ratio < float(p["vol_multiplier"]):
            return None

        if bar_range < 0.001:
            return None
        close_pct = (close - bar_low) / bar_range
        if bias == "LONG"  and close_pct < float(p["close_pct_long"]):
            return None
        if bias == "SHORT" and close_pct > float(p["close_pct_short"]):
            return None

        # Corps de la bougie / ATR (bougie indécise → rejeter)
        body_min = float(p.get("body_ratio_min", 0.20))
        if body_min > 0 and atr > 0:
            body = abs(close - bar_open)
            if body / atr < body_min:
                return None

    # ── 10. Calcul SL / TP ───────────────────────────────────────────────────
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
