"""
strategy_es_h1.py
==================
Variante H1 (bougies 1 heure) de la stratégie Order Flow ES (E-mini S&P 500).

Diffère de strategy_es.py (M5) sur deux points structurels :
  - SL/TP en multiples d'ATR / R plutôt qu'en ticks fixes — les ticks calibrés
    pour M5 (14/20/40 = 3.5/5/10 pts) seraient hors d'échelle sur une bougie
    H1, dont l'amplitude normale dépasse déjà souvent 10 pts.
  - Aucune position ne doit rester ouverte après la clôture de session RTH
    (16h ET) : une entrée H1 peut mettre plusieurs heures à atteindre son
    objectif, donc on refuse toute nouvelle entrée trop proche de la clôture
    (flatten_before_close_h1) et on force la sortie au marché à la clôture
    (voir pretrain_es_h1._try_exit_es_h1). Objectif : garder le principe
    "jamais de position hors session" déjà appliqué à XAU/EUR, et éviter
    l'exposition aux gaps overnight/weekend propres aux indices actions.

Filtres (ordre strict) :
  1.  Session gate (RTH : 9h30–16h00 ET)
  2.  Heures bloquées ET
  3.  Fenêtre de clôture (pas de nouvelle entrée dans les N dernières heures RTH)
  4.  ATR minimum (volatilité)
  5.  Biais EMA200 H1
  6.  EMA9/21 confirmation H1
  7.  RSI H1 momentum
  8.  ADX minimum (force de tendance)
  9.  VWAP alignment (intraday, reset ET)
 10.  Calcul SL (ATR × sl_atr_mult) / TP1 (R × tp1_r) / TP2 (R × tp2_r)

Paramètres optimisables via pretrain_es_h1.py / walk-forward.
"""

from __future__ import annotations
from typing import Optional

import pandas as pd

from strategy_es import add_indicators, size_contracts, TICK_SIZE, TICK_VALUE

# ── Paramètres par défaut ─────────────────────────────────────────────────────
DEFAULTS: dict = {
    # Filtres tendance H1
    "ema_fast":         9,
    "ema_slow":         21,
    "ema_trend":        200,

    # RSI H1 momentum (désactivé = 30/70 large, à resserrer via walk-forward)
    "rsi_long":         30,
    "rsi_short":        70,

    # Volatilité (0 = désactivé)
    "atr_min_pts":      0,

    # ADX (0 = désactivé)
    "adx_min":          0,

    # VWAP filter (0 = désactivé)
    "vwap_filter":      0,

    # Heures bloquées (vide = aucune)
    "bad_hours_et":     [],

    # SL / TP en multiples d'ATR et de R (remplace les ticks fixes de la
    # version M5 — inadaptés à l'amplitude d'une bougie H1)
    "sl_atr_mult": 1.2,
    "tp1_r":       0.8,   # TP1 déclenche la sortie de 50% + SL -> BE
    "tp2_r":       1.8,

    # Pas de nouvelle entrée dans les N dernières heures avant la clôture RTH
    # (une position ouverte trop tard n'a pas le temps de se développer avant
    # d'être flattenée de force)
    "flatten_before_close_h1": 1,

    # Durée max d'un trade en bougies H1 (= heures, appliqué par le simulateur
    # de pretrain ; la clôture de session prime de toute façon sur ce délai)
    "max_trade_hours": 8,

    # Session ET (Eastern Time)
    "session_open_h":  9,
    "session_open_m":  30,
    "session_close_h": 16,
    "session_close_m": 0,
}


def _round_tick(price: float) -> float:
    return round(round(price / TICK_SIZE) * TICK_SIZE, 2)


def _decimal_hour(ts_et) -> float:
    return ts_et.hour + ts_et.minute / 60.0


def evaluate(
    bars_h1: pd.DataFrame,
    params:  Optional[dict] = None,
    ts=None,
) -> Optional[dict]:
    """
    Évalue la bougie H1 courante (dernière ligne de bars_h1) et retourne un
    signal ou None. `bars_h1` doit déjà contenir les indicateurs
    (voir strategy_es.add_indicators, réutilisé tel quel — le calcul est
    indépendant du timeframe).
    """
    p = {**DEFAULTS, **(params or {})}

    if len(bars_h1) < int(p["ema_trend"]) + 10:
        return None

    cur   = bars_h1.iloc[-1]
    close = float(cur.get("close", 0) or 0)

    # ── 1. Session gate (Eastern Time) ───────────────────────────────────────
    ts_et = None
    if ts is not None:
        try:
            import pytz
            et_tz    = pytz.timezone("America/New_York")
            ts_et    = ts.astimezone(et_tz)
            dec_hour = _decimal_hour(ts_et)
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

    # ── 3. Fenêtre de clôture — pas de nouvelle entrée trop tard ─────────────
    if ts_et is not None:
        dec_hour  = _decimal_hour(ts_et)
        cls_dec   = p["session_close_h"] + p["session_close_m"] / 60.0
        flatten_h = float(p.get("flatten_before_close_h1", 0))
        if dec_hour >= cls_dec - flatten_h:
            return None

    # ── 4. ATR minimum ───────────────────────────────────────────────────────
    atr = float(cur.get("atr", 0) or 0)
    if atr <= 0 or atr < float(p["atr_min_pts"]):
        return None

    # ── 5. Biais EMA200 ───────────────────────────────────────────────────────
    ema200 = float(cur.get("ema200", 0) or 0)
    if ema200 <= 0:
        return None
    bias = "LONG" if close > ema200 else "SHORT"

    # ── 6. EMA9/21 confirmation ──────────────────────────────────────────────
    ema9  = float(cur.get("ema9",  0) or 0)
    ema21 = float(cur.get("ema21", 0) or 0)
    if bias == "LONG"  and ema9 < ema21:
        return None
    if bias == "SHORT" and ema9 > ema21:
        return None

    # ── 7. RSI momentum ──────────────────────────────────────────────────────
    rsi = float(cur.get("rsi", 50) or 50)
    if bias == "LONG"  and rsi < float(p["rsi_long"]):
        return None
    if bias == "SHORT" and rsi > float(p["rsi_short"]):
        return None

    # ── 8. ADX minimum ───────────────────────────────────────────────────────
    adx_min = float(p.get("adx_min", 0))
    if adx_min > 0:
        adx_val = float(cur.get("adx", 0) or 0)
        if adx_val < adx_min:
            return None

    # ── 9. VWAP alignment ────────────────────────────────────────────────────
    if bool(p.get("vwap_filter", 0)):
        vwap_val = float(cur.get("vwap", float("nan")) or float("nan"))
        if vwap_val == vwap_val and vwap_val > 0:
            if bias == "LONG"  and close < vwap_val:
                return None
            if bias == "SHORT" and close > vwap_val:
                return None

    # ── 10. SL / TP en ATR et R ──────────────────────────────────────────────
    sl_dist = atr * float(p["sl_atr_mult"])
    if sl_dist <= 0:
        return None

    if bias == "LONG":
        sl  = close - sl_dist
        tp1 = close + sl_dist * float(p["tp1_r"])
        tp2 = close + sl_dist * float(p["tp2_r"])
    else:
        sl  = close + sl_dist
        tp1 = close - sl_dist * float(p["tp1_r"])
        tp2 = close - sl_dist * float(p["tp2_r"])

    return {
        "bias":         bias,
        "entry":        _round_tick(close),
        "stop_loss":    _round_tick(sl),
        "take_profit1": _round_tick(tp1),
        "take_profit2": _round_tick(tp2),
        "atr":          round(atr, 2),
        "rsi":          round(rsi, 1),
        "sl_pts":       round(sl_dist, 2),
        "signal":       "h1_trend",
    }
