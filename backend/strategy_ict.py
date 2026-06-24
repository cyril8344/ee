"""
strategy_ict.py — Stratégie B : Order Block M5
===============================================
Pipeline :
1. Biais H1 (EMA50 vs EMA200)
2. Détection OBs M5 récents alignés avec le biais (non mitiguées)
3. Entrée si prix en retest du dernier OB valide
4. SL = derrière l'OB + buffer
5. TP1 = 0.7R, TP2 = 1.8R

OB baissier (LONG) : dernière bougie rouge avant impulse haussier ≥ 1.5×ATR
OB haussier (SHORT) : dernière bougie verte avant impulse baissier ≥ 1.5×ATR
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional, Dict, Any, List

import pandas as pd

from strategy import Signal, active_session, is_bad_timing, ATR_MIN

# ──────────────────────────────────────────────────────────────────────────────
# Paramètres
# ──────────────────────────────────────────────────────────────────────────────
OB_IMPULSE_ATR    = 1.5  # impulse minimum après l'OB pour le qualifier (en ATR)
OB_MAX_BARS       = 50   # cherche les OBs dans les 50 dernières bougies M5 (~4h)
OB_MIN_BODY_ATR   = 0.2  # corps minimum de la bougie OB (filtre dojis)
OB_MAX_HEIGHT_ATR = 1.5  # hauteur maximale de l'OB — OBs trop larges = R:R défavorable
OB_ENTRY_BUFFER   = 0.1  # entre à 0.1×ATR depuis l'extrême de l'OB (ordre limite)
SL_BUFFER_ATR     = 0.3  # buffer SL derrière l'extrême de l'OB
TP1_R             = 0.7
TP2_R             = 1.8


# ──────────────────────────────────────────────────────────────────────────────
# 1. Biais H1
# ──────────────────────────────────────────────────────────────────────────────
def _h1_bias(h1: pd.DataFrame) -> Optional[str]:
    """'LONG', 'SHORT' ou None (neutre — EMA50/EMA200 trop proches)."""
    if len(h1) == 0:
        return None
    last = h1.iloc[-1]
    ema50  = last.get("ema50",  None)
    ema200 = last.get("ema200", None)
    if ema50 is None or ema200 is None or pd.isna(ema50) or pd.isna(ema200):
        return None
    if float(ema50) > float(ema200):
        return "LONG"
    if float(ema50) < float(ema200):
        return "SHORT"
    return None


# ──────────────────────────────────────────────────────────────────────────────
# 2. Détection des Order Blocks M5
# ──────────────────────────────────────────────────────────────────────────────
def _find_order_blocks(
    df: pd.DataFrame,
    direction: str,
    atr_val: float,
) -> List[Dict[str, Any]]:
    """
    Retourne les OBs valides (non mitiguées) dans les OB_MAX_BARS dernières bougies.

    LONG  : OB = dernière bougie rouge (close < open) avant impulse haussier ≥ OB_IMPULSE_ATR×ATR
    SHORT : OB = dernière bougie verte (close > open) avant impulse baissier ≥ OB_IMPULSE_ATR×ATR

    Zone OB = [low, high] de cette bougie.
    Mitigation : exclut l'OB si le prix y est déjà repassé à travers après sa formation.
    """
    recent = df.tail(OB_MAX_BARS + 5)
    n = len(recent)
    if n < 5:
        return []

    min_impulse = OB_IMPULSE_ATR * atr_val
    min_body    = OB_MIN_BODY_ATR * atr_val
    obs: List[Dict[str, Any]] = []

    for i in range(n - 4):
        bar    = recent.iloc[i]
        b_open  = float(bar["open"])
        b_close = float(bar["close"])
        b_high  = float(bar["high"])
        b_low   = float(bar["low"])

        ob_height = b_high - b_low
        if ob_height > OB_MAX_HEIGHT_ATR * atr_val:
            continue  # OB trop large → R:R défavorable
        if abs(b_close - b_open) < min_body:
            continue  # doji → pas un OB valide

        after = recent.iloc[i + 1: i + 4]

        if direction == "LONG":
            if b_close >= b_open:
                continue  # doit être baissière
            impulse = float(after["high"].max()) - b_high
            if impulse < min_impulse:
                continue
        else:  # SHORT
            if b_close <= b_open:
                continue  # doit être haussière
            impulse = b_low - float(after["low"].min())
            if impulse < min_impulse:
                continue

        # Vérifier que l'OB n'est pas encore mitiguée
        post = recent.iloc[i + 1:]
        if direction == "LONG" and float(post["low"].min()) < b_low:
            continue  # prix passé sous l'OB → mitiguée
        if direction == "SHORT" and float(post["high"].max()) > b_high:
            continue  # prix passé au-dessus de l'OB → mitiguée

        obs.append({
            "low":  b_low,
            "high": b_high,
            "ts":   recent.index[i],
        })

    return obs


def _in_ob(bar_low: float, bar_high: float, ob: Dict) -> bool:
    """True si la bougie actuelle touche l'OB (wick ou corps)."""
    return bar_low <= ob["high"] and bar_high >= ob["low"]


# ──────────────────────────────────────────────────────────────────────────────
# 3. Évaluation principale
# ──────────────────────────────────────────────────────────────────────────────
def evaluate_ict(
    m5: pd.DataFrame,
    m15: pd.DataFrame,
    h1: pd.DataFrame,
    now: Optional[datetime] = None,
    check_session: bool = True,
    atr_min: float = ATR_MIN,
) -> Optional[Signal]:
    """Stratégie B — Order Block M5 avec biais H1."""
    if len(m5) < 50:
        return None

    cur = m5.iloc[-1]
    ts  = now or cur.name
    if isinstance(ts, pd.Timestamp):
        ts = ts.to_pydatetime()
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)

    # 1) Timing / session
    if is_bad_timing(ts):
        return None
    session = active_session(ts)
    if check_session and session is None:
        return None
    session = session or "London"

    # 2) ATR plancher
    atr_val = float(cur.get("atr", 0) or 0)
    if atr_val < atr_min:
        return None

    # 3) Biais H1
    direction = _h1_bias(h1)
    if direction is None:
        return None

    # 4) Order Blocks M5 récents non mitiguées
    obs = _find_order_blocks(m5, direction, atr_val)
    if not obs:
        return None

    # Dernier OB valide (le plus récent)
    ob = obs[-1]

    # 5) Prix actuel en retest de l'OB
    if not _in_ob(float(cur["low"]), float(cur["high"]), ob):
        return None

    # 6) Niveaux du trade — entrée ordre limite à l'extrême de l'OB
    # LONG : entrée au bas de l'OB (ob_low + buffer) → SL juste en dessous
    # SHORT : entrée au haut de l'OB (ob_high - buffer) → SL juste au-dessus
    # Risque fixe ~0.4×ATR (OB_ENTRY_BUFFER + SL_BUFFER_ATR)
    if direction == "LONG":
        entry = ob["low"] + OB_ENTRY_BUFFER * atr_val
        if float(cur["low"]) > entry:
            return None  # barre n'a pas touché le bas de l'OB → pas de fill
        sl   = ob["low"] - SL_BUFFER_ATR * atr_val
        risk = abs(entry - sl)
        if risk <= 0:
            return None
        tp1 = entry + TP1_R * risk
        tp2 = entry + TP2_R * risk
    else:
        entry = ob["high"] - OB_ENTRY_BUFFER * atr_val
        if float(cur["high"]) < entry:
            return None  # barre n'a pas touché le haut de l'OB → pas de fill
        sl   = ob["high"] + SL_BUFFER_ATR * atr_val
        risk = abs(entry - sl)
        if risk <= 0:
            return None
        tp1 = entry - TP1_R * risk
        tp2 = entry - TP2_R * risk

    ob_ts = ob["ts"]
    ob_ts_str = ob_ts.isoformat() if hasattr(ob_ts, "isoformat") else str(ob_ts)

    return Signal(
        direction=direction.lower(),
        bias=direction,
        session=session,
        entry=entry,
        stop_loss=sl,
        take_profit1=tp1,
        take_profit2=tp2,
        atr=atr_val,
        reason="OB_RETEST",
        risk_distance=risk,
        timestamp=ts,
        meta={
            "strategy": "B_OB",
            "ob_low":   round(ob["low"],  5),
            "ob_high":  round(ob["high"], 5),
            "ob_ts":    ob_ts_str,
        },
    )
