"""
strategy_ict.py — Stratégie B : Order Block M5 pur
====================================================
Pipeline minimal :
1. Session gate (London 8-12h / NY 14-18h CET)
2. ATR M5 ≥ seuil (volatilité minimale)
3. Détection OB M5 : dernière bougie contrariante avant impulse ≥ 1.0×ATR
4. Mitigation par close (OB invalide si un close a traversé la zone)
5. Retest : prix actuel entre dans la zone OB
6. SL derrière l'OB + buffer, TP1=0.7R, TP2=1.8R

Aucun filtre de tendance (pas de H1 bias, pas d'ADX).
La direction est déterminée par l'impulse M5 elle-même.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional, Dict, Any, List

import pandas as pd

from strategy import Signal, active_session, is_bad_timing, ATR_MIN

# ──────────────────────────────────────────────────────────────────────────────
# Paramètres
# ──────────────────────────────────────────────────────────────────────────────
OB_IMPULSE_ATR  = 1.0   # impulse minimum après l'OB (en ATR)
OB_MAX_BARS     = 30    # fenêtre de recherche OBs (30 M5 ≈ 2.5h)
OB_MIN_BODY_ATR = 0.1   # corps minimum bougie OB (filtre dojis)
SL_BUFFER_ATR   = 0.2   # buffer SL derrière l'extrême de l'OB
MAX_RISK_ATR    = 1.5   # plafond risque (SL ≤ 1.5×ATR de l'entrée)
TP1_R           = 0.7
TP2_R           = 1.8


# ──────────────────────────────────────────────────────────────────────────────
# Détection des Order Blocks M5
# ──────────────────────────────────────────────────────────────────────────────
def _find_order_blocks(
    df: pd.DataFrame,
    direction: str,
    atr_val: float,
) -> List[Dict[str, Any]]:
    """
    Retourne les OBs valides dans les OB_MAX_BARS dernières bougies M5.

    Critères :
    1. Bougie contrariante : rouge (LONG) / verte (SHORT)
    2. Corps ≥ OB_MIN_BODY_ATR×ATR (filtre dojis)
    3. Impulse ≥ OB_IMPULSE_ATR×ATR dans les 3 bougies suivantes
    4. Non mitiguée : aucun close n'a traversé la zone OB depuis sa formation
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

        # Corps minimum
        if abs(b_close - b_open) < min_body:
            continue

        after = recent.iloc[i + 1: i + 4]

        if direction == "LONG":
            # OB = bougie rouge avant impulse haussier
            if b_close >= b_open:
                continue
            if float(after["high"].max()) - b_high < min_impulse:
                continue
        else:
            # OB = bougie verte avant impulse baissier
            if b_close <= b_open:
                continue
            if b_low - float(after["low"].min()) < min_impulse:
                continue

        # Mitigation par close (pas par mèche)
        post_closes = recent.iloc[i + 1:]["close"]
        if direction == "LONG" and float(post_closes.min()) < b_low:
            continue
        if direction == "SHORT" and float(post_closes.max()) > b_high:
            continue

        obs.append({"low": b_low, "high": b_high, "ts": recent.index[i]})

    return obs


def _in_ob(bar_low: float, bar_high: float, ob: Dict) -> bool:
    """True si la bougie touche la zone OB."""
    return bar_low <= ob["high"] and bar_high >= ob["low"]


# ──────────────────────────────────────────────────────────────────────────────
# Évaluation principale
# ──────────────────────────────────────────────────────────────────────────────
def evaluate_ict(
    m5: pd.DataFrame,
    m15: pd.DataFrame,
    h1: pd.DataFrame,
    now: Optional[datetime] = None,
    check_session: bool = True,
    atr_min: float = ATR_MIN,
) -> Optional[Signal]:
    """Stratégie B — Order Block M5 pur, sans filtre de tendance."""
    if len(m5) < 10:
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

    # 2) ATR minimum
    atr_val = float(cur.get("atr", 0) or 0)
    if atr_val < atr_min:
        return None

    entry = float(cur["close"])

    # 3) Chercher OB LONG et SHORT, prendre le plus récent en retest
    best_ob   = None
    direction = None

    for d in ("LONG", "SHORT"):
        obs = _find_order_blocks(m5, d, atr_val)
        for ob in reversed(obs):  # plus récent en premier
            if _in_ob(float(cur["low"]), float(cur["high"]), ob):
                if best_ob is None or ob["ts"] > best_ob["ts"]:
                    best_ob   = ob
                    direction = d
                break

    if best_ob is None or direction is None:
        return None

    # 4) Niveaux du trade
    ob = best_ob
    if direction == "LONG":
        raw_sl = ob["low"] - SL_BUFFER_ATR * atr_val
        sl     = max(raw_sl, entry - MAX_RISK_ATR * atr_val)
    else:
        raw_sl = ob["high"] + SL_BUFFER_ATR * atr_val
        sl     = min(raw_sl, entry + MAX_RISK_ATR * atr_val)

    risk = abs(entry - sl)
    if risk <= 0:
        return None

    if direction == "LONG":
        tp1 = entry + TP1_R * risk
        tp2 = entry + TP2_R * risk
    else:
        tp1 = entry - TP1_R * risk
        tp2 = entry - TP2_R * risk

    ob_ts_str = ob["ts"].isoformat() if hasattr(ob["ts"], "isoformat") else str(ob["ts"])

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
            "ob_low":  round(ob["low"],  5),
            "ob_high": round(ob["high"], 5),
            "ob_ts":   ob_ts_str,
        },
    )
