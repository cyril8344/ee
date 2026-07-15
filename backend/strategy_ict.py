"""
strategy_ict.py — Stratégie B : Order Block M5
===============================================
Pipeline :
1. Biais H1 (EMA50 vs EMA200)
2. ADX H1 ≥ 20 (rejet marchés sans tendance)
3. Détection OBs M5 valides :
   a. Bougie contrariante + impulse ≥ 1.5×ATR dans les 3 bougies suivantes
   b. Corps min (filtre dojis) + hauteur max (filtre OBs larges)
   c. Non mitiguée (le prix n'a pas traversé l'OB depuis sa formation)
4. Entrée en retest de la zone OB [ob_low, ob_high]
5. SL = derrière l'OB + buffer, plafonné à MAX_RISK_ATR×ATR
6. TP1 = 0.7R, TP2 = 1.8R

OB LONG  : dernière bougie rouge avant impulse haussier ≥ 1.5×ATR
OB SHORT : dernière bougie verte avant impulse baissier ≥ 1.5×ATR
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional, Dict, Any, List

import pandas as pd

from strategy import (Signal, active_session, is_bad_timing, ATR_MIN,
                      nearest_support_below, nearest_resistance_above, CET)

# ──────────────────────────────────────────────────────────────────────────────
# Paramètres
# ──────────────────────────────────────────────────────────────────────────────
OB_IMPULSE_ATR    = 1.0  # impulse minimum après l'OB (en ATR) — 1.5 trop strict (0 OB détecté)
OB_MAX_BARS       = 50   # fenêtre de recherche OBs (50 M5 ≈ 4h)
OB_MIN_BODY_ATR   = 0.2  # corps minimum bougie OB (filtre dojis)
OB_MAX_HEIGHT_ATR = 1.5  # hauteur maximale OB (OBs larges → R:R défavorable)
ADX_MIN_H1        = 20   # ADX H1 minimum — EUR/USD typique 18-26, 28 trop strict
RSI_LONG_MIN      = 46   # RSI M5 minimum pour LONG (identique strategy A, validé Optuna)
RSI_SHORT_MAX     = 57   # RSI M5 maximum pour SHORT
RSI_M15_LONG_MIN  = 45   # RSI M15 minimum pour LONG (momentum M15 dans le bon sens)
RSI_M15_SHORT_MAX = 55   # RSI M15 maximum pour SHORT (Δ +4.9 discriminant)
RSI_H1_LONG_MIN   = 46   # RSI H1 minimum pour LONG (momentum H1 dans le bon sens)
RSI_H1_SHORT_MAX  = 54   # RSI H1 maximum pour SHORT (Δ +5.5 discriminant)
SL_BUFFER_ATR     = 0.7  # buffer SL derrière l'extrême de l'OB (0.3 → 0.7 : 62% faux stops couverts)
MAX_RISK_ATR      = 2.0  # plafond risque élargi pour accommoder le buffer SL augmenté
TP1_R             = 0.7
TP2_R             = 1.0

# S/R zone-to-zone (H1)
SR_ZONE_ATR_H1  = 1.0   # prix dans 1.0×ATR H1 d'un niveau → zone active
SR_MIN_TOUCHES  = 2     # touches minimales pour valider un niveau H1
SR_TP_MIN_R     = 1.0   # cible S/R doit être ≥ 1R pour remplacer TP2 fixe


# ──────────────────────────────────────────────────────────────────────────────
# 1. Biais H1
# ──────────────────────────────────────────────────────────────────────────────
def _h1_bias(h1: pd.DataFrame) -> Optional[str]:
    """'LONG', 'SHORT' ou None (neutre)."""
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
    Retourne les OBs valides dans les OB_MAX_BARS dernières bougies M5.

    Critères (tous requis) :
    1. Bougie contrariante : rouge (LONG) / verte (SHORT)
    2. Impulse ≥ OB_IMPULSE_ATR×ATR dans les 3 bougies suivantes
    3. Corps ≥ OB_MIN_BODY_ATR×ATR | hauteur ≤ OB_MAX_HEIGHT_ATR×ATR
    4. Non mitiguée : le prix n'a pas traversé l'OB depuis sa formation
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

        # Corps min + hauteur max
        if abs(b_close - b_open) < min_body:
            continue
        if (b_high - b_low) > OB_MAX_HEIGHT_ATR * atr_val:
            continue

        after = recent.iloc[i + 1: i + 4]

        if direction == "LONG":
            if b_close >= b_open:
                continue
            if float(after["high"].max()) - b_high < min_impulse:
                continue
        else:
            if b_close <= b_open:
                continue
            if b_low - float(after["low"].min()) < min_impulse:
                continue

        # OB non mitiguée
        post = recent.iloc[i + 1:]
        if direction == "LONG" and float(post["low"].min()) < b_low:
            continue
        if direction == "SHORT" and float(post["high"].max()) > b_high:
            continue

        obs.append({"low": b_low, "high": b_high, "ts": recent.index[i]})

    return obs


def _in_ob(bar_low: float, bar_high: float, ob: Dict) -> bool:
    """True si la bougie touche la zone OB [ob_low, ob_high]."""
    return bar_low <= ob["high"] and bar_high >= ob["low"]


# ──────────────────────────────────────────────────────────────────────────────
# 3. Niveaux S/R H1 avec critère de force
# ──────────────────────────────────────────────────────────────────────────────
def _h1_sr_levels(h1: pd.DataFrame, lookback: int = 30,
                   min_touches: int = SR_MIN_TOUCHES,
                   tol_atr: float = 0.5) -> dict:
    """
    Détecte les niveaux S/R H1 significatifs (touchés ≥ min_touches fois).
    Un niveau = swing high/low H1 avec k=2 bougies de chaque côté.
    La force = nombre de fois que le prix est revenu dans tol_atr×ATR du niveau.
    """
    if len(h1) < 8:
        return {"resistance": [], "support": []}

    sub  = h1.tail(lookback)
    h1_atr = float(sub["atr"].iloc[-1] or 1)
    tol  = tol_atr * h1_atr
    n    = len(sub)
    k    = 2  # bougies de chaque côté pour swing H1

    highs_arr = sub["high"].values
    lows_arr  = sub["low"].values

    raw_highs: list = []
    raw_lows:  list = []
    for i in range(k, n - k):
        if highs_arr[i] == max(highs_arr[i - k: i + k + 1]):
            raw_highs.append(float(highs_arr[i]))
        if lows_arr[i] == min(lows_arr[i - k: i + k + 1]):
            raw_lows.append(float(lows_arr[i]))

    def _count_touches(level: float) -> int:
        return sum(
            1 for i in range(n)
            if abs(highs_arr[i] - level) < tol or abs(lows_arr[i] - level) < tol
        )

    def _dedupe(levels: list) -> list:
        result: list = []
        for lv in sorted(levels):
            if not result or abs(lv - result[-1]) > tol:
                result.append(lv)
        return result

    strong_res = [lv for lv in _dedupe(raw_highs) if _count_touches(lv) >= min_touches]
    strong_sup = [lv for lv in _dedupe(raw_lows)  if _count_touches(lv) >= min_touches]

    return {"resistance": strong_res, "support": strong_sup}


# ──────────────────────────────────────────────────────────────────────────────
# 4. Évaluation principale
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

    # 3b) S/R zone H1 — flip directionnel si prix sur un niveau fort (≥2 touches)
    _price    = float(cur["close"])
    h1_atr    = float(h1.iloc[-1].get("atr", atr_val) or atr_val)
    _h1_sr    = _h1_sr_levels(h1, lookback=30)
    _zone_tol = SR_ZONE_ATR_H1 * h1_atr
    _sr_tp2_target = None

    _near_res = any(0 < (r - _price) < _zone_tol for r in _h1_sr["resistance"])
    _near_sup = any(0 < (_price - s) < _zone_tol for s in _h1_sr["support"])

    if _near_res:
        # Prix arrivé sous une résistance forte → SHORT vers le support
        direction = "SHORT"
        _t = nearest_support_below(_price, _h1_sr, min_gap=_zone_tol)
        if _t:
            _sr_tp2_target = _t
    elif _near_sup:
        # Prix revenu au-dessus d'un support fort → LONG vers la résistance
        direction = "LONG"
        _t = nearest_resistance_above(_price, _h1_sr, min_gap=_zone_tol)
        if _t:
            _sr_tp2_target = _t

    _sr_zone_active = _near_res or _near_sup

    # 4) ADX H1 — requis en toutes conditions (trend et S/R)
    h1_adx = float(h1.iloc[-1].get("adx", 0) or 0) if len(h1) > 0 else 0.0
    if h1_adx < ADX_MIN_H1:
        return None

    # 4b) RSI M5 momentum — filtre les entrées sans momentum dans le sens du trade
    rsi_m5 = float(cur.get("rsi", 50) or 50)
    if direction == "LONG"  and rsi_m5 < RSI_LONG_MIN:
        return None
    if direction == "SHORT" and rsi_m5 > RSI_SHORT_MAX:
        return None

    # 4c) RSI M15 — discriminant Δ +4.9 (SL dir 53.8 vs TP2 48.9)
    rsi_m15 = float(m15.iloc[-1].get("rsi", 50) or 50) if len(m15) > 0 else 50.0
    if direction == "LONG"  and rsi_m15 < RSI_M15_LONG_MIN:
        return None
    if direction == "SHORT" and rsi_m15 > RSI_M15_SHORT_MAX:
        return None

    # 4d) RSI H1 — discriminant Δ +5.5 (SL dir 54.9 vs TP2 49.4)
    rsi_h1 = float(h1.iloc[-1].get("rsi", 50) or 50) if len(h1) > 0 else 50.0
    if direction == "LONG"  and rsi_h1 < RSI_H1_LONG_MIN:
        return None
    if direction == "SHORT" and rsi_h1 > RSI_H1_SHORT_MAX:
        return None

    # 4f) VWAP alignment — ne pas entrer contre le VWAP intraday
    vwap_val = float(cur.get("vwap", float("nan")) or float("nan"))
    if not pd.isna(vwap_val) and vwap_val > 0:
        if direction == "LONG"  and float(cur["close"]) < vwap_val:
            return None
        if direction == "SHORT" and float(cur["close"]) > vwap_val:
            return None

    # 5) Order Blocks M5 — obligatoires hors S/R, optionnels (confluence) en mode S/R
    entry_price = float(cur["close"])
    obs = _find_order_blocks(m5, direction, atr_val)
    ob  = obs[-1] if obs else None

    # 6) Retest OB — obligatoire hors S/R, optionnel en mode S/R
    if not _sr_zone_active:
        if ob is None or not _in_ob(float(cur["low"]), float(cur["high"]), ob):
            return None
    # En mode S/R : on entre même sans OB (la zone S/R est la confluence principale)

    # 7) Niveaux du trade
    entry = entry_price

    if direction == "LONG":
        # SL sous l'OB si disponible, sinon sous le support H1 le plus proche SOUS l'entrée
        if ob is not None:
            raw_sl = ob["low"] - SL_BUFFER_ATR * atr_val
        else:
            _sup_below = [s for s in _h1_sr["support"] if s < entry]
            sup = max(_sup_below) if _sup_below else entry - MAX_RISK_ATR * atr_val
            raw_sl = sup - SL_BUFFER_ATR * atr_val
        sl   = max(raw_sl, entry - MAX_RISK_ATR * atr_val)
        risk = abs(entry - sl)
        if risk <= 0:
            return None
        tp1 = entry + TP1_R * risk
        tp2 = entry + TP2_R * risk
    else:
        if ob is not None:
            raw_sl = ob["high"] + SL_BUFFER_ATR * atr_val
        else:
            # Utiliser la résistance la plus proche AU-DESSUS de l'entrée (pas min global)
            _res_above = [r for r in _h1_sr["resistance"] if r > entry]
            res = min(_res_above) if _res_above else entry + MAX_RISK_ATR * atr_val
            raw_sl = res + SL_BUFFER_ATR * atr_val
        sl   = min(raw_sl, entry + MAX_RISK_ATR * atr_val)
        risk = abs(entry - sl)
        if risk <= 0:
            return None
        tp1 = entry - TP1_R * risk
        tp2 = entry - TP2_R * risk

    # TP2 ciblé sur S/R H1 si disponible et distance ≥ SR_TP_MIN_R × risk
    sr_tp2_used = False
    if _sr_tp2_target is not None:
        sr_dist = abs(_sr_tp2_target - entry)
        if sr_dist >= SR_TP_MIN_R * risk:
            tp2 = _sr_tp2_target
            sr_tp2_used = True

    ob_ts_str = None
    if ob is not None:
        ob_ts = ob["ts"]
        ob_ts_str = ob_ts.isoformat() if hasattr(ob_ts, "isoformat") else str(ob_ts)

    reason = ("SR_RETEST" if _sr_zone_active and ob is None
              else "SR_OB_RETEST" if _sr_zone_active
              else "OB_RETEST")

    return Signal(
        direction=direction.lower(),
        bias=direction,
        session=session,
        entry=entry,
        stop_loss=sl,
        take_profit1=tp1,
        take_profit2=tp2,
        atr=atr_val,
        reason=reason,
        risk_distance=risk,
        timestamp=ts,
        meta={
            "strategy":      "B_OB",
            "tp1_close_all": False,
            "ob_low":        round(ob["low"],  5) if ob else None,
            "ob_high":       round(ob["high"], 5) if ob else None,
            "ob_ts":         ob_ts_str,
            "adx_h1":        round(h1_adx, 1),
            "sr_zone":       "resistance" if _near_res else ("support" if _near_sup else None),
            "sr_tp2_target": round(_sr_tp2_target, 5) if sr_tp2_used else None,
        },
    )
