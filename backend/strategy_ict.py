"""
strategy_ict.py — Stratégie B : Order Block M5
===============================================
Pipeline :
1. Biais H1 (EMA50 vs EMA200)
2. Zone Discount/Premium H1 (LONG en discount <50% range, SHORT en premium >50%)
3. Détection OBs M5 valides (4 critères ICT) :
   a. Bougie contrariante + impulse ≥ 1.5×ATR
   b. FVG créée après l'OB (imbalance institutionnelle)
   c. BOS confirmé (impulse casse le swing récent)
   d. Corps min + hauteur max (filtre dojis et OBs larges)
   e. Non mitiguée
4. Entrée en retest de la zone OB+FVG (confluence)
5. SL = derrière l'OB + buffer, plafonné à MAX_RISK_ATR×ATR
6. TP1 = 0.7R, TP2 = 1.8R

OB LONG  : dernière bougie rouge avant impulse haussier ≥ 1.5×ATR + FVG + BOS haussier
OB SHORT : dernière bougie verte avant impulse baissier ≥ 1.5×ATR + FVG + BOS baissier
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional, Dict, Any, List

import pandas as pd

from strategy import Signal, active_session, is_bad_timing, ATR_MIN

# ──────────────────────────────────────────────────────────────────────────────
# Paramètres
# ──────────────────────────────────────────────────────────────────────────────
OB_IMPULSE_ATR    = 1.5  # impulse minimum après l'OB (en ATR)
OB_MAX_BARS       = 50   # fenêtre de recherche OBs (50 M5 ≈ 4h)
OB_MIN_BODY_ATR   = 0.2  # corps minimum bougie OB (filtre dojis)
OB_MAX_HEIGHT_ATR = 1.5  # hauteur maximale OB (OBs larges → R:R défavorable)
BOS_LOOKBACK      = 15   # barres lookback pour swing à casser (BOS)
H1_RANGE_BARS     = 20   # barres H1 pour zone Premium/Discount (~1 jour)
SL_BUFFER_ATR     = 0.3  # buffer SL derrière l'extrême de l'OB
MAX_RISK_ATR      = 1.5  # plafond risque (SL ≤ 1.5×ATR de l'entrée)
TP1_R             = 0.7
TP2_R             = 1.8


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
# 2. Zone Discount / Premium H1
# ──────────────────────────────────────────────────────────────────────────────
def _discount_premium(price: float, h1: pd.DataFrame) -> Optional[str]:
    """
    'DISCOUNT' si price < équilibre (50% range H1 récent), 'PREMIUM' si >.
    None si range indéterminé.
    LONG uniquement en DISCOUNT, SHORT uniquement en PREMIUM (zones institutionnelles).
    """
    if len(h1) < 5:
        return None
    window = h1.tail(H1_RANGE_BARS)
    hi  = float(window["high"].max())
    lo  = float(window["low"].min())
    rng = hi - lo
    if rng < 1e-8:
        return None
    eq = lo + 0.5 * rng
    return "DISCOUNT" if price < eq else "PREMIUM"


# ──────────────────────────────────────────────────────────────────────────────
# 3. Détection des Order Blocks M5
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
    3. FVG : bar[i+2].low > bar[i].high (LONG) | bar[i+2].high < bar[i].low (SHORT)
    4. BOS : impulse dépasse le swing high/low des BOS_LOOKBACK barres précédentes
    5. Corps ≥ OB_MIN_BODY_ATR×ATR | hauteur ≤ OB_MAX_HEIGHT_ATR×ATR
    6. Non mitiguée : le prix n'a pas traversé l'OB depuis sa formation

    Retourne fvg_ext : bord externe de la FVG, qui complète la zone de confluence.
      LONG  → fvg_ext = bar[i+2].low  (plafond de la zone OB+FVG = [ob_low,  fvg_ext])
      SHORT → fvg_ext = bar[i+2].high (plancher de la zone OB+FVG = [fvg_ext, ob_high])
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

        # Critère 5 : corps min + hauteur max
        if abs(b_close - b_open) < min_body:
            continue
        if (b_high - b_low) > OB_MAX_HEIGHT_ATR * atr_val:
            continue

        after = recent.iloc[i + 1: i + 4]

        if direction == "LONG":
            # Critère 1 : bougie baissière (rouge)
            if b_close >= b_open:
                continue

            # Critère 2 : impulse haussier
            impulse = float(after["high"].max()) - b_high
            if impulse < min_impulse:
                continue

            # Critère 3 : FVG — bar[i+2].low > bar[i].high
            fvg_ext = float(recent.iloc[i + 2]["low"])
            if fvg_ext <= b_high:
                continue  # pas de FVG (pas d'imbalance institutionnelle)

            # Critère 4 : BOS — impulse casse le swing high récent
            lookback_start = max(0, i - BOS_LOOKBACK)
            prior = recent.iloc[lookback_start:i]
            if len(prior) < 3:
                continue
            if float(after["high"].max()) <= float(prior["high"].max()):
                continue  # structure non cassée

        else:  # SHORT
            # Critère 1 : bougie haussière (verte)
            if b_close <= b_open:
                continue

            # Critère 2 : impulse baissier
            impulse = b_low - float(after["low"].min())
            if impulse < min_impulse:
                continue

            # Critère 3 : FVG — bar[i+2].high < bar[i].low
            fvg_ext = float(recent.iloc[i + 2]["high"])
            if fvg_ext >= b_low:
                continue  # pas de FVG

            # Critère 4 : BOS — impulse casse le swing low récent
            lookback_start = max(0, i - BOS_LOOKBACK)
            prior = recent.iloc[lookback_start:i]
            if len(prior) < 3:
                continue
            if float(after["low"].min()) >= float(prior["low"].min()):
                continue  # structure non cassée

        # Critère 6 : OB non mitiguée
        post = recent.iloc[i + 1:]
        if direction == "LONG" and float(post["low"].min()) < b_low:
            continue
        if direction == "SHORT" and float(post["high"].max()) > b_high:
            continue

        obs.append({
            "low":     b_low,
            "high":    b_high,
            "fvg_ext": fvg_ext,
            "ts":      recent.index[i],
        })

    return obs


def _in_confluence(bar_low: float, bar_high: float, ob: Dict, direction: str) -> bool:
    """
    True si la bougie touche la zone OB+FVG (confluence institutionnelle).
    LONG  : zone = [ob_low,  fvg_ext] — FVG au-dessus de l'OB
    SHORT : zone = [fvg_ext, ob_high] — FVG en dessous de l'OB
    """
    if direction == "LONG":
        return bar_low <= ob["fvg_ext"] and bar_high >= ob["low"]
    else:
        return bar_high >= ob["fvg_ext"] and bar_low <= ob["high"]


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
    """Stratégie B — Order Block M5 avec biais H1 et filtres ICT complets."""
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

    # 4) Zone Discount / Premium H1
    entry_price = float(cur["close"])
    zone = _discount_premium(entry_price, h1)
    if direction == "LONG" and zone != "DISCOUNT":
        return None
    if direction == "SHORT" and zone != "PREMIUM":
        return None

    # 5) Order Blocks M5 valides (FVG + BOS intégrés)
    obs = _find_order_blocks(m5, direction, atr_val)
    if not obs:
        return None

    # Dernier OB valide (le plus récent)
    ob = obs[-1]

    # 6) Prix actuel en confluence OB+FVG
    if not _in_confluence(float(cur["low"]), float(cur["high"]), ob, direction):
        return None

    # 7) Niveaux du trade
    entry = entry_price

    if direction == "LONG":
        raw_sl = ob["low"] - SL_BUFFER_ATR * atr_val
        sl     = max(raw_sl, entry - MAX_RISK_ATR * atr_val)
        risk   = abs(entry - sl)
        if risk <= 0:
            return None
        tp1 = entry + TP1_R * risk
        tp2 = entry + TP2_R * risk
    else:
        raw_sl = ob["high"] + SL_BUFFER_ATR * atr_val
        sl     = min(raw_sl, entry + MAX_RISK_ATR * atr_val)
        risk   = abs(entry - sl)
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
            "fvg_ext":  round(ob["fvg_ext"], 5),
            "ob_ts":    ob_ts_str,
            "zone":     zone or "UNKNOWN",
        },
    )
