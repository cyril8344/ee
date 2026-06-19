"""
strategy_ict.py
===============
Stratégie ICT : Sweep → MSS → Golden Pocket OTE → Order Block

Conditions d'entrée (toutes obligatoires) :
  1. Biais H1 EMA50 (LONG / SHORT)
  2. Sweep de liquidité récent (swing low/high aspiré puis récupéré)
  3. MSS (Market Structure Shift) : le prix casse un swing intermédiaire après le sweep
  4. Prix dans la Golden Pocket : retracement 70.5–78.6 % de l'impulse
  5. Prix à l'intérieur d'un Order Block correspondant au biais
  6. FVG nearby → confluence optionnelle (+poids)

Gestion du risque :
  SL = juste sous/au-dessus de l'OB origin (stop naturellement serré)
  TP1 = 1.5 R  (clôture 60 %)
  TP2 = 3.0 R  (clôture 40 %)
  Durée max = 60 minutes
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional, Dict, Any, List

import pandas as pd

from strategy import (
    Signal,
    compute_bias, active_session, is_bad_timing,
    liquidity_swept, find_order_blocks, find_fvgs, near_fvg,
    ATR_MIN, EMA_SLOW,
)

# --------------------------------------------------------------------------- #
# Paramètres ICT
# --------------------------------------------------------------------------- #
OB_LOOKBACK_ICT    = 40     # bougies pour la détection d'Order Blocks
SWEEP_LOOKBACK_ICT = 30     # bougies pour le sweep de liquidité
GOLDEN_LOW_PCT     = 0.705  # Golden Pocket bas  (70.5 %)
GOLDEN_HIGH_PCT    = 0.786  # Golden Pocket haut (78.6 % = Fib 0.786)
MAX_TRADE_MIN_ICT  = 60     # durée max d'un trade ICT (minutes)
MIN_IMPULSE_RATIO  = 0.5    # l'impulse doit faire au moins 0.5 × ATR
MSS_PIVOT_WINDOW   = 10     # bougies max avant le sweep pour chercher un pivot


# --------------------------------------------------------------------------- #
# MSS — Market Structure Shift
# --------------------------------------------------------------------------- #
def detect_mss(df: pd.DataFrame, bias: str, lookback: int = SWEEP_LOOKBACK_ICT) -> bool:
    """
    Après le sweep, le prix doit casser un swing intermédiaire (MSS/BOS).
    LONG : après le sweep low, ferme au-dessus d'un pivot high précédent.
    SHORT : après le sweep high, ferme en-dessous d'un pivot low précédent.
    """
    if len(df) < max(lookback, 10):
        return False

    sub = df.tail(lookback)

    if bias == "LONG":
        sweep_pos = int(sub["low"].values.argmin())
        if sweep_pos >= len(sub) - 1:
            return False
        # Chercher un pivot high dans la fenêtre précédant le sweep
        window_start = max(0, sweep_pos - MSS_PIVOT_WINDOW)
        before = sub.iloc[window_start:sweep_pos]
        if len(before) < 2:
            return False
        # Pivot high le plus récent : high local (entouré de bougies plus basses)
        pivot_high: Optional[float] = None
        for i in range(len(before) - 2, 0, -1):
            if (before["high"].iloc[i] > before["high"].iloc[i - 1] and
                    before["high"].iloc[i] > before["high"].iloc[i + 1]):
                pivot_high = float(before["high"].iloc[i])
                break
        if pivot_high is None:
            pivot_high = float(before["high"].max())
        # Le prix doit avoir clôturé au-dessus du pivot après le sweep
        after = sub.iloc[sweep_pos + 1:]
        return any(float(c) > pivot_high for c in after["close"].values)

    else:  # SHORT
        sweep_pos = int(sub["high"].values.argmax())
        if sweep_pos >= len(sub) - 1:
            return False
        window_start = max(0, sweep_pos - MSS_PIVOT_WINDOW)
        before = sub.iloc[window_start:sweep_pos]
        if len(before) < 2:
            return False
        pivot_low: Optional[float] = None
        for i in range(len(before) - 2, 0, -1):
            if (before["low"].iloc[i] < before["low"].iloc[i - 1] and
                    before["low"].iloc[i] < before["low"].iloc[i + 1]):
                pivot_low = float(before["low"].iloc[i])
                break
        if pivot_low is None:
            pivot_low = float(before["low"].min())
        after = sub.iloc[sweep_pos + 1:]
        return any(float(c) < pivot_low for c in after["close"].values)


# --------------------------------------------------------------------------- #
# Golden Pocket — zone Fibonacci 70.5–78.6 %
# --------------------------------------------------------------------------- #
def find_golden_pocket(
    df: pd.DataFrame,
    bias: str,
    lookback: int = SWEEP_LOOKBACK_ICT,
) -> Optional[Dict[str, float]]:
    """
    Calcule la Golden Pocket ICT : retracement 70.5–78.6 % de l'impulse
    post-sweep.

    Pour LONG : sweep low → impulse high → zone = [impulse - 78.6 %, impulse - 70.5 %]
    Pour SHORT : sweep high → impulse low → zone = [impulse + 70.5 %, impulse + 78.6 %]

    Retourne dict {low, high, sweep, impulse, pct_current} ou None.
    """
    if len(df) < max(lookback, 10):
        return None

    sub = df.tail(lookback)
    atr_val = float(sub["atr"].iloc[-1]) if "atr" in sub.columns else 1.0
    min_impulse = MIN_IMPULSE_RATIO * atr_val

    if bias == "LONG":
        sweep_pos = int(sub["low"].values.argmin())
        if sweep_pos >= len(sub) - 2:
            return None
        sweep_low = float(sub["low"].iloc[sweep_pos])
        after = sub.iloc[sweep_pos + 1:]
        impulse_high = float(after["high"].max())
        if impulse_high - sweep_low < min_impulse:
            return None
        span = impulse_high - sweep_low
        entry = float(sub["close"].iloc[-1])
        pct = (impulse_high - entry) / span if span > 0 else 0.0
        return {
            "low":         impulse_high - span * GOLDEN_HIGH_PCT,  # 78.6 % retrace
            "high":        impulse_high - span * GOLDEN_LOW_PCT,   # 70.5 % retrace
            "sweep":       sweep_low,
            "impulse":     impulse_high,
            "pct_current": round(pct, 3),
        }

    else:  # SHORT
        sweep_pos = int(sub["high"].values.argmax())
        if sweep_pos >= len(sub) - 2:
            return None
        sweep_high = float(sub["high"].iloc[sweep_pos])
        after = sub.iloc[sweep_pos + 1:]
        impulse_low = float(after["low"].min())
        if sweep_high - impulse_low < min_impulse:
            return None
        span = sweep_high - impulse_low
        entry = float(sub["close"].iloc[-1])
        pct = (entry - impulse_low) / span if span > 0 else 0.0
        return {
            "low":         impulse_low + span * GOLDEN_LOW_PCT,    # 70.5 % retrace
            "high":        impulse_low + span * GOLDEN_HIGH_PCT,   # 78.6 % retrace
            "sweep":       sweep_high,
            "impulse":     impulse_low,
            "pct_current": round(pct, 3),
        }


# --------------------------------------------------------------------------- #
# Évaluation principale
# --------------------------------------------------------------------------- #
def evaluate_ict(
    m5: pd.DataFrame,
    m15: pd.DataFrame,
    h1: pd.DataFrame,
    now: Optional[datetime] = None,
    check_session: bool = True,
    atr_min: float = ATR_MIN,
) -> Optional[Signal]:
    """
    Évalue la stratégie ICT sur la dernière barre M5 clôturée.
    Les DataFrames doivent déjà contenir les indicateurs (add_indicators).
    Retourne un Signal ou None.
    """
    if len(m5) < max(EMA_SLOW, 50) or len(h1) < 1:
        return None

    cur = m5.iloc[-1]
    ts = now or cur.name.to_pydatetime()
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)

    # 1) Eviter lundi matin / vendredi soir
    if is_bad_timing(ts):
        return None

    # 2) Session Londres / New York uniquement
    session = active_session(ts)
    if check_session and session is None:
        return None
    session = session or "London"

    # 3) Biais H1 EMA50
    bias = compute_bias(h1)
    if bias == "NEUTRE":
        return None

    # 4) Plancher ATR (volatilité minimale)
    atr_val = float(cur.get("atr", 0) or 0)
    if atr_val < atr_min:
        return None

    # 5) Sweep de liquidité récent
    if not liquidity_swept(m5, bias, lookback=SWEEP_LOOKBACK_ICT):
        return None

    # 6) MSS : Market Structure Shift après le sweep
    if not detect_mss(m5, bias):
        return None

    # 7) Golden Pocket — le prix doit être dans le retracement 70.5–78.6 %
    gp = find_golden_pocket(m5, bias)
    if gp is None:
        return None

    entry = float(cur["close"])
    if not (gp["low"] <= entry <= gp["high"]):
        return None

    # 8) Order Block obligatoire — l'entrée doit être dans un OB valide
    obs = find_order_blocks(m5, lookback=OB_LOOKBACK_ICT)
    ob_match: Optional[Dict[str, Any]] = None
    for ob in obs:
        is_match = (ob["type"] == "bullish" and bias == "LONG") or \
                   (ob["type"] == "bearish" and bias == "SHORT")
        if is_match and ob["low"] <= entry <= ob["high"]:
            ob_match = ob
            break

    if ob_match is None:
        return None

    # 9) Triggers (poids pour le système d'apprentissage)
    triggers = ["golden_pocket", "order_block"]
    if detect_mss(m5, bias):
        triggers.append("mss_confirmed")
    fvgs = find_fvgs(m5)
    if near_fvg(entry, bias, fvgs):
        triggers.append("near_fvg")

    # 10) SL juste sous/au-dessus de l'OB origin (stop serré ICT)
    if bias == "LONG":
        sl = min(ob_match["low"] - 0.1 * atr_val, entry - 0.5 * atr_val)
        direction = "long"
    else:
        sl = max(ob_match["high"] + 0.1 * atr_val, entry + 0.5 * atr_val)
        direction = "short"

    risk = abs(entry - sl)
    if risk <= 0:
        return None

    if direction == "long":
        tp1 = entry + 1.5 * risk
        tp2 = entry + 3.0 * risk
    else:
        tp1 = entry - 1.5 * risk
        tp2 = entry - 3.0 * risk

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
        max_duration_min=MAX_TRADE_MIN_ICT,
        meta={
            "rsi_m5":      round(float(cur.get("rsi", 50) or 50), 1),
            "gp_low":      round(gp["low"],        5),
            "gp_high":     round(gp["high"],       5),
            "gp_pct":      gp["pct_current"],
            "ob_low":      round(ob_match["low"],  5),
            "ob_high":     round(ob_match["high"], 5),
            "triggers":    triggers,
            "strategy":    "ICT",
        },
    )
