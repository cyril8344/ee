"""
strategy_ict.py
===============
Stratégie ICT pure : Sweep de liquidité → Order Block → OTE Fibonacci (62–79 %)

Conditions d'entrée (toutes obligatoires) :
  1. Biais H1 EMA200 (LONG / SHORT)
  2. Sweep de liquidité récent (swing low/high aspiré puis récupéré)
  3. Prix dans la zone OTE : retracement de 62–79 % de l'impulse
  4. Prix à l'intérieur d'un Order Block correspondant au biais
  5. FVG nearby → confluence optionnelle (+poids)

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
OTE_LOW_PCT        = 0.62   # retracement Fibonacci bas (62 %)
OTE_HIGH_PCT       = 0.79   # retracement Fibonacci haut (79 %)
MAX_TRADE_MIN_ICT  = 60     # durée max d'un trade ICT (minutes)
MIN_IMPULSE_RATIO  = 0.5    # l'impulse doit faire au moins 0.5 × ATR


# --------------------------------------------------------------------------- #
# OTE — zone Fibonacci 62–79 %
# --------------------------------------------------------------------------- #
def find_ote_zone(
    df: pd.DataFrame,
    bias: str,
    lookback: int = SWEEP_LOOKBACK_ICT,
) -> Optional[Dict[str, float]]:
    """
    Calcule la zone OTE (Optimal Trade Entry) de la stratégie ICT.

    Pour LONG : trouve le sweep low (plus bas de la fenêtre) et l'impulse high
    (plus haut des bougies suivant le sweep). La zone OTE est le retracement
    62–79 % de l'impulse.

    Pour SHORT : logique symétrique.

    Retourne dict {low, high, sweep, impulse} ou None.
    """
    if len(df) < lookback + 3:
        return None

    sub = df.tail(lookback)
    atr_val = float(sub["atr"].iloc[-1]) if "atr" in sub.columns else 1.0
    min_impulse = MIN_IMPULSE_RATIO * atr_val

    if bias == "LONG":
        sweep_pos = int(sub["low"].values.argmin())
        # Le sweep doit avoir au moins 2 bougies après lui (impulse existant)
        if sweep_pos >= len(sub) - 2:
            return None
        sweep_low = float(sub["low"].iloc[sweep_pos])
        after = sub.iloc[sweep_pos + 1:]
        impulse_high = float(after["high"].max())
        if impulse_high - sweep_low < min_impulse:
            return None
        span = impulse_high - sweep_low
        return {
            "low":     impulse_high - span * OTE_HIGH_PCT,  # 79 % → entrée basse
            "high":    impulse_high - span * OTE_LOW_PCT,   # 62 % → entrée haute
            "sweep":   sweep_low,
            "impulse": impulse_high,
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
        return {
            "low":     impulse_low + span * OTE_LOW_PCT,    # 62 % → entrée basse
            "high":    impulse_low + span * OTE_HIGH_PCT,   # 79 % → entrée haute
            "sweep":   sweep_high,
            "impulse": impulse_low,
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

    # 3) Biais H1 EMA200
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

    # 6) Zone OTE — le prix doit être dans le retracement 62–79 %
    ote = find_ote_zone(m5, bias)
    if ote is None:
        return None

    entry = float(cur["close"])
    if not (ote["low"] <= entry <= ote["high"]):
        return None

    # 7) Order Block obligatoire — l'entrée doit être dans un OB valide
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

    # 8) Triggers (poids pour le système d'apprentissage)
    triggers = ["ote_fibonacci", "order_block"]
    fvgs = find_fvgs(m5)
    if near_fvg(entry, bias, fvgs):
        triggers.append("near_fvg")

    # 9) SL juste sous/au-dessus de l'OB origin (stop serré ICT)
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
            "rsi_m5":    round(float(cur.get("rsi", 50) or 50), 1),
            "ote_low":   round(ote["low"],       5),
            "ote_high":  round(ote["high"],      5),
            "ob_low":    round(ob_match["low"],  5),
            "ob_high":   round(ob_match["high"], 5),
            "triggers":  triggers,
            "strategy":  "ICT",
        },
    )
