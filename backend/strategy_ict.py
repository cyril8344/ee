"""
strategy_ict.py
===============
Stratégie B — Smart Money Concepts (ICT) — Order Block + H1 trend

Pipeline : H1 biais → M5 Order Block (retour sur zone)

Conditions d'entrée :
  1. Biais H1 EMA50/EMA200
  2. Plancher ATR M5 (volatilité minimale)
  3. Order Block M5 — le prix actuel est dans un OB aligné avec le biais
     (OB = bougie opposée précédant une impulsion ≥ 1.0×ATR)

Gestion du risque :
  SL  = juste sous/au-dessus du bord de l'OB
  TP1 = TP1_R × risque  (défaut 1.5R)
  TP2 = TP2_R × risque  (défaut 3.0R)
  Durée max = MAX_TRADE_MIN_ICT minutes

Anti-look-ahead :
  Toutes les décisions utilisent uniquement les bougies clôturées AVANT `now`.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional, Dict, Any, List

import pandas as pd

from strategy import (
    Signal,
    compute_bias, active_session, is_bad_timing,
    find_fvgs, near_fvg,
    ATR_MIN, EMA_SLOW,
)

# ──────────────────────────────────────────────────────────────────────────────
# Paramètres
# ──────────────────────────────────────────────────────────────────────────────
OB_LOOKBACK_ICT    = 40    # bougies M5 pour détecter les Order Blocks
ACC_LOOKBACK       = 12    # bougies M15 de la phase d'accumulation
BRK_LOOKBACK       = 4     # bougies M15 pour détecter le breakout
ACC_RANGE_ATR      = 2.5   # range max de l'accumulation en multiples d'ATR
MAX_TRADE_MIN_ICT  = 60    # durée max d'un trade (minutes)
TP1_R              = 1.5   # TP1 en multiple de risque
TP2_R              = 3.0   # TP2 en multiple de risque
OB_SL_BUFFER_ATR   = 0.1  # buffer SL au-delà du bord de l'OB
OB_SL_MIN_ATR      = 0.5  # SL minimum en ATR


# ──────────────────────────────────────────────────────────────────────────────
# Détection accumulation + breakout sur M15
# ──────────────────────────────────────────────────────────────────────────────
def detect_accumulation_breakout(
    m15: pd.DataFrame,
    bias: str,
    acc_lookback: int = ACC_LOOKBACK,
    brk_lookback: int = BRK_LOOKBACK,
    range_atr: float = ACC_RANGE_ATR,
) -> bool:
    """
    Retourne True si :
    1. Les `acc_lookback` bougies M15 précédentes forment un range serré
       (high - low < range_atr × ATR) — phase d'accumulation/distribution
    2. Une des `brk_lookback` dernières bougies M15 casse ce range dans
       la direction du biais — breakout de la consolidation

    Pas de look-ahead : toutes les barres sont déjà clôturées.
    """
    total_needed = acc_lookback + brk_lookback + 1
    if len(m15) < total_needed:
        return False

    atr_val = float(m15.iloc[-1].get("atr", 1.0) or 1.0)

    # Phase accumulation : les acc_lookback bougies avant le breakout
    acc = m15.iloc[-(acc_lookback + brk_lookback): -brk_lookback]
    acc_high = float(acc["high"].max())
    acc_low  = float(acc["low"].min())
    acc_range = acc_high - acc_low

    if acc_range > range_atr * atr_val:
        return False  # trop volatile → pas en accumulation

    # Phase breakout : les brk_lookback dernières bougies
    brk = m15.iloc[-brk_lookback:]
    if bias == "LONG":
        return bool(any(float(c) > acc_high for c in brk["close"].values))
    else:
        return bool(any(float(c) < acc_low  for c in brk["close"].values))


# ──────────────────────────────────────────────────────────────────────────────
# Order Blocks avec age_bars
# ──────────────────────────────────────────────────────────────────────────────
def _find_obs_with_age(df: pd.DataFrame, lookback: int = OB_LOOKBACK_ICT) -> List[Dict[str, Any]]:
    """
    Détecte les Order Blocks sur les `lookback` dernières bougies.
    Un OB est la bougie précédant une bougie impulsive dans la direction opposée.
    Ajoute `age_bars` : nombre de bougies depuis la formation de l'OB.
    """
    sub = df.tail(lookback)
    atr_val = float(sub["atr"].iloc[-1]) if "atr" in sub.columns else 1.0
    n = len(sub)
    obs: List[Dict[str, Any]] = []
    for i in range(1, n - 1):
        c, nxt = sub.iloc[i], sub.iloc[i + 1]
        impulse = abs(nxt["close"] - nxt["open"])
        if impulse < 1.0 * atr_val:
            continue
        age = n - 1 - i
        if c["close"] < c["open"] and nxt["close"] > nxt["open"]:
            obs.append({"type": "bullish",
                        "low": float(c["close"]), "high": float(c["open"]),
                        "age_bars": age})
        elif c["close"] > c["open"] and nxt["close"] < nxt["open"]:
            obs.append({"type": "bearish",
                        "low": float(c["open"]), "high": float(c["close"]),
                        "age_bars": age})
    return obs


# ──────────────────────────────────────────────────────────────────────────────
# Évaluation principale — Stratégie B simplifiée
# ──────────────────────────────────────────────────────────────────────────────
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

    H1 → biais directionnel (EMA50/EMA200)
    M5 → entrée sur Order Block (retour sur zone, impulsion ≥ 1.0×ATR)

    Retourne un Signal ou None.
    """
    if len(m5) < max(EMA_SLOW, 50) or len(h1) < 1:
        return None

    cur = m5.iloc[-1]
    ts  = now or cur.name.to_pydatetime()
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)

    # 1) Timing
    if is_bad_timing(ts):
        return None

    # 2) Session Londres / New York
    session = active_session(ts)
    if check_session and session is None:
        return None
    session = session or "London"

    # 3) Biais H1
    bias = compute_bias(h1)
    if bias == "NEUTRE":
        return None
    direction = "long" if bias == "LONG" else "short"

    # 4) Plancher ATR M5
    atr_val = float(cur.get("atr", 0) or 0)
    if atr_val < atr_min:
        return None

    # 5) Order Block M5 — le prix actuel est dans un OB aligné avec le biais
    entry_price = float(cur["close"])
    obs = _find_obs_with_age(m5, lookback=OB_LOOKBACK_ICT)
    ob_match: Optional[Dict[str, Any]] = None
    for ob in obs:
        is_match = (ob["type"] == "bullish" and bias == "LONG") or \
                   (ob["type"] == "bearish" and bias == "SHORT")
        if is_match and ob["low"] <= entry_price <= ob["high"]:
            ob_match = ob
            break

    if ob_match is None:
        return None

    # 6) FVG confluence (optionnelle — loggée)
    fvgs    = find_fvgs(m5)
    fvg_hit = near_fvg(entry_price, bias, fvgs)

    # 7) SL juste sous/au-dessus du bord de l'OB
    if bias == "LONG":
        sl = min(ob_match["low"] - OB_SL_BUFFER_ATR * atr_val,
                 entry_price     - OB_SL_MIN_ATR     * atr_val)
    else:
        sl = max(ob_match["high"] + OB_SL_BUFFER_ATR * atr_val,
                 entry_price      + OB_SL_MIN_ATR     * atr_val)

    risk = abs(entry_price - sl)
    if risk <= 0:
        return None

    tp1 = entry_price + TP1_R * risk if direction == "long" else entry_price - TP1_R * risk
    tp2 = entry_price + TP2_R * risk if direction == "long" else entry_price - TP2_R * risk

    triggers = ["order_block"]
    if fvg_hit:
        triggers.append("near_fvg")

    ob_size_atr = (ob_match["high"] - ob_match["low"]) / atr_val if atr_val > 0 else 0.0
    ob_dist_atr = abs(entry_price - (ob_match["low"] if bias == "LONG" else ob_match["high"])) / atr_val if atr_val > 0 else 0.0

    return Signal(
        direction=direction,
        bias=bias,
        session=session,
        entry=entry_price,
        stop_loss=sl,
        take_profit1=tp1,
        take_profit2=tp2,
        atr=atr_val,
        reason="+".join(triggers),
        risk_distance=risk,
        timestamp=ts,
        max_duration_min=MAX_TRADE_MIN_ICT,
        meta={
            "strategy":        "ICT_B",
            "ob_size_atr":     round(ob_size_atr, 3),
            "ob_distance_atr": round(ob_dist_atr, 3),
            "ob_age_bars":     ob_match.get("age_bars", 0),
            "fvg_present":     int(fvg_hit),
            "rsi_m5":          round(float(cur.get("rsi", 50) or 50), 1),
            "triggers":        triggers,
        },
    )
