"""
strategy_ict.py
===============
Stratégie B — Smart Money Concepts (ICT)
Pipeline : M15 structure → M5 entrée → H1 biais

Conditions d'entrée (toutes obligatoires) :
  1. Biais H1 EMA50/EMA200 (compute_bias)
  2. EMA200 H1 — distance maximale configurable
  3. VWAP intraday — biais directionnel (rejet si prix trop loin du mauvais côté)
  4. Sweep de liquidité sur M15
  5. CHoCH (Change of Character) sur M15 — signal de retournement requis
  6. BOS (Break of Structure) sur M15 — confirmation optionnelle, loggé comme feature
  7. Golden Pocket 70.5–78.6 % sur M5 (retracement de l'impulse)
  8. Order Block obligatoire sur M5 (zone d'entrée précise)
  9. FVG confluence optionnelle sur M5 (loggé comme feature)

Gestion du risque :
  SL  = juste sous/au-dessus de l'OB (stop naturellement serré)
  TP1 = TP1_R × risque  (défaut 1.5R)
  TP2 = TP2_R × risque  (défaut 3.0R)
  Durée max = MAX_TRADE_MIN_ICT minutes

Anti-look-ahead :
  Chaque décision utilise uniquement les bougies clôturées AVANT le timestamp courant.
  Le Golden Pocket utilise l'impulse high/low déjà formé dans la fenêtre passée.
  Le VWAP se calcule en cumulatif depuis minuit UTC — pas de données futures.
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

# ──────────────────────────────────────────────────────────────────────────────
# Paramètres — tous en constantes nommées (Optuna-ready, jamais codés en dur)
# ──────────────────────────────────────────────────────────────────────────────
OB_LOOKBACK_ICT       = 40     # bougies M5 pour détecter les Order Blocks
SWEEP_LOOKBACK_ICT    = 30     # bougies M15 pour le sweep + CHoCH/BOS
GOLDEN_LOW_PCT        = 0.618  # Golden Pocket bas  (61.8 % = Fib classique)
GOLDEN_HIGH_PCT       = 0.786  # Golden Pocket haut (78.6 % = Fib 0.786)
MAX_TRADE_MIN_ICT     = 60     # durée max d'un trade (minutes)
MIN_IMPULSE_RATIO     = 0.5    # l'impulse M5/M15 doit valoir au moins N × ATR
MSS_PIVOT_WINDOW      = 10     # fenêtre (bougies) pour chercher le pivot pré-sweep
TP1_R                 = 1.5    # TP1 en multiple de risque
TP2_R                 = 3.0    # TP2 en multiple de risque
OB_SL_BUFFER_ATR      = 0.1   # buffer SL au-delà du bord de l'OB (en ATR)
OB_SL_MIN_ATR         = 0.5   # SL minimum en ATR (plancher de sécurité)
EMA200_MAX_DIST_ATR   = 3.0   # distance max prix/EMA200 H1 ; au-delà = pas de trade
VWAP_REJECT_ATR       = 1.0   # rejette si prix est à > N ATR du mauvais côté du VWAP


# ──────────────────────────────────────────────────────────────────────────────
# VWAP intraday — reset à minuit UTC
# ──────────────────────────────────────────────────────────────────────────────
def compute_vwap(df: pd.DataFrame) -> pd.Series:
    """
    VWAP typique (H+L+C)/3, pondéré par le volume, reset à minuit UTC chaque jour.
    Si la colonne 'volume' est absente ou nulle, utilise un poids uniforme.
    Pas de look-ahead : chaque valeur n'utilise que les bougies précédentes du même jour.
    """
    typical = (df["high"] + df["low"] + df["close"]) / 3.0
    if "volume" in df.columns:
        vol = df["volume"].clip(lower=1.0)
    else:
        vol = pd.Series(1.0, index=df.index)

    dates = pd.Series(df.index.date, index=df.index)
    result = pd.Series(index=df.index, dtype=float)
    for date in dates.unique():
        mask = dates == date
        tp = typical[mask]
        v  = vol[mask]
        result[mask] = (tp * v).cumsum() / v.cumsum()
    return result


# ──────────────────────────────────────────────────────────────────────────────
# CHoCH — Change of Character (signal de retournement principal)
# ──────────────────────────────────────────────────────────────────────────────
def detect_choch(df: pd.DataFrame, bias: str, lookback: int = SWEEP_LOOKBACK_ICT) -> bool:
    """
    CHoCH : après le sweep, le prix clôture au-delà d'un pivot de la phase précédente.

    LONG  : sweep low → clôture au-dessus d'un pivot high récent
    SHORT : sweep high → clôture en-dessous d'un pivot low récent

    Pas de look-ahead : argmin/argmax sur bougies clôturées, 'after' = ce qui suit dans le passé.
    """
    if len(df) < max(lookback, 10):
        return False

    sub = df.tail(lookback)

    if bias == "LONG":
        sweep_pos = int(sub["low"].values.argmin())
        if sweep_pos >= len(sub) - 1:
            return False
        before = sub.iloc[max(0, sweep_pos - MSS_PIVOT_WINDOW): sweep_pos]
        if len(before) < 2:
            return False
        pivot_high: Optional[float] = None
        for i in range(len(before) - 2, 0, -1):
            if (before["high"].iloc[i] > before["high"].iloc[i - 1] and
                    before["high"].iloc[i] > before["high"].iloc[i + 1]):
                pivot_high = float(before["high"].iloc[i])
                break
        if pivot_high is None:
            pivot_high = float(before["high"].max())
        after = sub.iloc[sweep_pos + 1:]
        return any(float(c) > pivot_high for c in after["close"].values)

    else:  # SHORT
        sweep_pos = int(sub["high"].values.argmax())
        if sweep_pos >= len(sub) - 1:
            return False
        before = sub.iloc[max(0, sweep_pos - MSS_PIVOT_WINDOW): sweep_pos]
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


# ──────────────────────────────────────────────────────────────────────────────
# BOS — Break of Structure (confirmation de la nouvelle tendance)
# ──────────────────────────────────────────────────────────────────────────────
def detect_bos(df: pd.DataFrame, bias: str, lookback: int = SWEEP_LOOKBACK_ICT) -> bool:
    """
    BOS : après le CHoCH, le prix forme un nouveau higher high (LONG) ou lower low (SHORT),
    confirmant que la structure a définitivement changé.

    Plus conservateur que CHoCH — indique une continuation de la nouvelle direction.
    Pas de look-ahead : tout calculé sur les bougies historiques clôturées.
    """
    if len(df) < max(lookback, 15):
        return False

    sub = df.tail(lookback)

    if bias == "LONG":
        sweep_pos = int(sub["low"].values.argmin())
        if sweep_pos >= len(sub) - 3:
            return False
        after_sweep = sub.iloc[sweep_pos + 1:]
        if len(after_sweep) < 3:
            return False
        highs = after_sweep["high"].values
        for i in range(1, len(highs)):
            if highs[i] > highs[i - 1]:
                lows_between = after_sweep["low"].values[:i]
                if len(lows_between) > 0 and float(lows_between.min()) < float(highs[0]):
                    return True
        return False

    else:  # SHORT
        sweep_pos = int(sub["high"].values.argmax())
        if sweep_pos >= len(sub) - 3:
            return False
        after_sweep = sub.iloc[sweep_pos + 1:]
        if len(after_sweep) < 3:
            return False
        lows = after_sweep["low"].values
        for i in range(1, len(lows)):
            if lows[i] < lows[i - 1]:
                highs_between = after_sweep["high"].values[:i]
                if len(highs_between) > 0 and float(highs_between.max()) > float(lows[0]):
                    return True
        return False


# ──────────────────────────────────────────────────────────────────────────────
# Golden Pocket — zone Fibonacci 70.5–78.6 %
# ──────────────────────────────────────────────────────────────────────────────
def find_golden_pocket(
    df: pd.DataFrame,
    bias: str,
    lookback: int = SWEEP_LOOKBACK_ICT,
) -> Optional[Dict[str, float]]:
    """
    Calcule la zone Golden Pocket ICT : retracement 70.5–78.6 % de l'impulse post-sweep.

    Pas de look-ahead :
    - Le sweep est le min/max de la fenêtre passée.
    - L'impulse est le max/min des bougies APRÈS le sweep (toutes déjà clôturées).
    - La barre courante (close) est la candidate à l'entrée.

    Retourne {low, high, sweep, impulse, pct_current} ou None.
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
        sweep_low    = float(sub["low"].iloc[sweep_pos])
        impulse_high = float(sub.iloc[sweep_pos + 1:]["high"].max())
        if impulse_high - sweep_low < min_impulse:
            return None
        span  = impulse_high - sweep_low
        entry = float(sub["close"].iloc[-1])
        pct   = (impulse_high - entry) / span if span > 0 else 0.0
        return {
            "low":         impulse_high - span * GOLDEN_HIGH_PCT,
            "high":        impulse_high - span * GOLDEN_LOW_PCT,
            "sweep":       sweep_low,
            "impulse":     impulse_high,
            "pct_current": round(pct, 3),
        }

    else:  # SHORT
        sweep_pos = int(sub["high"].values.argmax())
        if sweep_pos >= len(sub) - 2:
            return None
        sweep_high  = float(sub["high"].iloc[sweep_pos])
        impulse_low = float(sub.iloc[sweep_pos + 1:]["low"].min())
        if sweep_high - impulse_low < min_impulse:
            return None
        span  = sweep_high - impulse_low
        entry = float(sub["close"].iloc[-1])
        pct   = (entry - impulse_low) / span if span > 0 else 0.0
        return {
            "low":         impulse_low + span * GOLDEN_LOW_PCT,
            "high":        impulse_low + span * GOLDEN_HIGH_PCT,
            "sweep":       sweep_high,
            "impulse":     impulse_low,
            "pct_current": round(pct, 3),
        }


# ──────────────────────────────────────────────────────────────────────────────
# Order Blocks avec age_bars (version locale qui ne touche pas strategy.py)
# ──────────────────────────────────────────────────────────────────────────────
def _find_obs_with_age(df: pd.DataFrame, lookback: int = OB_LOOKBACK_ICT) -> List[Dict[str, Any]]:
    """
    Identique à find_order_blocks() de strategy.py mais ajoute 'age_bars'
    (nombre de bougies entre la formation de l'OB et la barre courante).
    N'importe pas de logique dans strategy.py.
    """
    sub = df.tail(lookback)
    atr_val = float(sub["atr"].iloc[-1]) if "atr" in sub.columns else 1.0
    n = len(sub)
    obs: List[Dict[str, Any]] = []
    for i in range(1, n - 1):
        c, nxt = sub.iloc[i], sub.iloc[i + 1]
        impulse = abs(nxt["close"] - nxt["open"])
        if impulse < 0.5 * atr_val:
            continue
        age = n - 1 - i  # bougies depuis la formation jusqu'à la barre courante
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
# Helpers features ML
# ──────────────────────────────────────────────────────────────────────────────
def _m15_swing_distance_atr(m15: pd.DataFrame, bias: str, atr_val: float) -> float:
    """Distance M5 close → dernier swing M15, en multiples d'ATR."""
    if len(m15) < 5 or atr_val <= 0:
        return 0.0
    sub = m15.tail(SWEEP_LOOKBACK_ICT)
    cur_price = float(m15.iloc[-1]["close"])
    if bias == "LONG":
        swing = float(sub["low"].min())
    else:
        swing = float(sub["high"].max())
    return round(abs(cur_price - swing) / atr_val, 3)


def _sweep_amplitude_atr(m15: pd.DataFrame, bias: str, atr_val: float) -> float:
    """Amplitude du sweep (swing aspiré → clôture de récupération) en ATR."""
    if len(m15) < SWEEP_LOOKBACK_ICT or atr_val <= 0:
        return 0.0
    sub = m15.tail(SWEEP_LOOKBACK_ICT)
    close_now = float(sub["close"].iloc[-1])
    if bias == "LONG":
        sweep_extreme = float(sub["low"].min())
        return round(max(0.0, (close_now - sweep_extreme) / atr_val), 3)
    else:
        sweep_extreme = float(sub["high"].max())
        return round(max(0.0, (sweep_extreme - close_now) / atr_val), 3)


def _fvg_context(price: float, bias: str,
                 fvgs: List[Dict[str, Any]], atr_val: float) -> Dict[str, float]:
    """Retourne {size_atr, entry_pct} du FVG dans lequel se trouve le prix, ou zéros."""
    for fvg in fvgs:
        match = (fvg["type"] == "bullish" and bias == "LONG") or \
                (fvg["type"] == "bearish" and bias == "SHORT")
        if match and fvg["low"] <= price <= fvg["high"]:
            span = fvg["high"] - fvg["low"]
            return {
                "size_atr":  round(span / atr_val, 3) if atr_val > 0 else 0.0,
                "entry_pct": round((price - fvg["low"]) / span, 3) if span > 0 else 0.5,
            }
    return {"size_atr": 0.0, "entry_pct": 0.0}


def _ema50_slope(h1: pd.DataFrame, n: int = 5) -> float:
    """Pente de l'EMA50 H1 sur les n dernières bougies (en prix/bougie)."""
    if "ema50" not in h1.columns or len(h1) < n + 1:
        return 0.0
    vals = h1["ema50"].tail(n + 1).values
    return round(float(vals[-1] - vals[0]) / n, 5)


# ──────────────────────────────────────────────────────────────────────────────
# Évaluation principale — Stratégie B
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
    Évalue la stratégie ICT/SMC sur la dernière barre M5 clôturée.

    M15 → structure (sweep + CHoCH/BOS)
    M5  → entrée précise (Golden Pocket + Order Block + FVG)
    H1  → biais (EMA50/EMA200 + VWAP)

    Retourne un Signal (avec meta enrichi pour le logging ML) ou None.
    Garantie no-look-ahead : toutes les données utilisées sont antérieures à `now`.
    """
    if len(m5) < max(EMA_SLOW, 50) or len(h1) < 1 or len(m15) < 20:
        return None

    cur = m5.iloc[-1]
    ts  = now or cur.name.to_pydatetime()
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

    # 3) Biais H1 — EMA50 direction, EMA200 filtre contradictoire
    bias = compute_bias(h1)
    if bias == "NEUTRE":
        return None
    direction = "long" if bias == "LONG" else "short"

    # 4) Distance EMA200 H1 — ne pas trader si trop loin contre la tendance
    h1_last   = h1.iloc[-1]
    h1_atr    = float(h1_last.get("atr", 1.0) or 1.0)
    h1_close  = float(h1_last["close"])
    h1_ema200 = float(h1_last.get("ema200", float("nan")))
    h1_ema50  = float(h1_last.get("ema50",  float("nan")))
    if not pd.isna(h1_ema200) and h1_atr > 0:
        dist_ema200 = (h1_close - h1_ema200) / h1_atr
        if direction == "long"  and dist_ema200 < -EMA200_MAX_DIST_ATR:
            return None
        if direction == "short" and dist_ema200 >  EMA200_MAX_DIST_ATR:
            return None
    else:
        dist_ema200 = 0.0

    # 5) Plancher ATR M5 (volatilité minimale)
    atr_val = float(cur.get("atr", 0) or 0)
    if atr_val < atr_min:
        return None

    # 6) VWAP intraday — biais directionnel
    vwap_series = compute_vwap(m5)
    vwap_now    = float(vwap_series.iloc[-1]) if len(vwap_series) > 0 else float("nan")
    entry_price = float(cur["close"])
    if not pd.isna(vwap_now) and atr_val > 0:
        vwap_dist = (entry_price - vwap_now) / atr_val
        if direction == "long"  and vwap_dist < -VWAP_REJECT_ATR:
            return None
        if direction == "short" and vwap_dist >  VWAP_REJECT_ATR:
            return None
    else:
        vwap_dist = 0.0

    # 7) Sweep de liquidité sur M15
    if not liquidity_swept(m15, bias, lookback=SWEEP_LOOKBACK_ICT):
        return None

    # 8) CHoCH sur M15 (requis — retournement de structure)
    if not detect_choch(m15, bias):
        return None

    # BOS sur M15 (optionnel — loggé comme feature, non bloquant)
    has_bos    = detect_bos(m15, bias)
    signal_type = "BOS" if has_bos else "CHoCH"

    # 9) Golden Pocket — prix M5 dans le retracement 70.5–78.6 % de l'impulse
    gp = find_golden_pocket(m5, bias)
    if gp is None:
        return None
    if not (gp["low"] <= entry_price <= gp["high"]):
        return None

    # 10) Order Block M5 — entrée dans un OB valide (avec age_bars pour ML)
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

    # 11) FVG confluence (optionnelle — loggée comme feature ML)
    fvgs    = find_fvgs(m5)
    fvg_hit = near_fvg(entry_price, bias, fvgs)
    fvg_ctx = _fvg_context(entry_price, bias, fvgs, atr_val)

    # 12) SL juste sous/au-dessus du bord de l'OB
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

    # 13) Triggers
    triggers = ["golden_pocket", "order_block", signal_type.lower()]
    if fvg_hit:
        triggers.append("near_fvg")

    # Features ML (toutes calculées ici, au moment de l'entrée, sans donnée future)
    ema50_dist_atr   = (h1_close - h1_ema50)  / h1_atr if (not pd.isna(h1_ema50)  and h1_atr > 0) else 0.0
    ema200_dist_atr  = (h1_close - h1_ema200) / h1_atr if (not pd.isna(h1_ema200) and h1_atr > 0) else 0.0
    ob_size_atr      = (ob_match["high"] - ob_match["low"]) / atr_val if atr_val > 0 else 0.0
    ob_dist_atr      = abs(entry_price - (ob_match["low"] if bias == "LONG" else ob_match["high"])) / atr_val if atr_val > 0 else 0.0

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
            "strategy":             "ICT_B",
            # Structure
            "signal_type":          signal_type,
            "m15_swing_dist_atr":   _m15_swing_distance_atr(m15, bias, atr_val),
            # Order Block
            "ob_size_atr":          round(ob_size_atr, 3),
            "ob_distance_atr":      round(ob_dist_atr, 3),
            "ob_age_bars":          ob_match.get("age_bars", 0),
            # FVG
            "fvg_present":          int(fvg_hit),
            "fvg_size_atr":         fvg_ctx["size_atr"],
            "fvg_entry_pct":        fvg_ctx["entry_pct"],
            # Sweep
            "sweep_amplitude_atr":  _sweep_amplitude_atr(m15, bias, atr_val),
            # Tendance / volatilité
            "h1_ema50_dist_atr":    round(ema50_dist_atr, 3),
            "h1_ema200_dist_atr":   round(ema200_dist_atr, 3),
            "vwap_distance_atr":    round(vwap_dist, 3),
            "h1_ema50_slope":       _ema50_slope(h1),
            "atr_entry":            round(atr_val, 4),
            # GP
            "gp_low":               round(gp["low"],   5),
            "gp_high":              round(gp["high"],  5),
            "gp_pct":               gp["pct_current"],
            # Trade params
            "sl_distance_atr":      round(risk / atr_val, 3) if atr_val > 0 else 0.0,
            "rr_target":            TP1_R,
            # Pour le système de patterns / logging
            "triggers":             triggers,
            "rsi_m5":               round(float(cur.get("rsi", 50) or 50), 1),
        },
    )
