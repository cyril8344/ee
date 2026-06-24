"""
strategy_ict.py — Stratégie B : AMD + FVG
==========================================
Accumulation (Asian session) → Manipulation (London sweep) → Distribution (FVG retest)

Pipeline :
1. Accumulation  = Session Asiatique 22:00–07:00 UTC → high/low identifiés
2. Manipulation  = Session London 07:00–12:00 UTC → sweep du high ou low asiatique
   - Sweep du LOW  → setup LONG (liquidité aspirée, renversement haussier attendu)
   - Sweep du HIGH → setup SHORT (liquidité aspirée, renversement baissier attendu)
3. Distribution  = Retest d'un FVG formé après le sweep → entrée dans la direction inverse
   SL  = derrière l'extrême du sweep (+ buffer)
   TP1 = 1.5R, TP2 = 3.0R
"""

from __future__ import annotations

from datetime import datetime, date, time, timezone, timedelta
from typing import Optional, Dict, Any, List

import pandas as pd

from strategy import Signal, active_session, is_bad_timing, ATR_MIN

# ──────────────────────────────────────────────────────────────────────────────
# Paramètres
# ──────────────────────────────────────────────────────────────────────────────
ASIAN_START_H   = 22    # 22:00 UTC (veille)
ASIAN_END_H     = 7     # 07:00 UTC
MANIP_START_H   = 7     # début manipulation London
MANIP_END_H     = 12    # fin manipulation London
SWEEP_BUFFER    = 0.15  # en ATR — marge pour confirmer le sweep
SL_BUFFER_ATR   = 0.3   # buffer SL au-delà de l'extrême du sweep
MAX_SL_ATR      = 4.0   # plafond SL pour éviter des stops géants
TP1_R           = 1.5
TP2_R           = 3.0
MAX_TRADE_MIN   = 90    # durée max en minutes
FVG_MIN_ATR     = 0.15  # taille minimum d'un FVG (en ATR)
FVG_MAX_BARS    = 60    # on ignore les FVGs formés il y a > 60 bougies M5 (= 5h)
MIN_ASIAN_BARS  = 4     # bougies minimum pour valider le range asiatique


# ──────────────────────────────────────────────────────────────────────────────
# 1. Range asiatique
# ──────────────────────────────────────────────────────────────────────────────
def _asian_range(df: pd.DataFrame, now: datetime) -> Optional[Dict[str, Any]]:
    """High/low de la session asiatique (22:00–07:00 UTC) précédant 'now'."""
    if df.index.tz is None:
        return None

    now_utc = now.astimezone(timezone.utc)
    today   = now_utc.date()

    asian_end   = datetime.combine(today, time(ASIAN_END_H, 0), tzinfo=timezone.utc)
    asian_start = datetime.combine(today - timedelta(days=1),
                                   time(ASIAN_START_H, 0), tzinfo=timezone.utc)

    # Avant 07:00 UTC, décaler d'un jour
    if now_utc.hour < ASIAN_END_H:
        asian_end   -= timedelta(days=1)
        asian_start -= timedelta(days=1)

    bars = df[(df.index >= asian_start) & (df.index < asian_end)]
    if len(bars) < MIN_ASIAN_BARS:
        return None

    return {
        "high":  float(bars["high"].max()),
        "low":   float(bars["low"].min()),
        "start": asian_start,
        "end":   asian_end,
    }


# ──────────────────────────────────────────────────────────────────────────────
# 2. Détection du sweep (manipulation London)
# ──────────────────────────────────────────────────────────────────────────────
def _detect_sweep(
    df: pd.DataFrame,
    asian: Dict[str, Any],
    now: datetime,
    atr_val: float,
) -> Optional[Dict[str, Any]]:
    """
    Retourne {"direction": "LONG"|"SHORT", "extreme": price, "ts": timestamp}
    - LONG  : low asiatique cassé (chasse de la liquidité basse → rebond haussier)
    - SHORT : high asiatique cassé (chasse de la liquidité haute → rebond baissier)
    Si les deux sont cassés, priorité au sweep le plus récent.
    """
    now_utc = now.astimezone(timezone.utc)
    today   = now_utc.date()

    m_start = datetime.combine(today, time(MANIP_START_H, 0), tzinfo=timezone.utc)
    m_end   = datetime.combine(today, time(MANIP_END_H, 0), tzinfo=timezone.utc)

    if now_utc < m_start:
        return None  # manipulation pas encore commencée

    london = df[(df.index >= m_start) & (df.index < min(m_end, now_utc))]
    if len(london) == 0:
        return None

    tol = SWEEP_BUFFER * atr_val

    low_bars  = london[london["low"]  < asian["low"]  - tol]
    high_bars = london[london["high"] > asian["high"] + tol]

    swept_low  = len(low_bars)  > 0
    swept_high = len(high_bars) > 0

    if not swept_low and not swept_high:
        return None

    # Double sweep → priorité au plus récent
    if swept_low and swept_high:
        ts_low  = low_bars.index[-1]
        ts_high = high_bars.index[-1]
        if ts_low > ts_high:
            swept_high = False
        else:
            swept_low = False

    if swept_low:
        return {
            "direction": "LONG",
            "extreme":   float(low_bars["low"].min()),
            "ts":        low_bars.index[-1],
        }
    else:
        return {
            "direction": "SHORT",
            "extreme":   float(high_bars["high"].max()),
            "ts":        high_bars.index[-1],
        }


# ──────────────────────────────────────────────────────────────────────────────
# 3. FVGs après le sweep
# ──────────────────────────────────────────────────────────────────────────────
def _fvgs_after_sweep(
    df: pd.DataFrame,
    sweep_ts: Any,
    direction: str,
    atr_val: float,
) -> List[Dict[str, float]]:
    """
    FVGs formés après le sweep, dans la direction du trade.
    FVG haussier (LONG)  : bar[i+2].low  > bar[i].high  → zone de support
    FVG baissier (SHORT) : bar[i+2].high < bar[i].low   → zone de résistance
    """
    post = df[df.index > sweep_ts].tail(FVG_MAX_BARS)
    if len(post) < 3:
        return []

    min_size = FVG_MIN_ATR * atr_val
    fvgs: List[Dict[str, float]] = []

    for i in range(len(post) - 2):
        b1 = post.iloc[i]
        b3 = post.iloc[i + 2]

        if direction == "LONG":
            gap_lo = float(b1["high"])
            gap_hi = float(b3["low"])
            if gap_hi > gap_lo and (gap_hi - gap_lo) >= min_size:
                fvgs.append({"low": gap_lo, "high": gap_hi})
        else:
            gap_hi = float(b1["low"])
            gap_lo = float(b3["high"])
            if gap_hi > gap_lo and (gap_hi - gap_lo) >= min_size:
                fvgs.append({"low": gap_lo, "high": gap_hi})

    return fvgs


def _in_fvg(bar_low: float, bar_high: float, fvgs: List[Dict]) -> bool:
    """True si la bougie actuelle touche un FVG (wick ou corps)."""
    for fvg in fvgs:
        if bar_low <= fvg["high"] and bar_high >= fvg["low"]:
            return True
    return False


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
    """
    Stratégie B — AMD + FVG.
    Nom gardé 'evaluate_ict' pour compatibilité avec main.py et pretrain.py.
    """
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

    # 3) Range asiatique
    asian = _asian_range(m5, ts)
    if asian is None:
        return None

    # 4) Sweep London (manipulation)
    sweep = _detect_sweep(m5, asian, ts, atr_val)
    if sweep is None:
        return None

    direction = sweep["direction"]

    # 5) FVGs formés après le sweep
    fvgs = _fvgs_after_sweep(m5, sweep["ts"], direction, atr_val)
    if not fvgs:
        return None

    # 6) Prix actuel dans un FVG → entrée
    if not _in_fvg(float(cur["low"]), float(cur["high"]), fvgs):
        return None

    # 7) Niveaux du trade
    entry = float(cur["close"])

    if direction == "LONG":
        raw_sl = min(sweep["extreme"], asian["low"]) - SL_BUFFER_ATR * atr_val
        sl = max(raw_sl, entry - MAX_SL_ATR * atr_val)
        tp1 = entry + TP1_R * abs(entry - sl)
        tp2 = entry + TP2_R * abs(entry - sl)
    else:
        raw_sl = max(sweep["extreme"], asian["high"]) + SL_BUFFER_ATR * atr_val
        sl = min(raw_sl, entry + MAX_SL_ATR * atr_val)
        tp1 = entry - TP1_R * abs(entry - sl)
        tp2 = entry - TP2_R * abs(entry - sl)

    risk = abs(entry - sl)
    if risk <= 0:
        return None

    return Signal(
        direction=direction.lower(),
        bias=direction,
        session=session,
        entry=entry,
        stop_loss=sl,
        take_profit1=tp1,
        take_profit2=tp2,
        atr=atr_val,
        reason="AMD_FVG",
        risk_distance=risk,
        timestamp=ts,
        meta={
            "strategy":   "B_AMD",
            "asian_high": round(asian["high"], 5),
            "asian_low":  round(asian["low"],  5),
            "sweep_dir":  direction,
            "sweep_ext":  round(sweep["extreme"], 5),
            "n_fvgs":     len(fvgs),
        },
    )
