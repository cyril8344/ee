"""
trade_audit.py
===============
Diagnostic post-mortem des trades clôturés : pour chaque trade, résume
pourquoi il est entré (biais, session, indicateurs/OB au moment du signal)
et pourquoi il est sorti (exit_reason, durée, P&L).

Flague en plus, pour les sorties "sl_after_tp1", la signature du bug
corrigé le 29/07/2026 (broker.py::update_position revérifiait le SL sur
la MÊME bougie M5 que celle qui venait de déclencher TP1 — cf. PR #310).
La seule façon fiable de distinguer un vrai retour au breakeven d'un faux
déclenchement est de rejouer les bougies M5 réelles de l'époque du trade,
donc ce module a besoin d'un accès data (Twelve Data / yfinance) — sans
ça les trades "sl_after_tp1" sont marqués "data_unavailable", pas comptés
comme faux positifs par défaut.
"""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

import pandas as pd

import database as db
from backtest import load_m5_data

REPLAY_LOOKAHEAD_BARS = 20  # ~1h40 de bougies M5 après TP1 pour chercher TP2


def _contract_size(symbol: str) -> float:
    return 100.0 if symbol == "XAUUSD" else 100000.0


def _parse_ts(raw: Optional[str]) -> Optional[pd.Timestamp]:
    if not raw:
        return None
    try:
        ts = pd.Timestamp(raw)
        if ts.tzinfo is None:
            ts = ts.tz_localize("UTC")
        return ts
    except Exception:
        return None


def _entry_summary(t: Dict[str, Any]) -> Dict[str, Any]:
    meta = t.get("meta") or {}
    if isinstance(meta, str):
        import json
        try:
            meta = json.loads(meta)
        except Exception:
            meta = {}
    summary = {
        "direction": t.get("direction"),
        "session": t.get("session"),
    }
    # Strat A : rsi_m5/rsi_m15 dans meta. Strat B (ICT) : ob_low/high, adx_h1, sr_zone.
    for k in ("rsi_m5", "rsi_m15", "ob_low", "ob_high", "adx_h1", "sr_zone", "strategy"):
        if k in meta:
            summary[k] = meta[k]
    return summary


def _replay_sl_after_tp1(t: Dict[str, Any], symbol: str) -> Dict[str, Any]:
    """Rejoue les bougies M5 réelles autour d'un trade 'sl_after_tp1' pour
    déterminer si la sortie est un vrai retour au breakeven ou la signature
    du bug (bougie de TP1 dont l'autre extrémité dépasse déjà le breakeven)."""
    ts_entry = _parse_ts(t.get("entry_time"))
    ts_exit = _parse_ts(t.get("exit_time"))
    if ts_entry is None:
        return {"status": "data_unavailable", "reason": "no_entry_time"}

    direction = t.get("direction")
    entry = float(t.get("entry_price") or 0)
    tp1 = float(t.get("take_profit1") or 0)
    tp2 = float(t.get("take_profit2") or 0)
    if not (entry and tp1 and tp2):
        return {"status": "data_unavailable", "reason": "missing_levels"}

    window_end = (ts_exit or ts_entry) + timedelta(minutes=REPLAY_LOOKAHEAD_BARS * 5 + 30)
    start_str = (ts_entry - timedelta(minutes=10)).strftime("%Y-%m-%d")
    end_str = window_end.strftime("%Y-%m-%d")
    try:
        m5 = load_m5_data(start_str, end_str, symbol=symbol)
    except Exception as exc:
        return {"status": "data_unavailable", "reason": f"fetch_error: {exc}"}
    if m5 is None or len(m5) == 0 or m5.attrs.get("provider") == "synthetic":
        return {"status": "data_unavailable", "reason": "no_real_data"}

    bars = m5[(m5.index >= ts_entry) & (m5.index <= window_end)]
    if len(bars) == 0:
        return {"status": "data_unavailable", "reason": "no_bars_in_window"}

    # 1) Trouver la bougie qui touche TP1 en premier
    tp1_idx = None
    for i, (ts, bar) in enumerate(bars.iterrows()):
        hit = (bar["low"] <= tp1) if direction == "short" else (bar["high"] >= tp1)
        if hit:
            tp1_idx = i
            break
    if tp1_idx is None:
        return {"status": "inconclusive", "reason": "tp1_not_found_in_replay"}

    tp1_bar = bars.iloc[tp1_idx]
    # 2) Signature du bug : la MÊME bougie a déjà, de l'autre côté, dépassé
    #    le breakeven (= entry, BE_BUFFER_R=0) — impossible à distinguer d'un
    #    vrai retour sans savoir si le haut/bas est survenu avant ou après le
    #    passage par TP1 intra-bougie.
    same_bar_be_touch = (
        (tp1_bar["high"] >= entry) if direction == "short" else (tp1_bar["low"] <= entry)
    )

    # 3) Rejeu "propre" (comme backtest.py) : à partir de la bougie SUIVANTE,
    #    est-ce que TP2 est atteint avant qu'un vrai retour au breakeven ne
    #    se produise ?
    would_reach_tp2 = False
    genuine_be_touch = False
    for j in range(tp1_idx + 1, min(tp1_idx + 1 + REPLAY_LOOKAHEAD_BARS, len(bars))):
        bar = bars.iloc[j]
        hit_tp2 = (bar["low"] <= tp2) if direction == "short" else (bar["high"] >= tp2)
        hit_be = (bar["high"] >= entry) if direction == "short" else (bar["low"] <= entry)
        if hit_tp2:
            would_reach_tp2 = True
            break
        if hit_be:
            genuine_be_touch = True
            break

    bug_signature = bool(same_bar_be_touch and not genuine_be_touch)

    return {
        "status": "ok",
        "bug_signature": bug_signature,
        "would_reach_tp2": would_reach_tp2,
        "genuine_be_touch_later": genuine_be_touch,
    }


def _corrected_pnl(t: Dict[str, Any], symbol: str) -> float:
    """P&L estimé si le reliquat (après TP1) avait été laissé courir jusqu'à TP2
    au lieu d'être coupé au breakeven. Approximation : ignore le slippage sur
    la jambe BE réelle (déjà quasi nul par construction, BE_BUFFER_R=0)."""
    direction = t.get("direction")
    sign = 1.0 if direction == "long" else -1.0
    entry = float(t.get("entry_price") or 0)
    tp2 = float(t.get("take_profit2") or 0)
    volume = float(t.get("volume") or 0)
    meta = t.get("meta") or {}
    if isinstance(meta, str):
        import json
        try:
            meta = json.loads(meta)
        except Exception:
            meta = {}
    close_ratio = 1.0 if meta.get("tp1_close_all") else 0.5
    lots_tp1 = round(volume * close_ratio, 2)
    remaining = round(volume - lots_tp1, 2)
    extra = (tp2 - entry) * sign * _contract_size(symbol) * remaining
    return round((t.get("pnl") or 0.0) + extra, 2)


def audit_trades(start: str, end: str, symbol: str = "XAUUSD",
                  replay_sl_after_tp1: bool = True) -> Dict[str, Any]:
    """Diagnostic complet entrée/sortie pour tous les trades clôturés de
    [start, end] (YYYY-MM-DD, UTC sur entry_time), avec vérification du bug
    'SL après TP1' quand replay_sl_after_tp1=True (nécessite un accès data réel)."""
    report = db.get_trade_report(limit=5000, symbol=symbol)
    all_trades = report.get("trades", [])

    out: List[Dict[str, Any]] = []
    for t in all_trades:
        et = (t.get("entry_time") or "")[:10]
        if et < start or et > end:
            continue
        row = {
            "id": t.get("id"),
            "entry_time": t.get("entry_time"),
            "exit_time": t.get("exit_time"),
            "duration_min": t.get("duration_min"),
            "entry_price": t.get("entry_price"),
            "exit_price": t.get("exit_price"),
            "pnl": t.get("pnl"),
            "exit_reason": t.get("exit_reason"),
            "entry": _entry_summary(t),
        }
        if replay_sl_after_tp1 and t.get("exit_reason") == "sl_after_tp1":
            replay = _replay_sl_after_tp1(t, symbol)
            row["bug_check"] = replay
            if replay.get("status") == "ok" and replay.get("would_reach_tp2"):
                row["corrected_pnl"] = _corrected_pnl(t, symbol)
        out.append(row)

    # Agrégats
    n_sl_after_tp1 = sum(1 for r in out if r["exit_reason"] == "sl_after_tp1")
    n_bug_flagged = sum(1 for r in out if r.get("bug_check", {}).get("bug_signature"))
    n_would_tp2 = sum(1 for r in out if r.get("bug_check", {}).get("would_reach_tp2"))
    n_data_unavailable = sum(
        1 for r in out
        if r["exit_reason"] == "sl_after_tp1" and r.get("bug_check", {}).get("status") == "data_unavailable"
    )
    pnl_actual = round(sum(r["pnl"] or 0.0 for r in out), 2)
    pnl_delta = round(sum(
        (r.get("corrected_pnl", r["pnl"]) or 0.0) - (r["pnl"] or 0.0) for r in out
    ), 2)

    by_week: Dict[str, Dict[str, float]] = defaultdict(lambda: {"pnl_actual": 0.0, "pnl_corrected": 0.0, "n": 0})
    for r in out:
        et = _parse_ts(r["entry_time"])
        if et is None:
            continue
        wk = f"{et.isocalendar().year}-W{et.isocalendar().week:02d}"
        by_week[wk]["n"] += 1
        by_week[wk]["pnl_actual"] += r["pnl"] or 0.0
        by_week[wk]["pnl_corrected"] += r.get("corrected_pnl", r["pnl"]) or 0.0
    for wk in by_week:
        by_week[wk]["pnl_actual"] = round(by_week[wk]["pnl_actual"], 2)
        by_week[wk]["pnl_corrected"] = round(by_week[wk]["pnl_corrected"], 2)

    return {
        "start": start, "end": end, "symbol": symbol,
        "n_trades": len(out),
        "n_sl_after_tp1": n_sl_after_tp1,
        "n_bug_flagged": n_bug_flagged,
        "n_would_have_reached_tp2": n_would_tp2,
        "n_data_unavailable": n_data_unavailable,
        "pnl_actual": pnl_actual,
        "pnl_corrected_estimate": round(pnl_actual + pnl_delta, 2),
        "pnl_delta_from_bug": pnl_delta,
        "by_week": dict(by_week),
        "trades": out,
    }
