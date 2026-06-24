"""
pretrain.py
===========
Pré-entraînement des systèmes d'apprentissage sur données historiques.

Rejoue les trades détectés par la stratégie sur une période passée et
alimente les 3 mécanismes d'apprentissage :
  1. Poids patterns  (Laplace smoothing)
  2. ML Gate         (régression logistique online)
  3. Seuils adaptatifs (ATR_MIN, EMA9 M5, EMA M15)

Le bot démarre ainsi avec la connaissance de centaines de trades
au lieu de partir de zéro.

Usage :
    from pretrain import run_pretrain
    result = run_pretrain("2024-01-01", "2024-12-31", symbol="XAUUSD")
"""

from __future__ import annotations

import threading
from datetime import datetime, timedelta, timezone
from typing import Dict, Any, Optional, Callable

import pandas as pd

from strategy import (
    add_indicators, evaluate, active_session,
    MAX_TRADE_MINUTES, SL_ATR_MULT, CET,
)
from strategy_ict import evaluate_ict
from backtest import load_m5_data, resample, _try_exit
import database as db
import data_provider as _dp
from ml_gate import OnlineLogisticRegression, AdaptiveThresholds, N_FEATURES

# --------------------------------------------------------------------------- #
# État de progression (partagé avec l'API)
# --------------------------------------------------------------------------- #
_progress: Dict[str, Any] = {
    "running":   False,
    "pct":       0,
    "bars_done": 0,
    "bars_total": 0,
    "trades":    0,
    "wins":      0,
    "status":    "idle",    # idle | running | done | error
    "last_result": None,
    "error":     None,
}
_lock = threading.Lock()


def get_progress() -> Dict[str, Any]:
    with _lock:
        return dict(_progress)


def _set(**kwargs):
    with _lock:
        _progress.update(kwargs)


# --------------------------------------------------------------------------- #
# Moteur de pré-entraînement
# --------------------------------------------------------------------------- #
MIN_LOT  = 0.01
LOT_STEP = 0.01


def _size_lots(capital: float, risk_pct: float, sl_distance: float,
               contract_size: float) -> float:
    """Même formule que le live : risk_amount / (sl_distance × contract_size)."""
    if sl_distance <= 0 or contract_size <= 0:
        return MIN_LOT
    risk_amount = capital * (risk_pct / 100.0)
    raw = risk_amount / (sl_distance * contract_size)
    steps = round(raw / LOT_STEP)
    return max(MIN_LOT, round(steps * LOT_STEP, 2))


def run_pretrain(
    start: str,
    end: str,
    symbol: str = "XAUUSD",
    atr_min: Optional[float] = None,
    reset: bool = True,
    capital: float = 1_000.0,
    risk_pct: float = 5.0,
    strategy_mode: str = "A",
) -> Dict[str, Any]:
    """
    Lance le pré-entraînement en mode bloquant.
    Appelé depuis un thread via launch_pretrain().

    capital / risk_pct : reproduisent le sizing du live (plus de lot 0.01 fixe).
    reset=True  : repart de zéro (recommandé la 1ère fois)
    reset=False : accumule sur l'historique existant
    """
    _set(running=True, pct=0, bars_done=0, trades=0, wins=0,
         status="running", error=None, last_result=None)

    contract_size = 100.0   if symbol == "XAUUSD" else 100000.0
    pip_size      = 0.1     if symbol == "XAUUSD" else 0.0001
    default_atr   = 3.0     if symbol == "XAUUSD" else 0.00030
    effective_atr = atr_min if atr_min is not None else default_atr
    spread    = 2.0  * pip_size   # $0.20 spread réaliste XAU/USD
    slippage  = 0.5  * pip_size   # $0.05 slippage

    try:
        # ---- Charger et préparer les données ----
        _set(status="Chargement des données…")
        m5_raw   = load_m5_data(start, end, symbol=symbol)
        if len(m5_raw) < 300:
            raise ValueError("Pas assez de données pour la période sélectionnée.")

        data_start_actual = m5_raw.index[0].isoformat()[:10]
        data_end_actual   = m5_raw.index[-1].isoformat()[:10]
        data_bars_total   = len(m5_raw)
        data_provider_errors = _dp.get_last_errors()

        m5       = add_indicators(m5_raw)
        m15_full = add_indicators(resample(m5_raw, "15min"))
        h1_full  = add_indicators(resample(m5_raw, "60min"))
        # ---- Initialiser les systèmes d'apprentissage ----
        if reset:
            gate     = OnlineLogisticRegression.__new__(OnlineLogisticRegression)
            gate.weights            = [0.0] * N_FEATURES
            gate.bias_w             = 0.0
            gate.n_samples          = 0
            gate.consecutive_losses = 0
            adaptive = AdaptiveThresholds.__new__(AdaptiveThresholds)
            adaptive.symbol          = symbol
            adaptive.atr_min_default = effective_atr
            adaptive.atr_min   = effective_atr
            adaptive.ema9_mult = 0.5
            adaptive.m15_mult  = 0.3
            adaptive.n_wins    = 0
            adaptive.n_losses  = 0
            adaptive.n_total   = 0
            db.reset_pattern_stats()
        else:
            gate     = OnlineLogisticRegression()
            adaptive = AdaptiveThresholds(atr_min_default=effective_atr, symbol=symbol)

        # ---- Boucle bar par bar ----
        warmup     = 210
        total      = len(m5) - warmup
        open_trade   = None
        n_trades     = 0
        n_wins       = 0
        equity       = float(capital)
        equity_curve = [{"ts": m5.index[warmup].isoformat(), "equity": equity}]
        n_false_stops        = 0   # SL direct → prix aurait atteint TP1 dans les 10 bougies suivantes
        n_sl_for_false_check = 0   # total SL directs analysés
        n_false_bes          = 0   # sl_after_tp1 → prix aurait atteint TP2 dans les 20 bougies suivantes
        n_be_for_false_check = 0   # total sl_after_tp1 analysés
        pnl_wins   = []   # PnL $ des trades gagnants
        pnl_losses = []   # PnL $ (abs) des trades perdants
        mae_wins   = []   # MAE en R des trades gagnants
        mfe_wins   = []   # MFE en R des trades gagnants
        mae_loss   = []   # MAE en R des trades perdants
        mfe_loss   = []   # MFE en R des trades perdants
        trades_log = []   # log détaillé par trade (pour analyse erreur/erreur)
        last_ict_asian_end = None  # verrou strat B : un seul trade par range asiatique

        _set(bars_total=total, status="Analyse des trades historiques…")

        for i in range(warmup, len(m5)):
            ts  = m5.index[i]
            bar = m5.iloc[i]

            # Progression
            done = i - warmup
            if done % 200 == 0:
                _set(pct=round(done / total * 100), bars_done=done,
                     trades=n_trades, wins=n_wins)

            # ---- Gérer le trade ouvert ----
            if open_trade is not None:
                exit_info = _try_exit(open_trade, bar, ts, slippage, contract_size)
                if exit_info is not None:
                    pnl, exit_price, exit_reason = exit_info
                    won = pnl > 0
                    n_trades += 1
                    equity += pnl
                    equity_curve.append({"ts": ts.isoformat(), "equity": round(equity, 2)})
                    if won:
                        n_wins += 1
                        pnl_wins.append(pnl)
                    else:
                        pnl_losses.append(abs(pnl))

                    # Collecter MAE/MFE en multiples de R
                    risk = open_trade.get("risk", 0.0)
                    mae_r = round(open_trade.get("mae", 0.0) / risk, 3) if risk > 0 else 0.0
                    mfe_r = round(open_trade.get("mfe", 0.0) / risk, 3) if risk > 0 else 0.0
                    if risk > 0:
                        if won:
                            mae_wins.append(mae_r)
                            mfe_wins.append(mfe_r)
                        else:
                            mae_loss.append(mae_r)
                            mfe_loss.append(mfe_r)

                    features = open_trade.get("ml_features")
                    if features:
                        gate.update(features, won)
                        adaptive.update(features, open_trade["entry"], won)

                    triggers = open_trade.get("triggers", [])
                    if triggers:
                        db.update_pattern_stats(triggers, won)

                    # ---- Analyse false stop ----
                    # Sur un SL direct : est-ce que le prix aurait atteint TP1
                    # dans les 10 bougies suivantes (50 min) ?
                    false_stop = False
                    false_stop_spike_atr = None
                    if exit_reason == "sl":
                        n_sl_for_false_check += 1
                        tp1_level  = open_trade["tp1"]
                        direction  = open_trade["direction"]
                        future     = m5.iloc[i + 1 : i + 11]
                        if direction == "long":
                            false_stop = bool((future["high"] >= tp1_level).any())
                        else:
                            false_stop = bool((future["low"] <= tp1_level).any())
                        if false_stop:
                            n_false_stops += 1
                        # Profondeur du spike : de combien (en ATR) le prix est-il
                        # allé au-delà du SL avant de revenir ?
                        _atr_snap = float(open_trade["ind_snap"].get("atr", 1.0) or 1.0)
                        _sl_lv = float(open_trade["stop_loss"])
                        _near = m5.iloc[i : i + 4]
                        if len(_near) > 0:
                            if direction == "long":
                                false_stop_spike_atr = round(
                                    max(0.0, (_sl_lv - float(_near["low"].min())) / _atr_snap), 3)
                            else:
                                false_stop_spike_atr = round(
                                    max(0.0, (float(_near["high"].max()) - _sl_lv) / _atr_snap), 3)
                        else:
                            false_stop_spike_atr = 0.0

                    # ---- Analyse false breakeven ----
                    # Sur un sl_after_tp1 : est-ce que le prix aurait atteint TP2
                    # dans les 20 bougies suivantes (100 min) ?
                    false_be = False
                    if exit_reason == "sl_after_tp1":
                        n_be_for_false_check += 1
                        tp2_level = open_trade["tp2"]
                        direction = open_trade["direction"]
                        future_be = m5.iloc[i + 1 : i + 21]
                        if direction == "long":
                            false_be = bool((future_be["high"] >= tp2_level).any())
                        else:
                            false_be = bool((future_be["low"] <= tp2_level).any())
                        if false_be:
                            n_false_bes += 1

                    _snap = open_trade.get("ind_snap", {})
                    trades_log.append({
                        "entry_ts":      open_trade["entry_time"].isoformat(),
                        "exit_ts":       ts.isoformat(),
                        "session":       open_trade.get("session", "?"),
                        "direction":     open_trade["direction"],
                        "entry":         round(open_trade["entry"], 3),
                        "exit_price":    round(float(exit_price), 3),
                        "exit_reason":   exit_reason,
                        "pnl":           round(pnl, 2),
                        "won":           won,
                        "mae_r":         mae_r,
                        "mfe_r":         mfe_r,
                        "patterns":      triggers,
                        "false_stop":          false_stop,
                        "false_stop_spike_atr": false_stop_spike_atr,
                        "false_be":            false_be,
                        "rsi_m5":        _snap.get("rsi_m5", 50),
                        "rsi_m15":       _snap.get("rsi_m15", 50),
                        "adx_h1":        _snap.get("adx_h1", 0),
                        "atr":           _snap.get("atr", 0),
                        "hour_cet":      _snap.get("hour_cet"),
                        "n_patterns":    _snap.get("n_patterns", 0),
                        "ema9_dist_r":   _snap.get("ema9_dist_r", 0),
                        "ema200_dist_r": _snap.get("ema200_dist_r", 0),
                        "vwap_side":     _snap.get("vwap_side", 0),
                        "h1_rsi":        _snap.get("h1_rsi", 50),
                        "body_ratio":    _snap.get("body_ratio", 0),
                    })

                    open_trade = None

            # ---- Chercher une entrée ----
            if open_trade is not None:
                continue

            # Pré-filtrage session rapide (évite 80 % des appels evaluate)
            sess = active_session(ts.to_pydatetime())
            if sess is None:
                continue

            m15_s = m15_full.iloc[:m15_full.index.searchsorted(ts, side="right")]
            h1_s  = h1_full.iloc[:h1_full.index.searchsorted(ts, side="right")]

            # evaluate() / evaluate_ict() SANS ml_gate ni adaptive → signal brut
            if strategy_mode == "B":
                ICT_M5_WINDOW = 576
                m5_win = m5.iloc[max(0, i - ICT_M5_WINDOW + 1): i + 1]
                sig = evaluate_ict(m5_win, m15_s, h1_s, now=ts.to_pydatetime(),
                                   atr_min=effective_atr)
                # Verrou : un seul trade par range asiatique
                if sig is not None:
                    asian_end = sig.meta.get("asian_end")
                    if asian_end is not None and asian_end == last_ict_asian_end:
                        sig = None
            else:
                sig = evaluate(
                    m5.iloc[:i + 1], m15_s, h1_s,
                    now=ts.to_pydatetime(),
                    check_session=True,
                    atr_min=effective_atr,
                )

            if sig is None:
                continue

            # Verrou strat B : mémoriser le range asiatique de ce trade
            if strategy_mode == "B":
                last_ict_asian_end = sig.meta.get("asian_end")

            fill = sig.entry + (spread + slippage) * (1 if sig.direction == "long" else -1)
            sl_dist = abs(fill - sig.stop_loss)
            volume = _size_lots(equity, risk_pct, sl_dist, contract_size)
            h1_cur  = h1_s.iloc[-1]  if len(h1_s)  > 0 else pd.Series(dtype=float)
            m15_cur = m15_s.iloc[-1] if len(m15_s) > 0 else pd.Series(dtype=float)
            open_trade = {
                "direction":    sig.direction,
                "session":      sess,
                "entry_time":   ts.to_pydatetime(),
                "entry":        fill,
                "stop_loss":    sig.stop_loss,
                "tp1":          sig.take_profit1,
                "tp2":          sig.take_profit2,
                "volume":       volume,
                "tp1_done":     False,
                "be_after_tp1": (strategy_mode == "B"),
                "remaining":    volume,
                "realised":     0.0,
                "max_exit_time": ts.to_pydatetime() + timedelta(minutes=sig.max_duration_min),
                "triggers":     sig.meta.get("triggers", []),
                "ml_features":  sig.meta.get("ml_features"),
                "risk":         sl_dist,
                "mae":          0.0,
                "mfe":          0.0,
                "ind_snap": {
                    "rsi_m5":       float(bar.get("rsi", 50) or 50),
                    "rsi_m15":      float(m15_cur.get("rsi", 50) or 50),
                    "adx_h1":       float(h1_cur.get("adx", 0) or 0),
                    "atr":          float(bar.get("atr", 0) or 0),
                    "hour_cet":     ts.astimezone(CET).hour,
                    "n_patterns":   len(sig.meta.get("triggers", [])),
                    "ema9_dist_r":  round(
                        abs(float(bar.get("close", 0) or 0) - float(bar.get("ema9", 0) or 0))
                        / max(float(bar.get("atr", 1) or 1), 0.001), 2
                    ),
                    "ema200_dist_r": round(
                        (float(h1_cur.get("close", 0) or 0) - float(h1_cur.get("ema200", 0) or 0))
                        / max(float(h1_cur.get("atr", 1) or 1), 0.001), 2
                    ),
                    "vwap_side": 1 if float(bar.get("close", 0) or 0) >= float(bar.get("vwap", bar.get("close", 0)) or 0) else 0,
                    "h1_rsi":    float(h1_cur.get("rsi", 50) or 50),
                    "body_ratio": round(
                        abs(float(bar.get("close", 0) or 0) - float(bar.get("open", 0) or 0))
                        / max(float(bar.get("atr", 1) or 1), 0.001), 2
                    ),
                },
            }

        # ---- WR par heure CET ----
        from collections import defaultdict as _dd
        _by_hour: dict = _dd(lambda: {"n": 0, "wins": 0})
        for _t in trades_log:
            _h = _t.get("hour_cet")
            if _h is not None:
                _by_hour[_h]["n"]    += 1
                _by_hour[_h]["wins"] += int(_t["won"])
        wr_by_hour = {
            str(_h): {"n": _v["n"], "wr": round(_v["wins"] / _v["n"], 3)}
            for _h, _v in sorted(_by_hour.items()) if _v["n"] >= 3
        }

        # ---- Profondeur des faux stops (spike ATR au-delà du SL) ----
        _spike_list = sorted([
            t["false_stop_spike_atr"]
            for t in trades_log
            if t.get("false_stop") and t.get("false_stop_spike_atr") is not None
        ])
        if _spike_list:
            _n_sp = len(_spike_list)
            def _perc(lst, p):
                return lst[max(0, min(len(lst) - 1, int(len(lst) * p / 100)))]
            false_stop_spike_stats = {
                "n":   _n_sp,
                "avg": round(sum(_spike_list) / _n_sp, 3),
                "p50": round(_perc(_spike_list, 50), 3),
                "p80": round(_perc(_spike_list, 80), 3),
                "p90": round(_perc(_spike_list, 90), 3),
                "coverage": {
                    str(thresh): round(
                        sum(1 for d in _spike_list if d <= thresh) / _n_sp * 100, 1
                    )
                    for thresh in [0.1, 0.2, 0.3, 0.5, 0.7, 1.0]
                },
            }
        else:
            false_stop_spike_stats = None

        # ---- % faux stops par heure CET ----
        _fs_by_hour: dict = _dd(lambda: {"n_sl": 0, "n_fs": 0})
        for _t in trades_log:
            if _t.get("exit_reason") == "sl":
                _h = _t.get("hour_cet")
                if _h is not None:
                    _fs_by_hour[_h]["n_sl"] += 1
                    if _t.get("false_stop"):
                        _fs_by_hour[_h]["n_fs"] += 1
        false_stop_by_hour = {
            str(_h): {
                "n_sl":          _v["n_sl"],
                "n_false_stops": _v["n_fs"],
                "pct_false":     round(_v["n_fs"] / _v["n_sl"] * 100, 1) if _v["n_sl"] else 0.0,
            }
            for _h, _v in sorted(_fs_by_hour.items()) if _v["n_sl"] >= 2
        }

        # ---- % faux stops par pattern ----
        _fs_by_pat: dict = _dd(lambda: {"n_sl": 0, "n_fs": 0})
        for _t in trades_log:
            if _t.get("exit_reason") == "sl":
                for _p in _t.get("patterns", []):
                    _fs_by_pat[_p]["n_sl"] += 1
                    if _t.get("false_stop"):
                        _fs_by_pat[_p]["n_fs"] += 1
        false_stop_by_pattern = {
            _p: {
                "n_sl":          _v["n_sl"],
                "n_false_stops": _v["n_fs"],
                "pct_false":     round(_v["n_fs"] / _v["n_sl"] * 100, 1) if _v["n_sl"] else 0.0,
            }
            for _p, _v in sorted(_fs_by_pat.items(), key=lambda x: -x[1]["n_sl"])
            if _v["n_sl"] >= 2
        }

        # ---- Diagnostic indicateurs par outcome ----
        def _grp_means(log, outcome):
            grp = [t for t in log if t.get("exit_reason") == outcome]
            if not grp:
                return None
            def _mean(key, default=0):
                vals = [t[key] for t in grp if t.get(key) is not None]
                return round(sum(vals) / len(vals), 2) if vals else default
            return {
                "n":             len(grp),
                "rsi_m5":        round(_mean("rsi_m5", 50), 1),
                "rsi_m15":       round(_mean("rsi_m15", 50), 1),
                "adx_h1":        round(_mean("adx_h1", 0), 1),
                "atr":           round(_mean("atr", 0), 2),
                "n_patterns":    round(_mean("n_patterns", 0), 1),
                "ema9_dist_r":   round(_mean("ema9_dist_r", 0), 2),
                "ema200_dist_r": round(_mean("ema200_dist_r", 0), 2),
                "vwap_above_pct": round(
                    sum(1 for t in grp if t.get("vwap_side") == 1) / len(grp) * 100, 1
                ),
                "h1_rsi":    round(_mean("h1_rsi", 50), 1),
                "body_ratio": round(_mean("body_ratio", 0), 2),
                "london_pct": round(
                    sum(1 for t in grp if t.get("session") == "London") / len(grp) * 100, 1
                ),
            }

        _diag_outcomes = [("sl", "SL_direct"), ("tp2", "TP2"),
                          ("sl_after_tp1", "SL_TP1"), ("timeout", "Timeout")]
        indicator_diagnostic = {
            label: _grp_means(trades_log, key)
            for key, label in _diag_outcomes
            if _grp_means(trades_log, key) is not None
        }

        _sl  = indicator_diagnostic.get("SL_direct", {})
        _tp2 = indicator_diagnostic.get("TP2", {})
        print("\n=== DIAGNOSTIC INDICATEURS PAR OUTCOME ===")
        print(f"{'Indicateur':<14}{'SL_direct':>10}{'TP2':>10}{'Δ(SL-TP2)':>12}")
        print("-" * 46)
        for _k, _lbl in [("rsi_m5","RSI M5"), ("rsi_m15","RSI M15"),
                          ("adx_h1","ADX H1"), ("atr","ATR"),
                          ("h1_rsi","RSI H1"), ("body_ratio","Corps/ATR"),
                          ("london_pct","London%")]:
            _sv = _sl.get(_k, "?")
            _tv = _tp2.get(_k, "?")
            _delta = f"{_sv - _tv:+.1f}" if isinstance(_sv, float) and isinstance(_tv, float) else ""
            print(f"{_lbl:<14}{str(_sv):>10}{str(_tv):>10}{_delta:>12}")
        print(f"  N: SL_direct={_sl.get('n','?')}, TP2={_tp2.get('n','?')}, "
              f"SL_TP1={indicator_diagnostic.get('SL_TP1',{}).get('n','?')}")
        print("==========================================\n")

        # ---- Persister les modèles appris ----
        db.save_ml_weights(gate.weights, gate.bias_w, gate.n_samples,
                           consecutive_losses=gate.consecutive_losses)
        db.save_adaptive_thresholds(symbol, {
            "atr_min":   adaptive.atr_min,
            "ema9_mult": adaptive.ema9_mult,
            "m15_mult":  adaptive.m15_mult,
            "n_wins":    adaptive.n_wins,
            "n_losses":  adaptive.n_losses,
            "n_total":   adaptive.n_total,
        })

        win_rate = round(n_wins / n_trades, 3) if n_trades else 0.0

        import statistics as _stats
        def _avg(lst): return round(_stats.mean(lst), 3) if lst else 0.0

        gross_profit = round(sum(pnl_wins),   2)
        gross_loss   = round(sum(pnl_losses), 2)
        net_pnl      = round(gross_profit - gross_loss, 2)
        profit_factor = round(gross_profit / gross_loss, 3) if gross_loss else 0.0
        avg_win  = round(_stats.mean(pnl_wins),   2) if pnl_wins   else 0.0
        avg_loss = round(_stats.mean(pnl_losses), 2) if pnl_losses else 0.0

        result = {
            "n_trades":      n_trades,
            "n_wins":        n_wins,
            "win_rate":      win_rate,
            "gross_profit":  gross_profit,
            "gross_loss":    gross_loss,
            "net_pnl":       net_pnl,
            "profit_factor": profit_factor,
            "avg_win":       avg_win,
            "avg_loss":      avg_loss,
            "period":        f"{start} → {end}",
            "data_coverage": {
                "requested_start": start,
                "requested_end":   end,
                "actual_start":    data_start_actual,
                "actual_end":      data_end_actual,
                "bars":            data_bars_total,
                "full_coverage":   data_start_actual <= start,
                "provider_errors": data_provider_errors or None,
            },
            "symbol":        symbol,
            "atr_min_final": round(adaptive.atr_min, 4),
            "ema9_mult_final": round(adaptive.ema9_mult, 3),
            "m15_mult_final":  round(adaptive.m15_mult, 3),
            "ml_samples":    gate.n_samples,
            "equity_curve":  equity_curve,
            "trades_log":    trades_log,
            "excursion": {
                "avg_mae_r_wins": _avg(mae_wins),
                "avg_mfe_r_wins": _avg(mfe_wins),
                "avg_mae_r_loss": _avg(mae_loss),
                "avg_mfe_r_loss": _avg(mfe_loss),
                # % pertes qui avaient atteint 0.5R favorable → "near-wins"
                "pct_loss_mfe_gt_half_r": round(
                    sum(1 for v in mfe_loss if v >= 0.5) / len(mfe_loss) * 100, 1
                ) if mfe_loss else 0.0,
                # % gains qui ont subi >0.5R adverse → entrée trop tôt
                "pct_win_mae_gt_half_r": round(
                    sum(1 for v in mae_wins if v >= 0.5) / len(mae_wins) * 100, 1
                ) if mae_wins else 0.0,
            },
            "false_stops": {
                "n_sl_direct":   n_sl_for_false_check,
                "n_false_stops": n_false_stops,
                "pct_false_stops": round(
                    n_false_stops / n_sl_for_false_check * 100, 1
                ) if n_sl_for_false_check else 0.0,
            },
            "false_breakevens": {
                "n_sl_after_tp1": n_be_for_false_check,
                "n_false_bes":    n_false_bes,
                # % de sl_after_tp1 où le prix aurait atteint TP2 dans les 20 bougies suivantes
                "pct_false_bes": round(
                    n_false_bes / n_be_for_false_check * 100, 1
                ) if n_be_for_false_check else 0.0,
            },
            "indicator_diagnostic":   indicator_diagnostic,
            "wr_by_hour":             wr_by_hour,
            "false_stop_spike_stats": false_stop_spike_stats,
            "false_stop_by_hour":     false_stop_by_hour,
            "false_stop_by_pattern":  false_stop_by_pattern,
        }
        _set(running=False, pct=100, bars_done=total, trades=n_trades,
             wins=n_wins, status="done", last_result=result)
        return result

    except Exception as exc:
        _set(running=False, status="error", error=str(exc))
        raise


# --------------------------------------------------------------------------- #
# Lancement asynchrone (appelé depuis l'API)
# --------------------------------------------------------------------------- #
def launch_pretrain(
    start: str,
    end: str,
    symbol: str = "XAUUSD",
    atr_min: Optional[float] = None,
    reset: bool = True,
    capital: float = 1_000.0,
    risk_pct: float = 5.0,
    strategy_mode: str = "A",
    on_complete=None,
) -> None:
    """Lance le pré-entraînement dans un thread daemon (non-bloquant)."""
    if get_progress()["running"]:
        return

    # Pre-set running=True before the thread starts so the API response is
    # consistent even if the thread hasn't had a chance to run yet.
    _set(running=True, pct=0, bars_done=0, bars_total=0, trades=0, wins=0,
         status="Démarrage…", error=None, last_result=None)

    def _run():
        try:
            run_pretrain(start, end, symbol=symbol, atr_min=atr_min, reset=reset,
                         capital=capital, risk_pct=risk_pct, strategy_mode=strategy_mode)
            if on_complete:
                on_complete()
        except Exception as exc:
            _set(running=False, status="error", error=str(exc))

    threading.Thread(target=_run, daemon=True, name="pretrain").start()
