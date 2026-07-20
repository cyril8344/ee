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

import strategy
from strategy import (
    add_indicators, evaluate, active_session, compute_bias,
    MAX_TRADE_MINUTES, SL_ATR_MULT, CET,
)
from strategy_ict import evaluate_ict
from backtest import load_m5_data, resample, _try_exit
import database as db
import data_provider as _dp
from ml_gate import AdaptiveThresholds

# --------------------------------------------------------------------------- #
# État de progression (partagé avec l'API)
# --------------------------------------------------------------------------- #
_progress: Dict[str, Any] = {
    "running":      False,
    "pct":          0,
    "bars_done":    0,
    "bars_total":   0,
    "trades":       0,
    "wins":         0,
    "status":       "idle",    # idle | running | done | error
    "last_result":  None,
    "strategy_mode": None,
    "error":        None,
}
# Derniers résultats complétés par stratégie (persistent entre les lancements)
_last_by_strategy: Dict[str, Any] = {"A": None, "B": None}
_lock = threading.Lock()


def get_progress() -> Dict[str, Any]:
    with _lock:
        return dict(_progress)


def get_last_results() -> Dict[str, Any]:
    with _lock:
        return dict(_last_by_strategy)


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
    capital: float = 10_000.0,
    risk_pct: float = 5.0,
    strategy_mode: str = "A",
    extra_overrides: Optional[Dict[str, Any]] = None,
    write_to_db: bool = True,
) -> Dict[str, Any]:
    """
    Lance le pré-entraînement en mode bloquant.
    Appelé depuis un thread via launch_pretrain().

    capital / risk_pct : reproduisent le sizing du live (plus de lot 0.01 fixe).
    reset=True  : repart de zéro (recommandé la 1ère fois)
    reset=False : accumule sur l'historique existant
    """
    _set(running=True, pct=0, bars_done=0, trades=0, wins=0,
         status="running", error=None, last_result=None, strategy_mode=strategy_mode)

    # Pendant le pretrain : désactiver BOOTSTRAP_MODE ET restaurer les seuils réels.
    # Sans ça, tous les filtres sont à 0 (valeurs bootstrap) → 5000+ trades bruités
    # → ML Gate apprend du bruit au lieu de vrais signaux de qualité (~200-300 trades).
    _PRETRAIN_OVERRIDES = {
        "BOOTSTRAP_MODE":        False,
        "ATR_MIN":               2.5,
        "ADX_MIN":               20.0,
        "RSI_M5_LONG_MIN":       45.0,
        "RSI_M5_SHORT_MAX":      55.0,
        "PATTERN_FLOOR":         0.67,
        "MIN_WEIGHT_SUM_LONG":   1.0,
        "ATR_REGIME_MIN_RATIO":  0.65,
        "TREND_BIAS_DISTANCE":   0.5,
        "M15_FILTER_ENABLED":    True,
        "EMA9_FILTER_ENABLED":   True,
        "VWAP_FILTER_ENABLED":   True,
        "BAD_HOURS_CET":         {8, 10, 14},
    }
    if extra_overrides:
        _PRETRAIN_OVERRIDES.update(extra_overrides)
    _saved_strategy = {k: getattr(strategy, k) for k in _PRETRAIN_OVERRIDES}
    for k, v in _PRETRAIN_OVERRIDES.items():
        setattr(strategy, k, v)

    contract_size = 100.0   if symbol == "XAUUSD" else 100000.0
    pip_size      = 0.1     if symbol == "XAUUSD" else 0.0001
    default_atr   = 3.0     if symbol == "XAUUSD" else 0.00030
    effective_atr = atr_min if atr_min is not None else default_atr
    if symbol == "XAUUSD":
        spread   = 2.0  * pip_size   # 2.0 pips XAU/USD = 0.20 (réaliste)
        slippage = 0.5  * pip_size   # 0.5 pips XAU/USD = 0.05
    else:
        spread   = 0.2  * pip_size   # 0.2 pips EUR/USD = 0.00002 (identique broker live)
        slippage = 0.05 * pip_size   # 0.05 pips EUR/USD = 0.000005

    try:
        # ---- Charger et préparer les données ----
        _set(status="Chargement des données…")
        m5_raw   = load_m5_data(start, end, symbol=symbol)
        if len(m5_raw) < 300:
            raise ValueError("Pas assez de données pour la période sélectionnée.")

        data_start_actual = m5_raw.index[0].isoformat()[:10]
        data_end_actual   = m5_raw.index[-1].isoformat()[:10]
        data_bars_total   = len(m5_raw)
        data_provider_used = m5_raw.attrs.get("provider", "unknown")
        try:
            data_end_gap_days = (pd.Timestamp(end) - pd.Timestamp(data_end_actual)).days
        except Exception:
            data_end_gap_days = 0
        data_provider_errors = _dp.get_last_errors()

        m5       = add_indicators(m5_raw)
        m15_full = add_indicators(resample(m5_raw, "15min"))
        h1_full  = add_indicators(resample(m5_raw, "60min"))
        h4_full  = add_indicators(resample(m5_raw, "240min"))
        # ---- Initialiser les systèmes d'apprentissage ----
        # Les instances pretrain ne sauvegardent JAMAIS en DB (isolation totale)
        # pour éviter que l'entraînement sur données historiques écrase ce que
        # le live a appris des vrais trades paper.
        _noop_save = lambda: None

        if reset:
            adaptive = AdaptiveThresholds.__new__(AdaptiveThresholds)
            adaptive.symbol          = symbol
            adaptive.atr_min_default = effective_atr
            adaptive.atr_min   = effective_atr
            adaptive.ema9_mult = 0.5
            adaptive.m15_mult  = 0.3
            adaptive.n_wins    = 0
            adaptive.n_losses  = 0
            adaptive.n_total   = 0
        else:
            adaptive = AdaptiveThresholds(atr_min_default=effective_atr, symbol=symbol)

        # Isolation : les updates restent en mémoire, jamais persistés en DB
        adaptive._save = _noop_save

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
        last_ob_ts = None  # verrou strat B : un seul trade par OB
        rejection_counts: Dict[str, int] = {}  # compter les rejets par étape pipeline
        # Limite journalière (prétrain : 10 trades/jour max)
        max_trades_day = 10
        _day_trades: Dict[str, int] = {}  # date_str → nb trades ce jour

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
                        adaptive.update(features, open_trade["entry"], won)

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
                        "false_stop":           false_stop,
                        "false_stop_spike_atr": false_stop_spike_atr,
                        "false_be":             false_be,
                        "rsi_m5":        _snap.get("rsi_m5", 50),
                        "rsi_m15":       _snap.get("rsi_m15", 50),
                        "adx_h1":        _snap.get("adx_h1", 0),
                        "atr":           _snap.get("atr", 0),
                        "hour_cet":      _snap.get("hour_cet"),
                        "ema9_dist_r":   _snap.get("ema9_dist_r", 0),
                        "ema200_dist_r": _snap.get("ema200_dist_r", 0),
                        "vwap_side":     _snap.get("vwap_side", 0),
                        "h1_rsi":        _snap.get("h1_rsi", 50),
                        "body_ratio":    _snap.get("body_ratio", 0),
                        "h4_bias":       _snap.get("h4_bias", 0),
                        "sl_dist_atr":   _snap.get("sl_dist_atr", 1.4),
                        "close_pct":     _snap.get("close_pct", 0.5),
                        "day_of_week":   _snap.get("day_of_week", 0),
                        "candles_to_exit": max(1, round(
                            (ts - open_trade["entry_time"]).total_seconds() / 300
                        )),
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
            h4_s = None
            if strategy_mode == "B":
                ICT_M5_WINDOW = 576
                m5_win = m5.iloc[max(0, i - ICT_M5_WINDOW + 1): i + 1]
                sig = evaluate_ict(m5_win, m15_s, h1_s, now=ts.to_pydatetime(),
                                   atr_min=effective_atr)
                # Verrou : un seul trade par OB (évite entrées multiples sur le même OB)
                if sig is not None:
                    ob_ts = sig.meta.get("ob_ts")
                    if ob_ts is not None and ob_ts == last_ob_ts:
                        sig = None
            else:
                h4_s = h4_full.iloc[:h4_full.index.searchsorted(ts, side="right")]
                sig = evaluate(
                    m5.iloc[:i + 1], m15_s, h1_s, h4=h4_s,
                    now=ts.to_pydatetime(),
                    check_session=True,
                    atr_min=effective_atr,
                    _reject_log=rejection_counts,
                )

            if sig is None:
                continue

            # Limite journalière (même règle que le live)
            day_key = ts.strftime("%Y-%m-%d")
            if _day_trades.get(day_key, 0) >= max_trades_day:
                continue
            _day_trades[day_key] = _day_trades.get(day_key, 0) + 1

            # Verrou strat B : mémoriser l'OB utilisé pour ce trade
            if strategy_mode == "B":
                last_ob_ts = sig.meta.get("ob_ts")

            fill = sig.entry + (spread + slippage) * (1 if sig.direction == "long" else -1)
            # Rejeter si TP1 non rentable après coûts aller+retour (spread + 2×slippage)
            _roundtrip = spread + 2 * slippage
            if sig.direction == "long"  and sig.take_profit1 <= sig.entry + _roundtrip:
                continue
            if sig.direction == "short" and sig.take_profit1 >= sig.entry - _roundtrip:
                continue
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
                "tp1_done":      False,
                "be_after_tp1":  True,
                "tp1_close_all": bool(sig.meta.get("tp1_close_all", False)),
                "remaining":    volume,
                "realised":     0.0,
                "max_exit_time": ts.to_pydatetime() + timedelta(minutes=sig.max_duration_min),
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
                    "sl_dist_atr": round(
                        sl_dist / max(float(bar.get("atr", 1) or 1), 0.001), 2
                    ),
                    "close_pct": round(
                        (float(bar.get("close", 0) or 0) - float(bar.get("low", 0) or 0))
                        / max(float(bar.get("high", 0) or 0) - float(bar.get("low", 0) or 0), 0.001), 2
                    ),
                    "day_of_week": ts.weekday(),
                    "h4_bias": (
                        1 if compute_bias(h4_s) == "LONG"
                        else -1 if compute_bias(h4_s) == "SHORT"
                        else 0
                    ) if h4_s is not None and len(h4_s) > 0 else 0,
                },
            }

        # ---- WR par session ----
        from collections import defaultdict as _dd
        _by_sess: dict = _dd(lambda: {"n": 0, "wins": 0})
        for _t in trades_log:
            _s = _t.get("session", "")
            if _s:
                _by_sess[_s]["n"] += 1
                _by_sess[_s]["wins"] += int(_t["won"])
        wr_by_session = {
            _s: {"n": _v["n"], "wr": round(_v["wins"] / _v["n"], 3)}
            for _s, _v in _by_sess.items() if _v["n"] >= 3
        }

        # ---- WR par heure CET ----
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

        # ---- Diagnostic par jour de semaine ----
        _DOW = ["Lun", "Mar", "Mer", "Jeu", "Ven", "Sam", "Dim"]
        _by_dow: dict = _dd(lambda: {"n": 0, "wins": 0, "sl": 0, "fs": 0, "sl_dist": [], "body": []})
        for _t in trades_log:
            _d = _t.get("day_of_week", 0)
            _k = _DOW[_d] if _d < len(_DOW) else str(_d)
            _by_dow[_k]["n"] += 1
            _by_dow[_k]["wins"] += int(_t["won"])
            if _t.get("exit_reason") == "sl":
                _by_dow[_k]["sl"] += 1
                if _t.get("false_stop"):
                    _by_dow[_k]["fs"] += 1
            _by_dow[_k]["sl_dist"].append(_t.get("sl_dist_atr", 1.4))
            _by_dow[_k]["body"].append(_t.get("body_ratio", 0))
        diag_by_dow = {
            _k: {
                "n":           _v["n"],
                "wr":          round(_v["wins"] / _v["n"] * 100, 1) if _v["n"] else 0,
                "sl_pct":      round(_v["sl"] / _v["n"] * 100, 1) if _v["n"] else 0,
                "fs_pct":      round(_v["fs"] / _v["sl"] * 100, 1) if _v["sl"] else 0,
                "sl_dist_atr": round(sum(_v["sl_dist"]) / len(_v["sl_dist"]), 2) if _v["sl_dist"] else 0,
                "body_ratio":  round(sum(_v["body"]) / len(_v["body"]), 2) if _v["body"] else 0,
            }
            for _k, _v in _by_dow.items() if _v["n"] >= 2
        }

        # ---- False stops par distance SL (buckets en ATR) ----
        _fs_by_dist: dict = _dd(lambda: {"n_sl": 0, "n_fs": 0})
        for _t in trades_log:
            if _t.get("exit_reason") == "sl":
                _d = _t.get("sl_dist_atr", 1.4)
                if _d < 0.8:   _bk = "<0.8 ATR (très serré)"
                elif _d < 1.2: _bk = "0.8–1.2 ATR"
                elif _d < 1.6: _bk = "1.2–1.6 ATR"
                else:          _bk = ">1.6 ATR (large)"
                _fs_by_dist[_bk]["n_sl"] += 1
                if _t.get("false_stop"):
                    _fs_by_dist[_bk]["n_fs"] += 1
        false_stop_by_sl_dist = {
            _k: {
                "n_sl":      _v["n_sl"],
                "n_fs":      _v["n_fs"],
                "pct_false": round(_v["n_fs"] / _v["n_sl"] * 100, 1) if _v["n_sl"] else 0.0,
            }
            for _k, _v in _fs_by_dist.items() if _v["n_sl"] >= 1
        }

        # ---- False stops par body_ratio ----
        _fs_by_body: dict = _dd(lambda: {"n_sl": 0, "n_fs": 0})
        for _t in trades_log:
            if _t.get("exit_reason") == "sl":
                _b = _t.get("body_ratio", 0)
                if _b < 0.15:   _bk = "<0.15 (doji)"
                elif _b < 0.30: _bk = "0.15–0.30 (faible)"
                elif _b < 0.50: _bk = "0.30–0.50 (moyen)"
                else:           _bk = ">0.50 (fort)"
                _fs_by_body[_bk]["n_sl"] += 1
                if _t.get("false_stop"):
                    _fs_by_body[_bk]["n_fs"] += 1
        false_stop_by_body = {
            _k: {
                "n_sl":      _v["n_sl"],
                "n_fs":      _v["n_fs"],
                "pct_false": round(_v["n_fs"] / _v["n_sl"] * 100, 1) if _v["n_sl"] else 0.0,
            }
            for _k, _v in _fs_by_body.items() if _v["n_sl"] >= 1
        }

        # ---- Diagnostic LONG vs SHORT ----
        _by_dir: dict = _dd(lambda: {
            "n": 0, "wins": 0, "sl": 0, "fs": 0,
            "sl_dist": [], "close_pct": [], "body": [], "candles": [], "atr": [],
        })
        for _t in trades_log:
            _dir = _t.get("direction", "long")
            _by_dir[_dir]["n"] += 1
            _by_dir[_dir]["wins"] += int(_t["won"])
            if _t.get("exit_reason") == "sl":
                _by_dir[_dir]["sl"] += 1
                if _t.get("false_stop"):
                    _by_dir[_dir]["fs"] += 1
            _by_dir[_dir]["sl_dist"].append(_t.get("sl_dist_atr", 1.4))
            _by_dir[_dir]["close_pct"].append(_t.get("close_pct", 0.5))
            _by_dir[_dir]["body"].append(_t.get("body_ratio", 0))
            _by_dir[_dir]["candles"].append(_t.get("candles_to_exit", 5))
            _by_dir[_dir]["atr"].append(_t.get("atr", 0))
        def _lmean(lst): return round(sum(lst) / len(lst), 2) if lst else 0
        diag_by_direction = {
            _dir: {
                "n":            _v["n"],
                "wr":           round(_v["wins"] / _v["n"] * 100, 1) if _v["n"] else 0,
                "sl_pct":       round(_v["sl"] / _v["n"] * 100, 1) if _v["n"] else 0,
                "fs_pct":       round(_v["fs"] / _v["sl"] * 100, 1) if _v["sl"] else 0,
                "sl_dist_atr":  _lmean(_v["sl_dist"]),
                "close_pct":    _lmean(_v["close_pct"]),
                "body_ratio":   _lmean(_v["body"]),
                "candles_exit": _lmean(_v["candles"]),
                "atr_entry":    _lmean(_v["atr"]),
            }
            for _dir, _v in _by_dir.items() if _v["n"] >= 3
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
                "ema9_dist_r":   round(_mean("ema9_dist_r", 0), 2),
                "ema200_dist_r": round(_mean("ema200_dist_r", 0), 2),
                "vwap_above_pct": round(
                    sum(1 for t in grp if t.get("vwap_side") == 1) / len(grp) * 100, 1
                ),
                "h1_rsi":    round(_mean("h1_rsi", 50), 1),
                "body_ratio": round(_mean("body_ratio", 0), 2),
                "h4_bias":   round(_mean("h4_bias", 0), 2),
                "sl_dist_atr": round(_mean("sl_dist_atr", 1.4), 2),
                "close_pct":  round(_mean("close_pct", 0.5), 2),
                "candles_to_exit": round(_mean("candles_to_exit", 5), 1),
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

        # ---- Persister les seuils adaptatifs (sauf en mode walk-forward) ----
        if write_to_db:
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
                "provider":        data_provider_used,
                "end_gap_days":    data_end_gap_days,
                # Couvre le début ET la fin de la période demandée (tolérance 3j pour
                # week-ends / dernier jour non encore clôturé côté provider).
                "full_coverage":   data_start_actual <= start and data_end_gap_days <= 3,
                "provider_errors": data_provider_errors or None,
            },
            "symbol":        symbol,
            "atr_min_final": round(adaptive.atr_min, 4),
            "ema9_mult_final": round(adaptive.ema9_mult, 3),
            "m15_mult_final":  round(adaptive.m15_mult, 3),
            "ml_samples":    0,
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
            "wr_by_session":          wr_by_session,
            "wr_by_hour":             wr_by_hour,
            "false_stop_spike_stats": false_stop_spike_stats,
            "false_stop_by_hour":     false_stop_by_hour,
            "false_stop_by_sl_dist":  false_stop_by_sl_dist,
            "false_stop_by_body":     false_stop_by_body,
            "diag_by_dow":            diag_by_dow,
            "diag_by_direction":      diag_by_direction,
            "rejection_counts":       dict(sorted(rejection_counts.items(), key=lambda x: -x[1])),
            "sl_atr_mult":            SL_ATR_MULT,
        }
        with _lock:
            _last_by_strategy[strategy_mode] = result
        _set(running=False, pct=100, bars_done=total, trades=n_trades,
             wins=n_wins, status="done", last_result=result)
        return result

    except Exception as exc:
        _set(running=False, status="error", error=str(exc))
        raise

    finally:
        for k, v in _saved_strategy.items():
            setattr(strategy, k, v)


# --------------------------------------------------------------------------- #
# Lancement asynchrone (appelé depuis l'API)
# --------------------------------------------------------------------------- #
def launch_pretrain(
    start: str,
    end: str,
    symbol: str = "XAUUSD",
    atr_min: Optional[float] = None,
    reset: bool = True,
    capital: float = 10_000.0,
    risk_pct: float = 5.0,
    strategy_mode: str = "A",
    on_complete=None,
    extra_overrides: Optional[Dict[str, Any]] = None,
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
                         capital=capital, risk_pct=risk_pct, strategy_mode=strategy_mode,
                         extra_overrides=extra_overrides)
            if on_complete:
                on_complete()
        except Exception as exc:
            _set(running=False, status="error", error=str(exc))

    threading.Thread(target=_run, daemon=True, name="pretrain").start()


# --------------------------------------------------------------------------- #
# Walk-forward : robustesse sur N fenêtres indépendantes
# --------------------------------------------------------------------------- #
def run_walk_forward(
    start: str,
    end: str,
    n_splits: int = 4,
    symbol: str = "XAUUSD",
    capital: float = 10_000.0,
    risk_pct: float = 5.0,
    strategy_mode: str = "A",
    extra_overrides: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Divise start→end en n_splits fenêtres indépendantes de même durée.
    Chaque fenêtre tourne en isolation totale (gate fraîche, write_to_db=False).

    Retourne par fenêtre : n_trades, win_rate, profit_factor, net_pnl,
    plus des métriques de cohérence globales (avg_pf, std_pf, is_robust).

    Règle d'or : si PF est cohérent (>1 dans ≥75% des fenêtres et std_pf<0.3),
    la stratégie est robuste et non overfittée.
    """
    import statistics as _st
    from datetime import date as _date, timedelta as _td
    import dateutil.parser as _dparse

    start_dt   = _dparse.parse(start).date()
    end_dt     = _dparse.parse(end).date()
    total_days = (end_dt - start_dt).days
    seg_days   = total_days // n_splits

    windows = []
    for k in range(n_splits):
        seg_start = (start_dt + _td(days=k * seg_days)).isoformat()
        seg_end   = end if k == n_splits - 1 else (start_dt + _td(days=(k + 1) * seg_days)).isoformat()
        try:
            r = run_pretrain(
                start=seg_start, end=seg_end,
                symbol=symbol, capital=capital, risk_pct=risk_pct,
                strategy_mode=strategy_mode, reset=True,
                extra_overrides=extra_overrides,
                write_to_db=False,
            )
            n_sl = r.get("false_stops", {}).get("n_sl_direct", 0)
            windows.append({
                "window":        k + 1,
                "period":        f"{seg_start} → {seg_end}",
                "n_trades":      r["n_trades"],
                "win_rate":      r["win_rate"],
                "profit_factor": r["profit_factor"],
                "net_pnl":       r["net_pnl"],
                "sl_direct_pct": round(n_sl / max(r["n_trades"], 1) * 100, 1),
                # Diagnostic : pourquoi si peu (ou pas) de trades sur cette fenêtre —
                # rejets par étape du pipeline + couverture réelle des données.
                "rejection_counts": r.get("rejection_counts", {}),
                "bars":             r.get("data_coverage", {}).get("bars"),
                "data_coverage":    r.get("data_coverage", {}),
            })
        except Exception as exc:
            windows.append({
                "window": k + 1,
                "period": f"{seg_start} → {seg_end}",
                "error":  str(exc),
            })

    valid_pfs = [
        w["profit_factor"] for w in windows
        if "profit_factor" in w and w.get("n_trades", 0) >= 5
    ]

    avg_pf = round(sum(valid_pfs) / len(valid_pfs), 3) if valid_pfs else 0.0
    std_pf = round(_st.stdev(valid_pfs), 3) if len(valid_pfs) >= 2 else 0.0
    pct_ok = round(sum(1 for pf in valid_pfs if pf > 1.0) / len(valid_pfs) * 100, 1) if valid_pfs else 0.0

    return {
        "windows":         windows,
        "n_splits":        n_splits,
        "is_robust":       bool(valid_pfs) and pct_ok >= 75 and std_pf < 0.30,
        "avg_pf":          avg_pf,
        "min_pf":          round(min(valid_pfs), 3) if valid_pfs else 0.0,
        "max_pf":          round(max(valid_pfs), 3) if valid_pfs else 0.0,
        "std_pf":          std_pf,
        "pct_profitable":  pct_ok,
    }
