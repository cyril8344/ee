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
    MAX_TRADE_MINUTES, SL_ATR_MULT,
)
from backtest import load_m5_data, resample, _try_exit
import database as db
from ml_gate import OnlineLogisticRegression, AdaptiveThresholds

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
def run_pretrain(
    start: str,
    end: str,
    symbol: str = "XAUUSD",
    atr_min: Optional[float] = None,
    reset: bool = True,
) -> Dict[str, Any]:
    """
    Lance le pré-entraînement en mode bloquant.
    Appelé depuis un thread via launch_pretrain().

    reset=True  : repart de zéro (recommandé la 1ère fois)
    reset=False : accumule sur l'historique existant
    """
    _set(running=True, pct=0, bars_done=0, trades=0, wins=0,
         status="running", error=None, last_result=None)

    contract_size = 100.0   if symbol == "XAUUSD" else 100000.0
    pip_size      = 0.1     if symbol == "XAUUSD" else 0.0001
    default_atr   = 3.0     if symbol == "XAUUSD" else 0.00030
    effective_atr = atr_min if atr_min is not None else default_atr
    spread    = 0.3  * pip_size
    slippage  = 0.1  * pip_size

    try:
        # ---- Charger et préparer les données ----
        _set(status="Chargement des données…")
        m5_raw   = load_m5_data(start, end, symbol=symbol)
        if len(m5_raw) < 300:
            raise ValueError("Pas assez de données pour la période sélectionnée.")

        m5       = add_indicators(m5_raw)
        m15_full = add_indicators(resample(m5_raw, "15min"))
        h1_full  = add_indicators(resample(m5_raw, "60min"))

        # ---- Initialiser les systèmes d'apprentissage ----
        if reset:
            gate     = OnlineLogisticRegression.__new__(OnlineLogisticRegression)
            gate.weights            = [0.0] * 6
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
        equity       = 10_000.0
        equity_curve = [{"ts": m5.index[warmup].isoformat(), "equity": equity}]
        pnl_wins   = []   # PnL $ des trades gagnants
        pnl_losses = []   # PnL $ (abs) des trades perdants
        mae_wins   = []   # MAE en R des trades gagnants
        mfe_wins   = []   # MFE en R des trades gagnants
        mae_loss   = []   # MAE en R des trades perdants
        mfe_loss   = []   # MFE en R des trades perdants

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
                    pnl, _, _ = exit_info
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
                    if risk > 0:
                        mae_r = open_trade.get("mae", 0.0) / risk
                        mfe_r = open_trade.get("mfe", 0.0) / risk
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

            # evaluate() SANS ml_gate ni adaptive → signal brut non filtré
            sig = evaluate(
                m5.iloc[:i + 1], m15_s, h1_s,
                now=ts.to_pydatetime(),
                check_session=True,
                atr_min=effective_atr,
            )

            if sig is None:
                continue

            fill = sig.entry + (spread + slippage) * (1 if sig.direction == "long" else -1)
            open_trade = {
                "direction":    sig.direction,
                "entry_time":   ts.to_pydatetime(),
                "entry":        fill,
                "stop_loss":    sig.stop_loss,
                "tp1":          sig.take_profit1,
                "tp2":          sig.take_profit2,
                "volume":       0.01,
                "tp1_done":     False,
                "remaining":    0.01,
                "realised":     0.0,
                "max_exit_time": ts.to_pydatetime() + timedelta(minutes=MAX_TRADE_MINUTES),
                "triggers":     sig.meta.get("triggers", []),
                "ml_features":  sig.meta.get("ml_features"),
                "risk":         abs(fill - sig.stop_loss),
                "mae":          0.0,
                "mfe":          0.0,
            }

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
            "symbol":        symbol,
            "atr_min_final": round(adaptive.atr_min, 4),
            "ema9_mult_final": round(adaptive.ema9_mult, 3),
            "m15_mult_final":  round(adaptive.m15_mult, 3),
            "ml_samples":    gate.n_samples,
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
            run_pretrain(start, end, symbol=symbol, atr_min=atr_min, reset=reset)
        except Exception as exc:
            _set(running=False, status="error", error=str(exc))

    threading.Thread(target=_run, daemon=True, name="pretrain").start()
