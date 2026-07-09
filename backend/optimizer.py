"""
optimizer.py
============
Grid search over key strategy parameters to find the best combination
for a given historical period. Optimises for Sharpe ratio.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional

from backtest import BacktestConfig, run_backtest
import strategy


@dataclass
class OptimizeConfig:
    start: str
    end: str
    symbol: str = "XAUUSD"
    capital: float = 10000.0
    risk_pct: float = 1.0
    spread_pips: float = 0.5   # aligné avec BacktestConfig (broker standard conservateur)
    slippage_pips: float = 0.1
    max_trades_per_day: int = 4
    daily_stop_pct: float = 2.0


PARAM_GRID = {
    "adx_min":      [20.0, 25.0, 28.0, 30.0],
    "rsi_low":      [42.0, 45.0, 47.0],
    "rsi_high":     [53.0, 55.0, 58.0],
    "sl_atr_mult":  [1.0, 1.2, 1.5],
    "sr_proximity": [0.3, 0.5, 0.7],
}


def run_optimize(cfg: OptimizeConfig) -> Dict[str, Any]:
    from backtest import load_m5_data
    keys = list(PARAM_GRID.keys())
    values = list(PARAM_GRID.values())
    combos = list(itertools.product(*values))

    results = []
    total = len(combos)

    # Load data ONCE — reused across all 108 backtests
    shared_data = load_m5_data(cfg.start, cfg.end, symbol=cfg.symbol)

    # Save original strategy params
    orig = {
        "adx_min": strategy.ADX_MIN,
        "rsi_low": strategy.RSI_LOW,
        "rsi_high": strategy.RSI_HIGH,
        "sl_atr_mult": strategy.SL_ATR_MULT,
        "sr_proximity": strategy.SR_PROXIMITY_ATR,
    }

    for i, combo in enumerate(combos):
        params = dict(zip(keys, combo))

        # Sanity check: RSI range must be valid
        if params["rsi_low"] >= params["rsi_high"]:
            continue

        # Apply params temporarily
        strategy.ADX_MIN = params["adx_min"]
        strategy.RSI_LOW = params["rsi_low"]
        strategy.RSI_HIGH = params["rsi_high"]
        strategy.SL_ATR_MULT = params["sl_atr_mult"]
        strategy.SR_PROXIMITY_ATR = params["sr_proximity"]

        bt_cfg = BacktestConfig(
            start=cfg.start, end=cfg.end, symbol=cfg.symbol,
            capital=cfg.capital, risk_pct=cfg.risk_pct,
            spread_pips=cfg.spread_pips, slippage_pips=cfg.slippage_pips,
            max_trades_per_day=cfg.max_trades_per_day,
            daily_stop_pct=cfg.daily_stop_pct,
        )

        try:
            report = run_backtest(bt_cfg, preloaded_data=shared_data)
            summary = report.get("summary", {})
            if summary.get("trades", 0) < 10:
                continue
            sharpe = summary.get("sharpe_ratio", 0) or 0
            pf = summary.get("profit_factor") or 0
            score = sharpe * (pf if pf else 0)
            results.append({
                "params": params,
                "sharpe": round(sharpe, 3),
                "profit_factor": round(float(pf), 3) if pf else 0,
                "winrate": summary.get("winrate", 0),
                "trades": summary.get("trades", 0),
                "net_profit_pct": summary.get("net_profit_pct", 0),
                "max_drawdown_pct": summary.get("max_drawdown_pct", 0),
                "score": round(score, 4),
            })
        except Exception:
            continue

    # Restore original params
    strategy.ADX_MIN = orig["adx_min"]
    strategy.RSI_LOW = orig["rsi_low"]
    strategy.RSI_HIGH = orig["rsi_high"]
    strategy.SL_ATR_MULT = orig["sl_atr_mult"]
    strategy.SR_PROXIMITY_ATR = orig["sr_proximity"]

    results.sort(key=lambda x: x["score"], reverse=True)
    top5 = results[:5]

    return {
        "total_combinations": total,
        "tested": len(results),
        "top_results": top5,
        "best": top5[0] if top5 else None,
        "param_grid": PARAM_GRID,
    }


# --------------------------------------------------------------------------- #
# Optimisation Bayésienne via Optuna + walk-forward (anti-overfitting)
# --------------------------------------------------------------------------- #

# Espace de recherche : paramètres à optimiser et leurs bornes
OPTUNA_SEARCH_SPACE = {
    "RSI_M5_LONG_MIN":      (43.0, 51.0),  # seuil RSI M5 pour LONG
    "RSI_M5_SHORT_MAX":     (49.0, 57.0),  # seuil RSI M5 pour SHORT
    "ATR_MIN":              (1.5,  4.0),   # volatilité minimale
    "ADX_MIN":              (15.0, 30.0),  # force de tendance minimale
    "TREND_BIAS_DISTANCE":  (0.3,  0.8),   # distance minimale EMA200
}


def run_optuna_optimize(
    start: str,
    end: str,
    n_trials: int = 30,
    n_splits: int = 3,
    symbol: str = "XAUUSD",
    capital: float = 10_000.0,
    risk_pct: float = 5.0,
    progress_cb=None,
) -> Dict[str, Any]:
    """
    Optimisation Bayésienne des paramètres stratégie via Optuna.

    Chaque essai évalue un jeu de paramètres sur un walk-forward en n_splits
    fenêtres indépendantes. Score = avg_pf × pct_rentables − std_pf × 0.5
    (récompense la cohérence, pénalise la variance inter-fenêtres).

    progress_cb(done, total, best_score) — appelé après chaque essai.
    """
    try:
        import optuna
    except ImportError:
        return {"error": "optuna non installé — lancez : pip install optuna>=3.6"}

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    from pretrain import run_walk_forward

    def objective(trial: "optuna.Trial") -> float:
        rsi_long  = trial.suggest_int("RSI_M5_LONG_MIN",  43, 51)
        rsi_short = trial.suggest_int("RSI_M5_SHORT_MAX", 49, 57)
        if rsi_long >= rsi_short:
            raise optuna.TrialPruned()

        overrides = {
            "RSI_M5_LONG_MIN":     float(rsi_long),
            "RSI_M5_SHORT_MAX":    float(rsi_short),
            "ATR_MIN":             trial.suggest_float("ATR_MIN", 1.5, 4.0, step=0.5),
            "ADX_MIN":             trial.suggest_float("ADX_MIN", 15.0, 30.0, step=2.5),
            "TREND_BIAS_DISTANCE": trial.suggest_float("TREND_BIAS_DISTANCE", 0.3, 0.8, step=0.1),
        }

        wf = run_walk_forward(
            start=start, end=end, n_splits=n_splits,
            symbol=symbol, capital=capital, risk_pct=risk_pct,
            extra_overrides=overrides,
        )

        score = wf["avg_pf"] * (wf["pct_profitable"] / 100.0) - wf["std_pf"] * 0.5
        if progress_cb:
            progress_cb(trial.number + 1, n_trials, round(score, 4))
        return score

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42),
    )
    study.optimize(objective, n_trials=n_trials, n_jobs=1)

    completed = [t for t in study.trials if t.state.name == "COMPLETE"]
    top10 = sorted(completed, key=lambda t: -(t.value or -9999))[:10]

    best = study.best_trial
    return {
        "best_params":  best.params,
        "best_score":   round(best.value, 4),
        "n_trials":     n_trials,
        "n_completed":  len(completed),
        "top_trials": [
            {"trial": t.number, "params": t.params, "score": round(t.value, 4)}
            for t in top10
        ],
        "search_space": OPTUNA_SEARCH_SPACE,
    }
