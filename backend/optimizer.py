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
    spread_pips: float = 0.3
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
    keys = list(PARAM_GRID.keys())
    values = list(PARAM_GRID.values())
    combos = list(itertools.product(*values))

    results = []
    total = len(combos)

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
            report = run_backtest(bt_cfg)
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
