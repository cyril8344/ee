"""
backtest_worker.py
==================
Background worker that runs perpetual optimization every N hours.

Uses Optuna (Bayesian optimization) if available, falls back to the
existing grid search from optimizer.py.

Optimization schedule: every 4 hours (configurable via AGENT_INTERVAL_HOURS env).
Looks back at the last 90 days of data by default.
"""

from __future__ import annotations

import logging
import os
import queue
import threading
import traceback
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Result dataclass
# --------------------------------------------------------------------------- #

@dataclass
class WorkerResult:
    params: Dict[str, Any]
    new_sharpe: float
    current_sharpe: float
    improvement_pct: float
    trades: int
    winrate: float
    timestamp: str
    symbol: str


# --------------------------------------------------------------------------- #
# Optuna search space — matches optimizer.PARAM_GRID
# --------------------------------------------------------------------------- #

def _optuna_search_space(trial) -> Dict[str, Any]:
    """Define Optuna trial parameters matching the grid-search space."""
    return {
        "adx_min":      trial.suggest_float("adx_min",     20.0, 30.0, step=1.0),
        "rsi_low":      trial.suggest_float("rsi_low",     42.0, 47.0, step=1.0),
        "rsi_high":     trial.suggest_float("rsi_high",    53.0, 58.0, step=1.0),
        "sl_atr_mult":  trial.suggest_float("sl_atr_mult",  1.0,  1.5, step=0.1),
        "sr_proximity": trial.suggest_float("sr_proximity", 0.3,  0.7, step=0.1),
        "ob_lookback":  trial.suggest_int("ob_lookback",   20,   60,   step=5),
        "ob_proximity": trial.suggest_float("ob_proximity", 0.2,  0.6, step=0.1),
        "fvg_min_size": trial.suggest_float("fvg_min_size", 0.2,  0.5, step=0.05),
    }


# --------------------------------------------------------------------------- #
# Worker
# --------------------------------------------------------------------------- #

class BacktestWorker:
    """
    Daemon thread that periodically re-optimises strategy parameters.

    Results that show >= 10% Sharpe improvement over the current baseline are
    put onto result_queue for AgentManager to decide whether to apply.
    """

    def __init__(
        self,
        result_queue: queue.Queue,
        symbol: str = "XAUUSD",
        lookback_days: int = 90,
    ) -> None:
        self.result_queue = result_queue
        self.symbol = symbol
        self.lookback_days = lookback_days
        self.interval_hours: float = float(
            os.environ.get("AGENT_INTERVAL_HOURS", "4")
        )
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._next_run: Optional[datetime] = None

    # ------------------------------------------------------------------ #
    # Lifecycle
    # ------------------------------------------------------------------ #

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run, daemon=True, name=f"BacktestWorker-{self.symbol}"
        )
        self._thread.start()
        logger.info("[BacktestWorker:%s] started (interval=%sh)", self.symbol, self.interval_hours)

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=5)
        logger.info("[BacktestWorker:%s] stopped", self.symbol)

    @property
    def next_run(self) -> Optional[datetime]:
        return self._next_run

    # ------------------------------------------------------------------ #
    # Main loop
    # ------------------------------------------------------------------ #

    def _run(self) -> None:
        # First run 30 min after startup so the user sees a Sharpe quickly
        self._next_run = datetime.now(timezone.utc) + timedelta(minutes=30)

        while not self._stop_event.is_set():
            now = datetime.now(timezone.utc)
            if now >= self._next_run:
                try:
                    self._run_optimization()
                except Exception:
                    logger.error("[BacktestWorker:%s] optimization error:\n%s",
                                 self.symbol, traceback.format_exc())
                self._next_run = datetime.now(timezone.utc) + timedelta(
                    hours=self.interval_hours
                )
                logger.info("[BacktestWorker:%s] next run at %s",
                            self.symbol, self._next_run.isoformat())

            # Sleep in short chunks so stop() is responsive
            self._stop_event.wait(timeout=30)

    # ------------------------------------------------------------------ #
    # Core optimization
    # ------------------------------------------------------------------ #

    def _run_optimization(self) -> None:
        from backtest import BacktestConfig, load_m5_data, run_backtest
        from optimizer import OptimizeConfig, run_optimize, PARAM_GRID
        import strategy

        end_dt = datetime.now(timezone.utc)
        start_dt = end_dt - timedelta(days=self.lookback_days)
        start_str = start_dt.strftime("%Y-%m-%d")
        end_str = end_dt.strftime("%Y-%m-%d")

        logger.info("[BacktestWorker:%s] optimising %s -> %s", self.symbol, start_str, end_str)

        # ---- load data once ----
        shared_data = load_m5_data(start_str, end_str, symbol=self.symbol)

        opt_cfg = OptimizeConfig(
            start=start_str, end=end_str, symbol=self.symbol,
        )
        bt_cfg_base = BacktestConfig(
            start=start_str, end=end_str, symbol=self.symbol,
        )

        # ---- current baseline Sharpe ----
        current_sharpe = self._get_current_sharpe(bt_cfg_base, shared_data)

        # ---- try Optuna first, fall back to grid ----
        best_params: Optional[Dict[str, Any]] = None
        best_sharpe: float = -999.0
        best_trades: int = 0
        best_winrate: float = 0.0

        try:
            import optuna
            optuna.logging.set_verbosity(optuna.logging.WARNING)
            best_params, best_sharpe, best_trades, best_winrate = self._optuna_optimize(
                optuna, strategy, bt_cfg_base, shared_data
            )
            logger.info("[BacktestWorker:%s] Optuna best sharpe=%.3f", self.symbol, best_sharpe)
        except ImportError:
            logger.info("[BacktestWorker:%s] Optuna not installed, using grid search", self.symbol)
        except Exception:
            logger.warning("[BacktestWorker:%s] Optuna failed:\n%s", self.symbol, traceback.format_exc())

        if best_params is None:
            # Fall back to grid search
            grid_result = run_optimize(opt_cfg)
            best = grid_result.get("best")
            if best:
                best_params = best["params"]
                best_sharpe = best["sharpe"]
                best_trades = best.get("trades", 0)
                best_winrate = best.get("winrate", 0.0)
                logger.info("[BacktestWorker:%s] Grid best sharpe=%.3f", self.symbol, best_sharpe)

        if best_params is None:
            logger.warning("[BacktestWorker:%s] no valid result from optimization", self.symbol)
            return

        # ---- evaluate improvement ----
        if current_sharpe and current_sharpe != 0:
            improvement_pct = (best_sharpe - current_sharpe) / abs(current_sharpe) * 100.0
        else:
            improvement_pct = 0.0

        logger.info(
            "[BacktestWorker:%s] current_sharpe=%.3f  new_sharpe=%.3f  improvement=%.1f%%",
            self.symbol, current_sharpe, best_sharpe, improvement_pct,
        )

        if improvement_pct >= 10.0:
            result = WorkerResult(
                params=best_params,
                new_sharpe=round(best_sharpe, 4),
                current_sharpe=round(current_sharpe, 4),
                improvement_pct=round(improvement_pct, 2),
                trades=best_trades,
                winrate=round(best_winrate, 2),
                timestamp=datetime.now(timezone.utc).isoformat(),
                symbol=self.symbol,
            )
            self.result_queue.put(result)
            logger.info("[BacktestWorker:%s] result queued (improvement %.1f%%)",
                        self.symbol, improvement_pct)

    def _get_current_sharpe(self, bt_cfg, shared_data) -> float:
        """Run a backtest with current strategy params and return Sharpe."""
        from backtest import run_backtest
        try:
            report = run_backtest(bt_cfg, preloaded_data=shared_data)
            return float(report.get("summary", {}).get("sharpe_ratio", 0) or 0)
        except Exception:
            logger.warning("[BacktestWorker:%s] current-sharpe backtest failed:\n%s",
                           self.symbol, traceback.format_exc())
            return 0.0

    def _optuna_optimize(
        self,
        optuna,
        strategy,
        bt_cfg,
        shared_data,
        n_trials: int = 50,
    ):
        """Run Optuna study, return (best_params, best_sharpe, trades, winrate)."""
        from backtest import run_backtest

        # Save originals so we can restore after each trial
        _orig = {
            "adx_min":      strategy.ADX_MIN,
            "rsi_low":      strategy.RSI_LOW,
            "rsi_high":     strategy.RSI_HIGH,
            "sl_atr_mult":  strategy.SL_ATR_MULT,
            "sr_proximity": strategy.SR_PROXIMITY_ATR,
            "ob_lookback":  strategy.OB_LOOKBACK,
            "ob_proximity": strategy.OB_PROXIMITY_ATR,
            "fvg_min_size": strategy.FVG_MIN_SIZE_ATR,
        }

        trial_metadata: Dict[int, Dict] = {}

        def objective(trial):
            params = _optuna_search_space(trial)
            if params["rsi_low"] >= params["rsi_high"]:
                raise optuna.exceptions.TrialPruned()

            strategy.ADX_MIN = params["adx_min"]
            strategy.RSI_LOW = params["rsi_low"]
            strategy.RSI_HIGH = params["rsi_high"]
            strategy.SL_ATR_MULT = params["sl_atr_mult"]
            strategy.SR_PROXIMITY_ATR = params["sr_proximity"]
            strategy.OB_LOOKBACK = int(params["ob_lookback"])
            strategy.OB_PROXIMITY_ATR = params["ob_proximity"]
            strategy.FVG_MIN_SIZE_ATR = params["fvg_min_size"]

            try:
                report = run_backtest(bt_cfg, preloaded_data=shared_data)
                summary = report.get("summary", {})
                trades = summary.get("trades", 0)
                if trades < 10:
                    raise optuna.exceptions.TrialPruned()
                sharpe = float(summary.get("sharpe_ratio", 0) or 0)
                trial_metadata[trial.number] = {
                    "trades": trades,
                    "winrate": summary.get("winrate", 0.0),
                }
                return -sharpe  # minimize negative sharpe = maximize sharpe
            except optuna.exceptions.TrialPruned:
                raise
            except Exception:
                raise optuna.exceptions.TrialPruned()
            finally:
                # Always restore originals
                strategy.ADX_MIN = _orig["adx_min"]
                strategy.RSI_LOW = _orig["rsi_low"]
                strategy.RSI_HIGH = _orig["rsi_high"]
                strategy.SL_ATR_MULT = _orig["sl_atr_mult"]
                strategy.SR_PROXIMITY_ATR = _orig["sr_proximity"]
                strategy.OB_LOOKBACK = _orig["ob_lookback"]
                strategy.OB_PROXIMITY_ATR = _orig["ob_proximity"]
                strategy.FVG_MIN_SIZE_ATR = _orig["fvg_min_size"]

        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=n_trials, n_jobs=1, show_progress_bar=False)

        best_trial = study.best_trial
        best_params = best_trial.params
        best_sharpe = -best_trial.value
        meta = trial_metadata.get(best_trial.number, {})
        return best_params, best_sharpe, meta.get("trades", 0), meta.get("winrate", 0.0)
