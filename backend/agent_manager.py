"""
agent_manager.py
================
Orchestrator that coordinates agent_memory and backtest_worker.

Responsibilities:
- Start/stop all background workers
- Receive WorkerResults from the queue
- Decide whether to apply new parameters (improvement >= threshold, no open trade)
- Apply params to strategy module at runtime (no restart needed)
- Persist approved params to backend/agent_config.json
- Expose status for the API
"""

from __future__ import annotations

import json
import logging
import os
import queue
import threading
import traceback
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from backtest_worker import BacktestWorker, WorkerResult

logger = logging.getLogger(__name__)

_DATA_DIR     = "/data" if os.path.isdir("/data") else os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH   = os.path.join(_DATA_DIR, "agent_config.json")
HISTORY_PATH  = os.path.join(_DATA_DIR, "agent_history.json")
MAX_HISTORY   = 200

# Minimum improvement to accept new params (10 %)
IMPROVEMENT_THRESHOLD_PCT = 10.0

# Minimum trades in backtest to trust the result
MIN_BACKTEST_TRADES = 20

# Symbols the agent monitors
DEFAULT_SYMBOLS = ["XAUUSD", "EURUSD"]

# Mapping from WorkerResult.params keys -> strategy module attribute names
PARAM_MAP: Dict[str, str] = {
    "adx_min":      "ADX_MIN",
    "rsi_low":      "RSI_LOW",
    "rsi_high":     "RSI_HIGH",
    "sl_atr_mult":  "SL_ATR_MULT",
    "sr_proximity": "SR_PROXIMITY_ATR",
    "ob_lookback":  "OB_LOOKBACK",
    "ob_proximity": "OB_PROXIMITY_ATR",
    "fvg_min_size": "FVG_MIN_SIZE_ATR",
}


class AgentManager:
    """
    Manages perpetual optimisation workers and applies improved strategy
    parameters automatically when conditions are met.

    Parameters
    ----------
    bot_state_ref : BotState
        Reference to the live BotState object from main.py. Used to check
        whether any position is currently open before applying new params.
    """

    def __init__(self, bot_state_ref) -> None:
        self._state_ref = bot_state_ref
        self._result_queue: queue.Queue = queue.Queue()
        self._workers: Dict[str, BacktestWorker] = {}
        self._lock = threading.Lock()

        # Status fields (thread-safe via _lock)
        self._running = False
        self._last_run: Optional[str] = None
        self._next_run: Optional[str] = None
        self._current_sharpe: Optional[float] = None
        self._last_improvement: Optional[float] = None
        self._params_applied = False
        self._applied_at: Optional[str] = None
        self._run_history: List[Dict[str, Any]] = []

        self._monitor_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

        self._load_history()

    # ------------------------------------------------------------------ #
    # Lifecycle
    # ------------------------------------------------------------------ #

    def start(self) -> None:
        """Start background workers for each active symbol."""
        if os.environ.get("DISABLE_AGENT", "").lower() in ("1", "true", "yes"):
            logger.info("[AgentManager] disabled via DISABLE_AGENT env var")
            return

        with self._lock:
            if self._running:
                return
            self._running = True

        active_markets: List[str] = getattr(
            self._state_ref, "market_states", {}
        ).keys() or DEFAULT_SYMBOLS

        for sym in active_markets:
            worker = BacktestWorker(result_queue=self._result_queue, symbol=sym)
            worker.start()
            self._workers[sym] = worker
            logger.info("[AgentManager] worker started for %s", sym)

        self._stop_event.clear()
        self._monitor_thread = threading.Thread(
            target=self._monitor_loop, daemon=True, name="AgentManager-monitor"
        )
        self._monitor_thread.start()
        logger.info("[AgentManager] started")

    def stop(self) -> None:
        """Stop all workers and the monitor thread."""
        self._stop_event.set()
        for sym, worker in self._workers.items():
            worker.stop()
            logger.info("[AgentManager] worker stopped for %s", sym)
        self._workers.clear()
        if self._monitor_thread:
            self._monitor_thread.join(timeout=5)
        with self._lock:
            self._running = False
        logger.info("[AgentManager] stopped")

    # ------------------------------------------------------------------ #
    # Status
    # ------------------------------------------------------------------ #

    def status(self) -> Dict[str, Any]:
        """Return current agent status dict for the /api/agent endpoint."""
        with self._lock:
            next_run: Optional[str] = None
            for worker in self._workers.values():
                nr = worker.next_run
                if nr is not None:
                    next_run = nr.isoformat()
                    break  # take first worker's next run

            return {
                "running": self._running,
                "last_run": self._last_run,
                "next_run": next_run,
                "current_sharpe": self._current_sharpe,
                "last_improvement": self._last_improvement,
                "params_applied": self._params_applied,
                "applied_at": self._applied_at,
                "workers": list(self._workers.keys()),
            }

    def history(self) -> List[Dict[str, Any]]:
        """Return the list of past optimization runs (newest first)."""
        with self._lock:
            return list(reversed(self._run_history))

    # ------------------------------------------------------------------ #
    # Monitor loop
    # ------------------------------------------------------------------ #

    def _monitor_loop(self) -> None:
        """Background thread: drains result_queue every 30 s."""
        while not self._stop_event.is_set():
            self._stop_event.wait(timeout=30)
            try:
                self._drain_queue()
            except Exception:
                logger.error("[AgentManager] monitor loop error:\n%s", traceback.format_exc())

    def _drain_queue(self) -> None:
        while True:
            try:
                result: WorkerResult = self._result_queue.get_nowait()
            except queue.Empty:
                break

            now_iso = datetime.now(timezone.utc).isoformat()
            applied = self._should_apply(result)

            with self._lock:
                self._last_run = now_iso
                self._current_sharpe = result.current_sharpe
                entry: Dict[str, Any] = {
                    "timestamp": now_iso,
                    "symbol": result.symbol,
                    "current_sharpe": round(result.current_sharpe, 4),
                    "new_sharpe": round(result.new_sharpe, 4),
                    "improvement_pct": round(result.improvement_pct, 2),
                    "trades": result.trades,
                    "applied": applied,
                    "params": result.params,
                }
                self._run_history.append(entry)
                if len(self._run_history) > MAX_HISTORY:
                    self._run_history = self._run_history[-MAX_HISTORY:]

            self._save_history()

            logger.info(
                "[AgentManager] result received symbol=%s  improvement=%.1f%%  trades=%d",
                result.symbol, result.improvement_pct, result.trades,
            )

            if applied:
                self._apply_params(result)
            else:
                logger.info(
                    "[AgentManager] result NOT applied (improvement=%.1f%%, trades=%d, has_open=%s)",
                    result.improvement_pct, result.trades, self._has_open_position(),
                )

    # ------------------------------------------------------------------ #
    # Decision logic
    # ------------------------------------------------------------------ #

    def _should_apply(self, result: WorkerResult) -> bool:
        """Return True iff the result passes all gates."""
        if result.improvement_pct < IMPROVEMENT_THRESHOLD_PCT:
            return False
        if result.trades < MIN_BACKTEST_TRADES:
            return False
        if self._has_open_position():
            return False
        return True

    def _has_open_position(self) -> bool:
        """Return True if any MarketState has an open position."""
        try:
            market_states = getattr(self._state_ref, "market_states", {})
            for ms in market_states.values():
                if ms.position is not None:
                    return True
        except Exception:
            pass
        return False

    # ------------------------------------------------------------------ #
    # Apply params
    # ------------------------------------------------------------------ #

    def _apply_params(self, result: WorkerResult) -> None:
        """Thread-safely apply new strategy params and persist to JSON."""
        import strategy

        with self._lock:
            for param_key, attr_name in PARAM_MAP.items():
                if param_key in result.params:
                    val = result.params[param_key]
                    try:
                        setattr(strategy, attr_name, float(val))
                        logger.info(
                            "[AgentManager] strategy.%s = %s", attr_name, val
                        )
                    except Exception:
                        logger.warning(
                            "[AgentManager] failed to set strategy.%s = %s",
                            attr_name, val
                        )

            now_iso = datetime.now(timezone.utc).isoformat()
            self._params_applied = True
            self._last_improvement = result.improvement_pct
            self._applied_at = now_iso

        config_data = {
            "applied_at": now_iso,
            "symbol": result.symbol,
            "sharpe_improvement_pct": result.improvement_pct,
            "params": result.params,
        }
        try:
            with open(CONFIG_PATH, "w", encoding="utf-8") as fh:
                json.dump(config_data, fh, indent=2)
            logger.info("[AgentManager] params saved to %s", CONFIG_PATH)
        except Exception:
            logger.warning("[AgentManager] could not save agent_config.json:\n%s",
                           traceback.format_exc())

        logger.info(
            "[AgentManager] params APPLIED -- symbol=%s  sharpe %.3f->%.3f  (+%.1f%%)",
            result.symbol, result.current_sharpe, result.new_sharpe, result.improvement_pct,
        )

    # ------------------------------------------------------------------ #
    # Load saved config on startup
    # ------------------------------------------------------------------ #

    def load_saved_config(self) -> None:
        """
        On startup, read agent_config.json (if it exists) and apply the saved
        strategy params so the bot resumes with the last optimised config.
        """
        if not os.path.isfile(CONFIG_PATH):
            logger.info("[AgentManager] no saved config found at %s", CONFIG_PATH)
            return

        try:
            with open(CONFIG_PATH, "r", encoding="utf-8") as fh:
                data = json.load(fh)
        except Exception:
            logger.warning("[AgentManager] could not read agent_config.json:\n%s",
                           traceback.format_exc())
            return

        params = data.get("params", {})
        if not params:
            return

        import strategy

        with self._lock:
            for param_key, attr_name in PARAM_MAP.items():
                if param_key in params:
                    val = params[param_key]
                    try:
                        setattr(strategy, attr_name, float(val))
                        logger.info(
                            "[AgentManager] (load) strategy.%s = %s", attr_name, val
                        )
                    except Exception:
                        pass

            self._params_applied = True
            self._applied_at = data.get("applied_at")
            self._last_improvement = data.get("sharpe_improvement_pct")

        logger.info(
            "[AgentManager] loaded saved config from %s (applied_at=%s)",
            CONFIG_PATH, data.get("applied_at"),
        )

    # ------------------------------------------------------------------ #
    # History persistence
    # ------------------------------------------------------------------ #

    def _load_history(self) -> None:
        if not os.path.isfile(HISTORY_PATH):
            return
        try:
            with open(HISTORY_PATH, "r", encoding="utf-8") as fh:
                self._run_history = json.load(fh)
        except Exception:
            self._run_history = []

    def _save_history(self) -> None:
        try:
            with self._lock:
                data = list(self._run_history)
            with open(HISTORY_PATH, "w", encoding="utf-8") as fh:
                json.dump(data, fh, indent=2)
        except Exception:
            pass
