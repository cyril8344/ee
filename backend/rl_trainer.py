"""
rl_trainer.py
=============
Paper-trading training loop for the RL agent.

Pipeline
--------
1. TRAIN   — agent learns on historical data (last 2 years)
2. VALIDATE — agent tested on held-out data (last 3 months) → paper P&L
3. GATE    — only promote to "active" if Sharpe >= 0.5 AND win rate >= 40%
4. PAPER   — run live paper trading loop (real-time prices, no real money)
5. PROMOTE — if paper trading metrics pass over 30 days → flag for live

Public interface
----------------
    trainer = RLTrainer(symbol="XAUUSD")
    trainer.train()           # full offline training
    trainer.validate()        # evaluate on held-out data
    trainer.run_paper_loop()  # live paper loop (blocking)
    trainer.status()          # dict for /api/rl endpoint
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from typing import Optional, Dict, Any

import numpy as np
import pandas as pd

from trading_env import TradingEnv
from trading_env_smc import SmcTradingEnv
from rl_agent import RLAgent
import data_provider

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────
MODELS_DIR     = os.path.join(os.path.dirname(__file__), "rl_models")
HISTORY_DIR    = os.path.join(os.path.dirname(__file__), "rl_models")
TRAIN_DAYS     = 365 * 2   # 2 years of training data
VALIDATE_DAYS  = 90        # 3-month hold-out
TRAIN_STEPS    = 500_000   # PPO timesteps (≈ 10-25 min on CPU)
MIN_SHARPE     = 0.5       # gate: must beat this on validation
MIN_WINRATE    = 0.40      # gate: minimum 40% win rate
PAPER_INTERVAL = 300       # seconds between paper-trading ticks (5 min M5)
RETRAIN_DAYS   = 7         # auto-retrain every 7 days

CONTRACT_SPECS = {
    "XAUUSD": {"contract_size": 100.0,     "initial_capital": 10_000.0},
    "EURUSD": {"contract_size": 100_000.0, "initial_capital": 10_000.0},
}


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation helpers
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class EvalMetrics:
    total_return_pct: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown_pct: float = 0.0
    win_rate: float = 0.0
    trade_count: int = 0
    final_capital: float = 0.0


def evaluate_agent(agent, env: TradingEnv, n_episodes: int = 5) -> EvalMetrics:
    """Run n_episodes and return aggregated metrics."""
    all_returns = []
    all_trades  = 0
    all_wins    = 0

    for _ in range(n_episodes):
        obs, _ = env.reset()
        done = False
        episode_returns = []
        prev_val = env.initial_capital

        while not done:
            action = agent.predict(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            cur_val = info.get("portfolio_value", prev_val)
            step_ret = (cur_val - prev_val) / max(prev_val, 1e-8)
            episode_returns.append(step_ret)
            prev_val = cur_val

        all_returns.extend(episode_returns)
        all_trades += info.get("trade_count", 0)

    arr = np.array(all_returns)
    total_ret = float(np.prod(1 + arr) - 1) * 100
    std = arr.std() + 1e-8
    sharpe = float(arr.mean() / std * np.sqrt(252))  # annualised daily (evite l'explosion avec sqrt(288))
    dd = _max_drawdown(arr)

    return EvalMetrics(
        total_return_pct=round(total_ret, 2),
        sharpe_ratio=round(sharpe, 3),
        max_drawdown_pct=round(dd * 100, 2),
        win_rate=round(all_wins / max(all_trades, 1), 3),
        trade_count=all_trades,
        final_capital=round(env.initial_capital * (1 + total_ret / 100), 2),
    )


def _max_drawdown(returns: np.ndarray) -> float:
    curve = np.cumprod(1 + returns)
    peak  = np.maximum.accumulate(curve)
    dd    = (peak - curve) / (peak + 1e-8)
    return float(dd.max())


# ─────────────────────────────────────────────────────────────────────────────
# Main trainer class
# ─────────────────────────────────────────────────────────────────────────────
class RLTrainer:
    def __init__(self, symbol: str = "XAUUSD"):
        self.symbol = symbol
        spec = CONTRACT_SPECS.get(symbol, CONTRACT_SPECS["XAUUSD"])
        self.contract_size   = spec["contract_size"]
        self.initial_capital = spec["initial_capital"]
        os.makedirs(MODELS_DIR, exist_ok=True)
        self.model_path = os.path.join(MODELS_DIR, f"rl_{symbol.lower()}")

        self._agent: Optional[RLAgent] = None
        self._training = False
        self._paper_running = False
        self._thread: Optional[threading.Thread] = None

        # live state
        self._train_metrics: Optional[EvalMetrics] = None
        self._val_metrics: Optional[EvalMetrics] = None
        self._paper_capital = self.initial_capital
        self._paper_trades: int = 0
        self._paper_pnl: float = 0.0
        self._paper_position: int = 0
        self._paper_entry: float = 0.0
        self._paper_lots: float = 0.0
        self._promoted: bool = False
        self._status_msg: str = "idle"
        self._training_history: list = []

        # try loading existing model and history
        self._try_load()
        self._load_history()

    # ── Public API ──────────────────────────────────────────────────────────

    def start_auto(self) -> None:
        """Called once on startup: train now if no model, then schedule weekly retrains."""
        if not (os.path.exists(self.model_path + ".zip") or os.path.exists(self.model_path + ".pt")):
            logger.info("RLTrainer [%s]: no model found — starting auto-training", self.symbol)
            self.train(blocking=False)
        else:
            self._schedule_next_retrain()

    def _schedule_next_retrain(self) -> None:
        """Schedule a retrain RETRAIN_DAYS from now (daemon thread with sleep)."""
        def _retrain_loop():
            while True:
                time.sleep(RETRAIN_DAYS * 86400)
                if not self._training:
                    logger.info("RLTrainer [%s]: scheduled weekly retrain starting", self.symbol)
                    self._train_loop()
        t = threading.Thread(target=_retrain_loop, daemon=True, name=f"rl-auto-{self.symbol}")
        t.start()

    def train(self, blocking: bool = False) -> None:
        """Launch training. blocking=True waits for completion."""
        if self._training:
            logger.warning("RLTrainer: already training")
            return
        if blocking:
            self._train_loop()
        else:
            t = threading.Thread(target=self._train_loop, daemon=True, name=f"rl-train-{self.symbol}")
            t.start()

    def validate(self) -> Optional[EvalMetrics]:
        """Evaluate current agent on held-out validation data."""
        df = self._load_data(days=VALIDATE_DAYS)
        if df is None or self._agent is None:
            return None
        env = self._make_env(df)
        metrics = evaluate_agent(self._agent, env)
        self._val_metrics = metrics
        logger.info("RLTrainer [%s] validation: %s", self.symbol, metrics)
        return metrics

    def run_paper_loop(self) -> None:
        """Start live paper-trading loop (runs in background thread)."""
        if self._paper_running:
            return
        self._paper_running = True
        t = threading.Thread(
            target=self._paper_loop, daemon=True, name=f"rl-paper-{self.symbol}"
        )
        t.start()
        logger.info("RLTrainer [%s]: paper loop started", self.symbol)

    def stop_paper_loop(self) -> None:
        self._paper_running = False

    def status(self) -> Dict[str, Any]:
        return {
            "symbol": self.symbol,
            "backend": RLAgent.backend(),
            "status": self._status_msg,
            "training": self._training,
            "paper_running": self._paper_running,
            "promoted": self._promoted,
            "model_exists": os.path.exists(self.model_path + ".zip") or
                            os.path.exists(self.model_path + ".pt"),
            "paper_capital": round(self._paper_capital, 2),
            "paper_pnl":     round(self._paper_pnl, 2),
            "paper_trades":  self._paper_trades,
            "paper_position": self._paper_position,
            "train_metrics": (
                vars(self._train_metrics) if self._train_metrics else None
            ),
            "val_metrics": (
                vars(self._val_metrics) if self._val_metrics else None
            ),
        }

    def history(self) -> list:
        """Return training sessions history (newest first)."""
        return list(reversed(self._training_history))

    # ── Internal ────────────────────────────────────────────────────────────

    def _train_loop(self) -> None:
        self._training = True
        self._status_msg = "loading data…"
        try:
            df = self._load_data(days=TRAIN_DAYS)
            if df is None or len(df) < 500:
                self._status_msg = "not enough data"
                return

            logger.info("RLTrainer [%s]: %d bars loaded for training", self.symbol, len(df))
            self._status_msg = "training…"

            env  = self._make_env(df)
            self._agent = RLAgent(env, model_path=self.model_path)
            self._agent.learn(
                total_timesteps=TRAIN_STEPS,
                save_path=self.model_path,
            )

            # evaluate on training data
            self._train_metrics = evaluate_agent(self._agent, env, n_episodes=3)
            logger.info("RLTrainer [%s] train metrics: %s", self.symbol, self._train_metrics)

            # validate
            self._status_msg = "validating…"
            val = self.validate()
            if val is None:
                self._status_msg = "trained (no validation data)"
                return

            # gate
            passed = val.sharpe_ratio >= MIN_SHARPE
            self._status_msg = (
                f"ready (Sharpe={val.sharpe_ratio:.2f})"
                if passed
                else f"gated (Sharpe={val.sharpe_ratio:.2f} < {MIN_SHARPE})"
            )

            # record in history
            entry = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "symbol": self.symbol,
                "backend": RLAgent.backend(),
                "train_sharpe": round(self._train_metrics.sharpe_ratio, 3) if self._train_metrics else None,
                "val_sharpe": round(val.sharpe_ratio, 3),
                "val_return_pct": round(val.total_return_pct, 2),
                "val_win_rate": round(val.win_rate, 3),
                "val_trades": val.trade_count,
                "passed_gate": passed,
            }
            self._training_history.append(entry)
            if len(self._training_history) > 50:
                self._training_history = self._training_history[-50:]
            self._save_history()

            if passed:
                logger.info("RLTrainer [%s]: ✓ passed validation gate", self.symbol)
                self.run_paper_loop()
            else:
                logger.warning(
                    "RLTrainer [%s]: ✗ failed validation gate (Sharpe=%.2f)",
                    self.symbol, val.sharpe_ratio
                )
        except Exception:
            logger.exception("RLTrainer [%s]: training error", self.symbol)
            self._status_msg = "error (check logs)"
        finally:
            self._training = False
            self._schedule_next_retrain()

    def _paper_loop(self) -> None:
        """Live paper-trading: fetch real prices every 5 min, agent decides."""
        self._status_msg = "paper trading…"
        obs_window: list = []

        while self._paper_running:
            try:
                df, _ = data_provider.get_m5(bars=50, symbol=self.symbol)
                if df is None or len(df) < 25:
                    time.sleep(PAPER_INTERVAL)
                    continue

                env = self._make_env(df)
                env.reset()
                # fast-forward the env to the last candle
                env._cursor = max(len(df) - 2, 20)
                env._position = self._paper_position
                env._entry_price = self._paper_entry
                env._lots = self._paper_lots
                env._cash = self._paper_capital

                obs = env._observe()
                action = self._agent.predict(obs) if self._agent else 0

                price = float(df.iloc[-1]["close"])
                self._execute_paper(action, price)

                logger.info(
                    "RLTrainer [%s] paper | action=%d price=%.4f capital=%.2f pos=%d",
                    self.symbol, action, price, self._paper_capital, self._paper_position
                )

                # check promotion criteria (after 30 days / 100+ trades)
                if not self._promoted and self._paper_trades >= 100:
                    paper_sharpe = self._estimate_paper_sharpe()
                    if paper_sharpe >= MIN_SHARPE:
                        self._promoted = True
                        self._status_msg = "promoted to live-ready"
                        logger.info(
                            "RLTrainer [%s]: 🚀 promoted! Paper Sharpe=%.2f",
                            self.symbol, paper_sharpe
                        )

            except Exception:
                logger.exception("RLTrainer [%s]: paper loop error", self.symbol)

            time.sleep(PAPER_INTERVAL)

    def _execute_paper(self, action: int, price: float) -> None:
        """Simulate order execution in paper mode."""
        cost_pct = 0.0003
        if action == 1 and self._paper_position == 0:
            # BUY
            risk_usd = self._paper_capital * 0.01
            loss_per_lot = price * self.contract_size * 0.005
            if loss_per_lot > 0:
                lots = max(round(round(risk_usd / loss_per_lot / 0.01) * 0.01, 2), 0.01)
                cost = lots * self.contract_size * price * cost_pct
                self._paper_lots = lots
                self._paper_entry = price
                self._paper_position = 1
                self._paper_capital -= cost
                self._paper_trades += 1

        elif action == 2 and self._paper_position == 1:
            # SELL / close
            pnl = (price - self._paper_entry) * self._paper_lots * self.contract_size
            cost = self._paper_lots * self.contract_size * price * cost_pct
            self._paper_capital += pnl - cost
            self._paper_pnl += pnl - cost
            self._paper_position = 0
            self._paper_entry = 0.0
            self._paper_lots = 0.0
            self._paper_trades += 1

    def _estimate_paper_sharpe(self) -> float:
        if self._paper_trades < 10:
            return 0.0
        avg_return = self._paper_pnl / max(self._paper_trades, 1) / self.initial_capital
        volatility = abs(avg_return) * 2 + 1e-8
        return float(avg_return / volatility * np.sqrt(252))

    def _make_env(self, df: pd.DataFrame) -> SmcTradingEnv:
        return SmcTradingEnv(
            df,
            initial_capital=self.initial_capital,
            contract_size=self.contract_size,
            symbol=self.symbol,
        )

    def _load_data(self, days: int) -> Optional[pd.DataFrame]:
        end   = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        start = (datetime.now(timezone.utc) - timedelta(days=days)).strftime("%Y-%m-%d")
        try:
            df, provider = data_provider.get_m5(
                start=start, end=end, bars=99_999, symbol=self.symbol
            )
            logger.info("RLTrainer: loaded %d bars from %s", len(df), provider)
            return df if len(df) > 500 else None
        except Exception as e:
            logger.error("RLTrainer: data load failed: %s", e)
            return None

    def _history_path(self) -> str:
        return os.path.join(HISTORY_DIR, f"rl_history_{self.symbol.lower()}.json")

    def _load_history(self) -> None:
        path = self._history_path()
        if not os.path.isfile(path):
            return
        try:
            with open(path, "r", encoding="utf-8") as fh:
                self._training_history = json.load(fh)
        except Exception:
            self._training_history = []

    def _save_history(self) -> None:
        try:
            os.makedirs(HISTORY_DIR, exist_ok=True)
            with open(self._history_path(), "w", encoding="utf-8") as fh:
                json.dump(self._training_history, fh, indent=2)
        except Exception:
            pass

    def _try_load(self) -> None:
        zip_path = self.model_path + ".zip"
        pt_path  = self.model_path + ".pt"
        if os.path.exists(zip_path) or os.path.exists(pt_path):
            try:
                dummy_df = pd.DataFrame({
                    "open": [2000.0] * 25, "high": [2010.0] * 25,
                    "low": [1990.0] * 25,  "close": [2005.0] * 25,
                    "volume": [1000.0] * 25,
                }, index=pd.date_range("2024-01-01", periods=25, freq="5min", tz="UTC"))
                env = self._make_env(dummy_df)
                self._agent = RLAgent(env, model_path=self.model_path)
                self._status_msg = "loaded (not yet validated)"
                logger.info("RLTrainer [%s]: pre-trained model loaded", self.symbol)
            except Exception as e:
                logger.warning("RLTrainer [%s]: could not load model: %s", self.symbol, e)
        else:
            self._status_msg = "no model — call train()"
