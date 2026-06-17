"""
trading_env.py
==============
Gymnasium trading environment for the RL agent.

Observation (109 features, all normalised to [-1, 1] or [0, 1]):
  - 20 candles × 5 features (OHLCV normalised)     = 100
  - Technical indicators (RSI, EMA ratios, ADX, ATR) =   7
  - Portfolio state (position, unreal PnL, cash%)    =   2
                                                     ─────
                                                       109

Actions:
  0 = HOLD   (do nothing)
  1 = BUY    (open long at full risk-sized position)
  2 = SELL   (close long OR open short)

Reward:
  +  portfolio value change (normalised by initial capital)
  +  Sharpe bonus every 20 steps
  -  transaction cost on each trade
  -  excessive idle penalty (encourage activity without overtrading)
  -  drawdown penalty (protect capital)
"""

from __future__ import annotations

import math
from collections import deque
from typing import Optional, Tuple, Dict, Any

import numpy as np
import pandas as pd

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:
    import gym
    from gym import spaces


# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────
LOOKBACK        = 20          # candles in observation window
TRANSACTION_COST = 0.0003     # 0.03% per trade (spread + slippage)
MAX_STEPS       = 2000        # max steps per episode
IDLE_PENALTY    = -0.0001     # per step with no position (encourages trading)
DD_PENALTY_MULT = 2.0         # drawdown penalty multiplier


# ─────────────────────────────────────────────────────────────────────────────
# Environment
# ─────────────────────────────────────────────────────────────────────────────
class TradingEnv(gym.Env):
    """
    Single-asset trading environment.

    Parameters
    ----------
    df          : M5 OHLCV DataFrame (tz-aware UTC index)
    initial_capital : starting cash in USD
    contract_size   : units per lot (100 for gold, 100000 for forex)
    risk_per_trade  : fraction of capital to risk per trade
    symbol          : display name
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        df: pd.DataFrame,
        initial_capital: float = 10_000.0,
        contract_size: float = 100.0,
        risk_per_trade: float = 0.01,
        symbol: str = "XAUUSD",
    ):
        super().__init__()
        self.df = df.reset_index(drop=True)
        self.initial_capital = initial_capital
        self.contract_size = contract_size
        self.risk_per_trade = risk_per_trade
        self.symbol = symbol

        n_features = LOOKBACK * 5 + 7 + 2   # 109
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(n_features,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(3)   # 0=HOLD 1=BUY 2=SELL

        # episode state (set in reset)
        self._step = 0
        self._cursor = LOOKBACK
        self._cash = initial_capital
        self._position = 0          # +1 long, -1 short, 0 flat
        self._entry_price = 0.0
        self._lots = 0.0
        self._portfolio_value = initial_capital
        self._peak_value = initial_capital
        self._reward_history: deque = deque(maxlen=20)
        self._trade_count = 0

    # ── Gymnasium API ────────────────────────────────────────────────────────

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[dict] = None
    ) -> Tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        # Start at a random point so the agent generalises
        max_start = max(len(self.df) - MAX_STEPS - LOOKBACK - 1, LOOKBACK)
        self._cursor = int(self.np_random.integers(LOOKBACK, max(LOOKBACK + 1, max_start)))
        self._step = 0
        self._cash = self.initial_capital
        self._position = 0
        self._entry_price = 0.0
        self._lots = 0.0
        self._portfolio_value = self.initial_capital
        self._peak_value = self.initial_capital
        self._reward_history.clear()
        self._trade_count = 0
        return self._observe(), {}

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, dict]:
        prev_value = self._portfolio_value
        price = float(self.df.loc[self._cursor, "close"])

        reward = 0.0
        info: Dict[str, Any] = {}

        # ── execute action ───────────────────────────────────────────────────
        if action == 1 and self._position == 0:          # BUY (open long)
            self._lots = self._size_position(price)
            if self._lots > 0:
                cost = self._lots * self.contract_size * price * TRANSACTION_COST
                self._entry_price = price
                self._position = 1
                self._cash -= cost
                self._trade_count += 1
                reward -= TRANSACTION_COST

        elif action == 2 and self._position == 1:        # SELL (close long)
            pnl = (price - self._entry_price) * self._lots * self.contract_size
            cost = self._lots * self.contract_size * price * TRANSACTION_COST
            self._cash += pnl - cost
            reward += pnl / self.initial_capital
            reward -= TRANSACTION_COST
            self._position = 0
            self._entry_price = 0.0
            self._lots = 0.0
            self._trade_count += 1

        elif action == 0 and self._position == 0:        # HOLD flat
            reward += IDLE_PENALTY

        # ── mark-to-market ───────────────────────────────────────────────────
        unreal = 0.0
        if self._position != 0:
            unreal = (price - self._entry_price) * self._lots * self.contract_size * self._position
        self._portfolio_value = self._cash + unreal

        # ── drawdown penalty ─────────────────────────────────────────────────
        if self._portfolio_value > self._peak_value:
            self._peak_value = self._portfolio_value
        dd = (self._peak_value - self._portfolio_value) / self._peak_value
        if dd > 0.02:                    # >2% drawdown from peak
            reward -= dd * DD_PENALTY_MULT

        # ── Sharpe bonus every 20 steps ──────────────────────────────────────
        step_return = (self._portfolio_value - prev_value) / max(prev_value, 1e-8)
        self._reward_history.append(step_return)
        if self._step % 20 == 0 and len(self._reward_history) >= 5:
            arr = np.array(self._reward_history)
            std = arr.std() + 1e-8
            sharpe_bonus = (arr.mean() / std) * 0.01
            reward += sharpe_bonus

        # ── advance ──────────────────────────────────────────────────────────
        self._cursor += 1
        self._step += 1

        terminated = self._portfolio_value <= self.initial_capital * 0.5   # -50% ruin
        truncated  = (
            self._step >= MAX_STEPS or
            self._cursor >= len(self.df) - 1
        )

        info = {
            "portfolio_value": round(self._portfolio_value, 2),
            "position": self._position,
            "trade_count": self._trade_count,
            "drawdown": round(dd, 4),
        }

        return self._observe(), float(reward), terminated, truncated, info

    def render(self) -> None:
        price = float(self.df.loc[self._cursor, "close"])
        print(
            f"Step {self._step:4d} | Price {price:.2f} | "
            f"Portfolio ${self._portfolio_value:.2f} | "
            f"Pos {self._position:+d} | Trades {self._trade_count}"
        )

    # ── Internal helpers ─────────────────────────────────────────────────────

    def _size_position(self, price: float) -> float:
        """Fixed-fractional position sizing."""
        risk_usd = self._cash * self.risk_per_trade
        loss_per_lot = price * self.contract_size * 0.005   # assume 0.5% SL
        if loss_per_lot <= 0:
            return 0.0
        raw = risk_usd / loss_per_lot
        return max(round(round(raw / 0.01) * 0.01, 2), 0.0)

    def _observe(self) -> np.ndarray:
        start = self._cursor - LOOKBACK
        window = self.df.iloc[start: self._cursor]

        opens   = window["open"].values.astype(np.float32)
        highs   = window["high"].values.astype(np.float32)
        lows    = window["low"].values.astype(np.float32)
        closes  = window["close"].values.astype(np.float32)
        vols    = window["volume"].values.astype(np.float32)

        # normalise prices relative to last close
        ref = closes[-1] if closes[-1] != 0 else 1.0
        o_n = np.clip((opens  / ref) - 1.0, -0.1, 0.1) * 10
        h_n = np.clip((highs  / ref) - 1.0, -0.1, 0.1) * 10
        l_n = np.clip((lows   / ref) - 1.0, -0.1, 0.1) * 10
        c_n = np.clip((closes / ref) - 1.0, -0.1, 0.1) * 10
        v_avg = vols.mean() + 1e-8
        v_n   = np.clip(vols / v_avg - 1.0, -3.0, 3.0) / 3.0

        price_features = np.concatenate([o_n, h_n, l_n, c_n, v_n])  # 100

        # indicators on the window
        rsi    = self._rsi(closes, 14)
        ema9   = self._ema(closes, 9)
        ema21  = self._ema(closes, 21)
        adx    = self._adx(highs, lows, closes, 14)
        atr    = self._atr(highs, lows, closes, 14)
        bb_pos = self._bollinger_pos(closes, 20)

        ind = np.array([
            (rsi / 100.0) * 2 - 1,                          # [-1,1]
            np.clip((closes[-1] / ema9 - 1) * 20, -1, 1),   # EMA9 distance
            np.clip((closes[-1] / ema21 - 1) * 20, -1, 1),  # EMA21 distance
            np.clip(adx / 50.0 - 1, -1, 1),                 # ADX
            np.clip(atr / ref * 100 - 1, -1, 1),            # ATR %
            np.clip(bb_pos, -1, 1),                          # Bollinger position
            float(self._position),                           # [-1,0,1]
        ], dtype=np.float32)   # 7

        # portfolio state
        pnl_pct = np.clip(
            (self._portfolio_value - self.initial_capital) / self.initial_capital * 10,
            -1, 1
        )
        unreal = 0.0
        if self._position != 0 and self._entry_price > 0:
            unreal = (ref - self._entry_price) / self._entry_price
        unreal_clip = np.clip(unreal * 20, -1, 1)
        port = np.array([pnl_pct, unreal_clip], dtype=np.float32)   # 2

        obs = np.concatenate([price_features, ind, port])
        return obs.astype(np.float32)

    # ── Technical indicators (vectorised, no-dependency) ────────────────────

    @staticmethod
    def _ema(arr: np.ndarray, period: int) -> float:
        if len(arr) < period:
            return float(arr[-1])
        k = 2.0 / (period + 1)
        ema = arr[0]
        for v in arr[1:]:
            ema = v * k + ema * (1 - k)
        return float(ema)

    @staticmethod
    def _rsi(closes: np.ndarray, period: int = 14) -> float:
        if len(closes) < period + 1:
            return 50.0
        deltas = np.diff(closes)
        gains  = np.where(deltas > 0, deltas, 0.0)
        losses = np.where(deltas < 0, -deltas, 0.0)
        avg_g  = gains[-period:].mean()
        avg_l  = losses[-period:].mean()
        if avg_l == 0:
            return 100.0
        rs = avg_g / avg_l
        return float(100 - 100 / (1 + rs))

    @staticmethod
    def _atr(highs: np.ndarray, lows: np.ndarray,
             closes: np.ndarray, period: int = 14) -> float:
        if len(closes) < 2:
            return float(highs[-1] - lows[-1])
        tr_list = []
        for i in range(1, len(closes)):
            tr = max(highs[i] - lows[i],
                     abs(highs[i] - closes[i - 1]),
                     abs(lows[i]  - closes[i - 1]))
            tr_list.append(tr)
        return float(np.mean(tr_list[-period:]))

    @staticmethod
    def _adx(highs: np.ndarray, lows: np.ndarray,
             closes: np.ndarray, period: int = 14) -> float:
        if len(closes) < period + 1:
            return 20.0
        plus_dm  = np.maximum(np.diff(highs), 0)
        minus_dm = np.maximum(-np.diff(lows), 0)
        mask = plus_dm > minus_dm
        plus_dm  = np.where(mask, plus_dm, 0.0)
        minus_dm = np.where(~mask, minus_dm, 0.0)
        tr_arr = np.array([
            max(highs[i] - lows[i],
                abs(highs[i] - closes[i - 1]),
                abs(lows[i]  - closes[i - 1]))
            for i in range(1, len(closes))
        ], dtype=np.float64)
        atr14   = tr_arr[-period:].mean() + 1e-8
        pdi     = plus_dm[-period:].mean() / atr14 * 100
        mdi     = minus_dm[-period:].mean() / atr14 * 100
        dx_denom = pdi + mdi
        if dx_denom == 0:
            return 0.0
        return float(abs(pdi - mdi) / dx_denom * 100)

    @staticmethod
    def _bollinger_pos(closes: np.ndarray, period: int = 20) -> float:
        if len(closes) < period:
            return 0.0
        w = closes[-period:]
        mean = w.mean()
        std  = w.std() + 1e-8
        return float((closes[-1] - mean) / (2 * std))
