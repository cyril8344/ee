"""
broker.py
=========
Execution layer for the XAU/USD scalping bot.

Two implementations sharing one interface:

- PaperBroker : fully simulated fills against the latest market price.
  Used by default; needs no external dependency.
- MT5Broker   : thin wrapper over the MetaTrader5 Python API for live/paper
  accounts.  Only usable on Windows with the `MetaTrader5` package and a
  running terminal.  Falls back gracefully if unavailable.

Market data
-----------
`get_rates()` returns recent M5 OHLCV.  MT5 provides it natively; the paper
broker pulls 5-minute data via yfinance (GC=F proxy) and caches it briefly,
with a synthetic fallback so the engine always has something to chew on.
"""

from __future__ import annotations

import time as _time
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from typing import Optional, Dict, Any, List

import numpy as np
import pandas as pd

from risk_manager import CONTRACT_SIZE

PIP = 0.1


# --------------------------------------------------------------------------- #
# Data helper shared by paper broker
# --------------------------------------------------------------------------- #
class MarketData:
    """Cached 5-minute market data feed (data_provider + synthetic fallback)."""

    def __init__(self, ttl_seconds: int = 60, symbol: str = "XAUUSD"):
        self.ttl = ttl_seconds
        self.symbol = symbol
        self._cache: Optional[pd.DataFrame] = None
        self._fetched_at: float = 0.0
        self._lock = threading.Lock()

    def get_m5(self, bars: int = 500) -> pd.DataFrame:
        with self._lock:
            now = _time.time()
            if self._cache is not None and (now - self._fetched_at) < self.ttl:
                return self._cache.tail(bars).copy()
            df = self._fetch()
            self._cache = df
            self._fetched_at = now
            return df.tail(bars).copy()

    def _fetch(self) -> pd.DataFrame:
        try:
            import data_provider
            df, _provider = data_provider.get_m5(bars=500, symbol=self.symbol)
            if df is not None and len(df) > 0:
                return df
        except Exception:
            pass
        return self._synthetic()

    def _synthetic(self) -> pd.DataFrame:
        end = pd.Timestamp.now(tz="UTC").floor("5min")
        idx = pd.date_range(end - pd.Timedelta(days=5), end, freq="5min", tz="UTC")
        idx = idx[idx.weekday < 5]
        n = len(idx)
        rng = np.random.default_rng(int(end.timestamp()) // 300)
        if self.symbol == "EURUSD":
            price0 = 1.08
            vol = 0.00012
            spread_base = 0.00003
        else:
            price0 = 2000.0
            vol = 0.0008
            spread_base = 0.2
        rets = rng.normal(0, vol, n)
        close = price0 * np.exp(np.cumsum(rets))
        spread = np.abs(rng.normal(0, spread_base * 2.5, n)) + spread_base
        open_ = np.concatenate([[price0], close[:-1]])
        df = pd.DataFrame({
            "open": open_,
            "high": np.maximum(close + spread, np.maximum(open_, close)),
            "low": np.minimum(close - spread, np.minimum(open_, close)),
            "close": close,
            "volume": np.abs(rng.normal(1000, 250, n)).round(),
        }, index=idx)
        df.index.name = "time"
        return df


# --------------------------------------------------------------------------- #
# Position model
# --------------------------------------------------------------------------- #
@dataclass
class Position:
    ticket: int
    direction: str
    entry: float
    volume: float
    stop_loss: float
    take_profit1: float
    take_profit2: float
    open_time: datetime
    tp1_done: bool = False
    remaining: float = 0.0
    realised: float = 0.0
    risk_amount: float = 0.0
    session: str = ""
    meta: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.remaining == 0.0:
            self.remaining = self.volume

    def unrealised_pnl(self, price: float, contract_size: float = CONTRACT_SIZE) -> float:
        sign = 1.0 if self.direction == "long" else -1.0
        return (price - self.entry) * sign * contract_size * self.remaining + self.realised


# --------------------------------------------------------------------------- #
# Base interface
# --------------------------------------------------------------------------- #
class BaseBroker:
    name = "base"

    def connected(self) -> bool:
        raise NotImplementedError

    def get_rates_m5(self, bars: int = 500) -> pd.DataFrame:
        raise NotImplementedError

    def get_price(self) -> float:
        raise NotImplementedError

    def market_order(self, direction: str, volume: float, sl: float,
                     tp1: float, tp2: float, session: str = "",
                     meta: Optional[Dict[str, Any]] = None) -> Position:
        raise NotImplementedError

    def update_position(self, pos: Position) -> Optional[Dict[str, Any]]:
        """Check SL/TP/partials against current price.  Returns close info
        dict (with 'reason','exit_price','pnl','closed') or None."""
        raise NotImplementedError

    def close_position(self, pos: Position, reason: str = "manual") -> Dict[str, Any]:
        raise NotImplementedError


# --------------------------------------------------------------------------- #
# Paper broker
# --------------------------------------------------------------------------- #
class PaperBroker(BaseBroker):
    name = "paper"

    def __init__(self, spread_pips: float = 0.3, slippage_pips: float = 0.1,
                 symbol: str = "XAUUSD", contract_size: float = 100.0):
        self.spread = spread_pips * PIP
        self.slippage = slippage_pips * PIP
        self.contract_size = contract_size
        self.data = MarketData(symbol=symbol)
        self._ticket = 1000

    def connected(self) -> bool:
        return True

    def get_rates_m5(self, bars: int = 500) -> pd.DataFrame:
        return self.data.get_m5(bars)

    def get_price(self) -> float:
        df = self.data.get_m5(2)
        return float(df["close"].iloc[-1])

    def market_order(self, direction, volume, sl, tp1, tp2, session="", meta=None, risk_amount=0.0) -> Position:
        self._ticket += 1
        price = self.get_price()
        # fill with spread + slippage
        if direction == "long":
            fill = price + self.spread + self.slippage
        else:
            fill = price - self.spread - self.slippage
        return Position(
            ticket=self._ticket, direction=direction, entry=fill,
            volume=volume, stop_loss=sl, take_profit1=tp1,
            take_profit2=tp2, open_time=datetime.now(timezone.utc),
            remaining=volume, risk_amount=risk_amount, session=session, meta=meta or {},
        )

    def update_position(self, pos: Position) -> Optional[Dict[str, Any]]:
        price = self.get_price()
        direction = pos.direction
        sign = 1.0 if direction == "long" else -1.0

        def pnl_for(p, lots):
            return (p - pos.entry) * sign * self.contract_size * lots

        # Emergency stop: never lose more than 3× the intended risk
        if pos.risk_amount > 0:
            unrealised = pnl_for(price, pos.remaining) + pos.realised
            if unrealised < -(pos.risk_amount * 3):
                pos.realised += pnl_for(price - self.slippage * sign, pos.remaining)
                return {"closed": True, "reason": "emergency_stop",
                        "exit_price": price, "pnl": pos.realised}

        # TP1 partial (60%)
        if not pos.tp1_done:
            hit = price >= pos.take_profit1 if direction == "long" else price <= pos.take_profit1
            if hit:
                lots60 = round(min(pos.volume * 0.6, pos.remaining), 2)
                pos.realised += pnl_for(pos.take_profit1 - self.slippage * sign, lots60)
                pos.remaining = round(pos.remaining - lots60, 2)
                pos.tp1_done = True
                # Move stop loss to breakeven after TP1
                pos.stop_loss = pos.entry
                if pos.remaining < 0.01:
                    return {"closed": True, "reason": "tp1",
                            "exit_price": pos.take_profit1, "pnl": pos.realised}
                return {"closed": False, "reason": "tp1_partial",
                        "exit_price": pos.take_profit1, "pnl": pos.realised}

        # Stop loss
        hit_sl = price <= pos.stop_loss if direction == "long" else price >= pos.stop_loss
        if hit_sl:
            pos.realised += pnl_for(pos.stop_loss - self.slippage * sign, pos.remaining)
            return {"closed": True,
                    "reason": "sl" if not pos.tp1_done else "sl_after_tp1",
                    "exit_price": pos.stop_loss, "pnl": pos.realised}

        # TP2 + trailing stop after TP1
        if pos.tp1_done:
            hit_tp2 = price >= pos.take_profit2 if direction == "long" else price <= pos.take_profit2
            if hit_tp2:
                pos.realised += pnl_for(pos.take_profit2 - self.slippage * sign, pos.remaining)
                return {"closed": True, "reason": "tp2",
                        "exit_price": pos.take_profit2, "pnl": pos.realised}

            # Trailing stop: if price moved 0.5×ATR in our favour, trail SL at 0.3×ATR behind price
            atr_val = pos.meta.get("atr", 0) or 0
            if atr_val > 0:
                if direction == "long":
                    trail_trigger = pos.entry + 0.5 * atr_val
                    if price >= trail_trigger:
                        new_sl = price - 0.3 * atr_val
                        if new_sl > pos.stop_loss:
                            pos.stop_loss = new_sl
                else:
                    trail_trigger = pos.entry - 0.5 * atr_val
                    if price <= trail_trigger:
                        new_sl = price + 0.3 * atr_val
                        if new_sl < pos.stop_loss:
                            pos.stop_loss = new_sl

        return None

    def close_position(self, pos: Position, reason: str = "manual") -> Dict[str, Any]:
        price = self.get_price()
        sign = 1.0 if pos.direction == "long" else -1.0
        fill = price - self.slippage * sign
        pnl = pos.realised + (fill - pos.entry) * sign * self.contract_size * pos.remaining
        pos.remaining = 0.0
        return {"closed": True, "reason": reason, "exit_price": fill, "pnl": pnl}


# --------------------------------------------------------------------------- #
# MetaTrader 5 broker
# --------------------------------------------------------------------------- #
class MT5Broker(BaseBroker):
    name = "mt5"

    def __init__(self, symbol: str = "XAUUSD"):
        self.symbol = symbol
        self._mt5 = None
        self._ok = False
        self._init()

    def _init(self):
        try:
            import MetaTrader5 as mt5  # type: ignore
            self._mt5 = mt5
            if mt5.initialize():
                self._ok = True
                info = mt5.symbol_info(self.symbol)
                if info is not None and not info.visible:
                    mt5.symbol_select(self.symbol, True)
        except Exception:
            self._ok = False

    def connected(self) -> bool:
        return self._ok

    def get_rates_m5(self, bars: int = 500) -> pd.DataFrame:
        if not self._ok:
            raise RuntimeError("MT5 not connected")
        mt5 = self._mt5
        rates = mt5.copy_rates_from_pos(self.symbol, mt5.TIMEFRAME_M5, 0, bars)
        df = pd.DataFrame(rates)
        df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
        df = df.set_index("time")
        df = df.rename(columns={"tick_volume": "volume"})
        return df[["open", "high", "low", "close", "volume"]]

    def get_price(self) -> float:
        tick = self._mt5.symbol_info_tick(self.symbol)
        return float((tick.bid + tick.ask) / 2.0)

    def market_order(self, direction, volume, sl, tp1, tp2, session="", meta=None) -> Position:
        mt5 = self._mt5
        tick = mt5.symbol_info_tick(self.symbol)
        price = tick.ask if direction == "long" else tick.bid
        order_type = mt5.ORDER_TYPE_BUY if direction == "long" else mt5.ORDER_TYPE_SELL
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": self.symbol,
            "volume": float(volume),
            "type": order_type,
            "price": price,
            "sl": float(sl),
            "tp": float(tp2),
            "deviation": 20,
            "magic": 770077,
            "comment": "xau-scalper",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        result = mt5.order_send(request)
        ticket = getattr(result, "order", 0) or 0
        return Position(
            ticket=ticket, direction=direction, entry=float(price),
            volume=volume, stop_loss=sl, take_profit1=tp1, take_profit2=tp2,
            open_time=datetime.now(timezone.utc), remaining=volume,
            session=session, meta=meta or {},
        )

    def _send_partial_close(self, pos: Position, lots: float, price: float) -> None:
        mt5 = self._mt5
        order_type = mt5.ORDER_TYPE_SELL if pos.direction == "long" else mt5.ORDER_TYPE_BUY
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": self.symbol,
            "volume": float(round(lots, 2)),
            "type": order_type,
            "position": pos.ticket,
            "price": price,
            "deviation": 20,
            "magic": 770077,
            "comment": "tp1-partial",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        mt5.order_send(request)

    def _update_sl(self, pos: Position, new_sl: float) -> None:
        mt5 = self._mt5
        request = {
            "action": mt5.TRADE_ACTION_SLTP,
            "symbol": self.symbol,
            "position": pos.ticket,
            "sl": float(new_sl),
            "tp": float(pos.take_profit2),
        }
        mt5.order_send(request)

    def update_position(self, pos: Position) -> Optional[Dict[str, Any]]:
        price = self.get_price()
        sign = 1.0 if pos.direction == "long" else -1.0

        # TP1 partial close (60%) — sent as real MT5 order
        if not pos.tp1_done:
            hit = price >= pos.take_profit1 if pos.direction == "long" else price <= pos.take_profit1
            if hit:
                lots60 = round(min(pos.volume * 0.6, pos.remaining), 2)
                tick = self._mt5.symbol_info_tick(self.symbol)
                fill_price = tick.bid if pos.direction == "long" else tick.ask
                self._send_partial_close(pos, lots60, fill_price)
                pnl60 = (fill_price - pos.entry) * sign * CONTRACT_SIZE * lots60
                pos.realised += pnl60
                pos.remaining = round(pos.remaining - lots60, 2)
                pos.tp1_done = True
                # Move SL to breakeven on MT5 server
                pos.stop_loss = pos.entry
                self._update_sl(pos, pos.entry)
                if pos.remaining < 0.01:
                    return {"closed": True, "reason": "tp1",
                            "exit_price": fill_price, "pnl": pos.realised}
                return {"closed": False, "reason": "tp1_partial",
                        "exit_price": fill_price, "pnl": pos.realised}

        # Trailing stop after TP1 — update SL on MT5 server
        if pos.tp1_done:
            atr_val = pos.meta.get("atr", 0) or 0
            if atr_val > 0:
                if pos.direction == "long":
                    trail_trigger = pos.entry + 0.5 * atr_val
                    if price >= trail_trigger:
                        new_sl = price - 0.3 * atr_val
                        if new_sl > pos.stop_loss:
                            pos.stop_loss = new_sl
                            self._update_sl(pos, new_sl)
                else:
                    trail_trigger = pos.entry - 0.5 * atr_val
                    if price <= trail_trigger:
                        new_sl = price + 0.3 * atr_val
                        if new_sl < pos.stop_loss:
                            pos.stop_loss = new_sl
                            self._update_sl(pos, new_sl)

        return None

    def close_position(self, pos: Position, reason: str = "manual") -> Dict[str, Any]:
        mt5 = self._mt5
        tick = mt5.symbol_info_tick(self.symbol)
        price = tick.bid if pos.direction == "long" else tick.ask
        order_type = mt5.ORDER_TYPE_SELL if pos.direction == "long" else mt5.ORDER_TYPE_BUY
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": self.symbol,
            "volume": float(pos.remaining),
            "type": order_type,
            "position": pos.ticket,
            "price": price,
            "deviation": 20,
            "magic": 770077,
            "comment": f"close-{reason}",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        mt5.order_send(request)
        sign = 1.0 if pos.direction == "long" else -1.0
        pnl = pos.realised + (price - pos.entry) * sign * CONTRACT_SIZE * pos.remaining
        pos.remaining = 0.0
        return {"closed": True, "reason": reason, "exit_price": float(price), "pnl": pnl}


def make_broker(mode: str = "paper", symbol: str = "XAUUSD",
                spread_pips: float = 0.3, slippage_pips: float = 0.1,
                contract_size: float = 100.0) -> BaseBroker:
    """Factory: returns an MT5 broker for live mode if available, else paper."""
    if mode == "live":
        mt5 = MT5Broker(symbol)
        if mt5.connected():
            return mt5
        # fall back to paper if MT5 unavailable
    return PaperBroker(spread_pips, slippage_pips, symbol=symbol, contract_size=contract_size)
