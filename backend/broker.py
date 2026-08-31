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
import strategy


# --------------------------------------------------------------------------- #
# Data helper shared by paper broker
# --------------------------------------------------------------------------- #
# Bougies M5 récupérées par fetch et gardées en cache. 5000 = plafond `outputsize`
# de Twelve Data : la facturation se fait à la requête, pas à la bougie, donc élargir
# la fenêtre ne coûte aucun appel supplémentaire. Il en faut autant parce que les
# frames H1/H4 sont resamplées depuis cette série : à 2000 bougies M5 on n'obtenait
# que ~166 bougies H1, insuffisant pour une EMA200 H1 convergée (voir CONTEXT_BARS_M5
# dans main.py).
LIVE_FETCH_BARS = 5000


class MarketData:
    """Cached 5-minute market data feed (data_provider + synthetic fallback)."""

    def __init__(self, ttl_seconds: int = 300, symbol: str = "XAUUSD",
                 data_symbol: Optional[str] = None, price_scale: float = 1.0):
        self.ttl = ttl_seconds
        self.symbol = symbol
        # data_symbol : ticker réellement fetché si différent du symbole interne
        # (ex. "SPY" pour un marché interne "ES" — proxy gratuit, voir pretrain_es.py).
        # price_scale : facteur appliqué aux prix après fetch pour retrouver l'échelle
        # du symbole interne (SPY ×10 ≈ échelle ES). 1.0 = comportement inchangé.
        self.data_symbol = data_symbol or symbol
        self.price_scale = price_scale
        self._cache: Optional[pd.DataFrame] = None
        self._fetched_at: float = 0.0
        self._lock = threading.Lock()
        self.provider: Optional[str] = None  # dernier provider utilisé

    def get_m5(self, bars: int = 500) -> pd.DataFrame:
        with self._lock:
            now = _time.time()
            # Retry synthetic data quickly so real data is picked up as soon as it returns.
            ttl = 15.0 if self.provider == "synthetic" else self.ttl
            if self._cache is not None and (now - self._fetched_at) < ttl:
                return self._cache.tail(bars).copy()
            df = self._fetch()
            self._cache = df
            self._fetched_at = now
            return df.tail(bars).copy()

    def _fetch(self) -> pd.DataFrame:
        try:
            import data_provider
            df, provider = data_provider.get_m5(bars=LIVE_FETCH_BARS, symbol=self.data_symbol)
            if df is not None and len(df) > 0:
                self.provider = provider
                return self._scale(df)
        except Exception:
            pass
        self.provider = "synthetic"
        return self._scale(self._synthetic())

    def _scale(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.price_scale == 1.0:
            return df
        df = df.copy()
        for col in ("open", "high", "low", "close"):
            if col in df.columns:
                df[col] = df[col] * self.price_scale
        return df

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
        elif self.symbol == "ES":
            price0 = 580.0   # échelle SPY — mis à l'échelle ×10 par _scale() ensuite
            vol = 0.0003
            spread_base = 0.03
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
    mfe: float = 0.0
    tp1_bar_time: Optional[Any] = None
    entry_bar_time: Optional[Any] = None

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

    def close_position(self, pos: Position, reason: str = "manual",
                       price: Optional[float] = None) -> Dict[str, Any]:
        raise NotImplementedError


# --------------------------------------------------------------------------- #
# Paper broker
# --------------------------------------------------------------------------- #
class PaperBroker(BaseBroker):
    name = "paper"

    def __init__(self, spread_pips: float = 0.3, slippage_pips: float = 0.1,
                 symbol: str = "XAUUSD", contract_size: float = 100.0,
                 pip_size: float = 0.1, data_symbol: Optional[str] = None,
                 price_scale: float = 1.0):
        # pip_size is the price value of 1 pip (XAU: 0.1, EUR/USD: 0.0001)
        self.spread = spread_pips * pip_size
        self.slippage = slippage_pips * pip_size
        self.contract_size = contract_size
        self.data = MarketData(symbol=symbol, data_symbol=data_symbol, price_scale=price_scale)
        self._ticket = 1000

    def connected(self) -> bool:
        return True

    def is_synthetic(self) -> bool:
        """True si les données live proviennent du générateur synthétique (non fiables)."""
        return self.data.provider == "synthetic"

    def get_rates_m5(self, bars: int = 500) -> pd.DataFrame:
        return self.data.get_m5(bars)

    def get_price(self) -> float:
        df = self.data.get_m5(2)
        return float(df["close"].iloc[-1])

    def market_order(self, direction, volume, sl, tp1, tp2, session="", meta=None, risk_amount=0.0) -> Position:
        self._ticket += 1
        df = self.data.get_m5(2)
        bar = df.iloc[-1]
        price = float(bar["close"])
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
            entry_bar_time=bar.name,
        )

    def update_position(self, pos: Position) -> Optional[Dict[str, Any]]:
        df = self.data.get_m5(2)
        bar = df.iloc[-1]
        bar_time = bar.name
        # Ne jamais résoudre un exit sur la MÊME bougie que l'entrée : cette bougie
        # peut déjà être complète au moment du fill (cache MarketData jusqu'à 300s),
        # donc son high/low reflète aussi du mouvement ANTÉRIEUR à l'ouverture réelle
        # du trade — un TP1/SL "touché" dessus n'est pas un vrai mouvement de prix
        # après entrée. backtest.py/pretrain_es.py (validés en walk-forward) ne
        # vérifient jamais l'exit sur la bougie d'entrée non plus (boucle bar-par-bar :
        # l'exit n'est évalué qu'à partir de l'itération SUIVANTE) — même garantie ici.
        if pos.entry_bar_time is not None and bar_time == pos.entry_bar_time:
            return None
        price = float(bar["close"])
        bar_high = float(bar["high"])
        bar_low = float(bar["low"])
        direction = pos.direction
        sign = 1.0 if direction == "long" else -1.0

        def pnl_for(p, lots):
            return (p - pos.entry) * sign * self.contract_size * lots

        # Emergency stop: never lose more than 2× the intended risk
        if pos.risk_amount > 0:
            unrealised = pnl_for(price, pos.remaining) + pos.realised
            if unrealised < -(pos.risk_amount * 2):
                pos.realised += pnl_for(price - self.slippage * sign, pos.remaining)
                return {"closed": True, "reason": "emergency_stop",
                        "exit_price": price, "pnl": pos.realised}

        # Update MFE avec le high/low intrabar (plus précis que le close seul)
        if direction == "long":
            pos.mfe = max(pos.mfe, bar_high - pos.entry)
        else:
            pos.mfe = max(pos.mfe, pos.entry - bar_low)

        # Early exit: sans conviction après EARLY_EXIT_MINUTES — MFE < EARLY_EXIT_MFE_R × R
        if not pos.tp1_done:
            elapsed_min = (datetime.now(timezone.utc) - pos.open_time).total_seconds() / 60
            risk_dist = abs(pos.entry - pos.stop_loss)
            if (elapsed_min >= strategy.EARLY_EXIT_MINUTES and risk_dist > 0
                    and pos.mfe / risk_dist < strategy.EARLY_EXIT_MFE_R):
                pos.realised += pnl_for(price - self.slippage * sign, pos.remaining)
                return {"closed": True, "reason": "early_exit",
                        "exit_price": price, "pnl": pos.realised}

        # TP1 — sortie 50% (ou 100% si tp1_close_all) à 0.7R
        # Utilise bar_high/bar_low pour capturer les touches intrabar (évite les faux SL)
        if not pos.tp1_done:
            hit = bar_high >= pos.take_profit1 if direction == "long" else bar_low <= pos.take_profit1
            if hit:
                close_ratio = 1.0 if pos.meta.get("tp1_close_all") else 0.5
                lots50 = round(min(pos.volume * close_ratio, pos.remaining), 2)
                if lots50 < 0.01:
                    lots50 = pos.remaining  # trop petit pour spliter → close total
                pos.realised += pnl_for(pos.take_profit1 - self.slippage * sign, lots50)
                pos.remaining = round(pos.remaining - lots50, 2)
                pos.tp1_done = True
                pos.tp1_bar_time = bar_time
                # BE : SL à l'entrée (+ marge optionnelle BE_BUFFER_R×R, 0 par défaut),
                # vérifié sur bougies suivantes
                be_buffer = strategy.BE_BUFFER_R * abs(pos.entry - pos.stop_loss)
                pos.stop_loss = pos.entry - be_buffer * sign
                if pos.remaining < 0.01:
                    return {"closed": True, "reason": "tp1",
                            "exit_price": pos.take_profit1, "pnl": pos.realised}
                return {"closed": False, "reason": "tp1_partial",
                        "exit_price": pos.take_profit1, "pnl": pos.realised}

        # Stop loss — bar_low/bar_high pour capturer les touches intrabar.
        # Après TP1, ne pas re-checker sur la MÊME bougie qui vient de déclencher TP1 :
        # son high/low peut avoir été atteint AVANT le mouvement qui a touché TP1 (ex.
        # bougie qui ouvre au-dessus de l'entrée puis chute d'un trait), ce qui donnerait
        # un faux "SL après TP1" au BE sans qu'aucune bougie suivante n'y soit jamais
        # revenue (cf. backtest.py::_try_exit qui gère déjà ce cas via `return None`).
        skip_same_bar = pos.tp1_done and pos.tp1_bar_time is not None and bar_time == pos.tp1_bar_time
        hit_sl = (not skip_same_bar) and (
            bar_low <= pos.stop_loss if direction == "long" else bar_high >= pos.stop_loss)
        if hit_sl:
            pos.realised += pnl_for(pos.stop_loss - self.slippage * sign, pos.remaining)
            return {"closed": True,
                    "reason": "sl" if not pos.tp1_done else "sl_after_tp1",
                    "exit_price": pos.stop_loss, "pnl": pos.realised}

        # TP2 — sortie 50% restants
        if pos.tp1_done:
            hit_tp2 = bar_high >= pos.take_profit2 if direction == "long" else bar_low <= pos.take_profit2
            if hit_tp2:
                pos.realised += pnl_for(pos.take_profit2 - self.slippage * sign, pos.remaining)
                return {"closed": True, "reason": "tp2",
                        "exit_price": pos.take_profit2, "pnl": pos.realised}

        return None

    def close_position(self, pos: Position, reason: str = "manual",
                       price: Optional[float] = None) -> Dict[str, Any]:
        # `price`, quand fourni (ex: TP2/SL touché en temps réel par _price_tick), est le
        # niveau qui a déclenché la clôture. Sans lui, on retombait sur self.get_price()
        # — le dernier close M5 mis en cache, jusqu'à 300s (5min) plus vieux que le prix
        # temps réel qui a réellement déclenché le trigger. Un TP2 touché en cours de
        # bougie se retrouvait alors rempli à un prix bien moins favorable que la cible
        # réelle, sous-évaluant le P&L réalisé sans aucune trace de l'écart.
        if price is None:
            price = self.get_price()
        sign = 1.0 if pos.direction == "long" else -1.0
        fill = price - self.slippage * sign
        pnl = pos.realised + (fill - pos.entry) * sign * self.contract_size * pos.remaining
        pos.remaining = 0.0
        return {"closed": True, "reason": reason, "exit_price": fill, "pnl": pnl}


class ESPaperBroker(PaperBroker):
    """PaperBroker pour la stratégie ES : reproduit fidèlement la logique de
    sortie de pretrain_es.py::_try_exit_es (BE inconditionnel après TP1, pas
    de marge BE_BUFFER_R, pas d'early exit à 15 min — ce sont des mécaniques
    Strat A jamais validées pour ES) et arrondit en contrats entiers plutôt
    qu'en lots 0.01. Le timeout 45min/9 bougies reste géré au niveau de la
    boucle principale (main.py), avec une durée spécifique à ES."""
    name = "paper_es"

    def update_position(self, pos: Position) -> Optional[Dict[str, Any]]:
        df = self.data.get_m5(2)
        bar = df.iloc[-1]
        bar_time = bar.name
        # Même garde que le broker générique : jamais d'exit sur la bougie d'entrée
        # (voir commentaire dans PaperBroker.update_position).
        if pos.entry_bar_time is not None and bar_time == pos.entry_bar_time:
            return None
        bar_high = float(bar["high"])
        bar_low = float(bar["low"])
        direction = pos.direction
        sign = 1.0 if direction == "long" else -1.0

        def pnl_for(p, lots):
            return (p - pos.entry) * sign * self.contract_size * lots

        if direction == "long":
            pos.mfe = max(pos.mfe, bar_high - pos.entry)
        else:
            pos.mfe = max(pos.mfe, pos.entry - bar_low)

        # TP1 — sortie 50% (contrats entiers), BE inconditionnel ensuite
        if not pos.tp1_done:
            hit = bar_high >= pos.take_profit1 if direction == "long" else bar_low <= pos.take_profit1
            if hit:
                lots50 = round(pos.volume * 0.5)
                if lots50 < 1 or lots50 >= pos.remaining:
                    lots50 = pos.remaining  # pas assez de contrats pour spliter → close total
                pos.realised += pnl_for(pos.take_profit1 - self.slippage * sign, lots50)
                pos.remaining = round(pos.remaining - lots50)
                pos.tp1_done = True
                pos.tp1_bar_time = bar_time
                pos.stop_loss = pos.entry  # BE inconditionnel — pas de BE_BUFFER_R pour ES
                if pos.remaining < 1:
                    return {"closed": True, "reason": "tp1",
                            "exit_price": pos.take_profit1, "pnl": pos.realised}
                return {"closed": False, "reason": "tp1_partial",
                        "exit_price": pos.take_profit1, "pnl": pos.realised}

        # SL — même garde-fou "même bougie que TP1" que le broker générique
        # (voir Position.tp1_bar_time / commentaire dans PaperBroker.update_position)
        skip_same_bar = pos.tp1_done and pos.tp1_bar_time is not None and bar_time == pos.tp1_bar_time
        hit_sl = (not skip_same_bar) and (
            bar_low <= pos.stop_loss if direction == "long" else bar_high >= pos.stop_loss)
        if hit_sl:
            pos.realised += pnl_for(pos.stop_loss - self.slippage * sign, pos.remaining)
            return {"closed": True,
                    "reason": "sl" if not pos.tp1_done else "sl_after_tp1",
                    "exit_price": pos.stop_loss, "pnl": pos.realised}

        # TP2 — sortie du reliquat
        if pos.tp1_done:
            hit_tp2 = bar_high >= pos.take_profit2 if direction == "long" else bar_low <= pos.take_profit2
            if hit_tp2:
                pos.realised += pnl_for(pos.take_profit2 - self.slippage * sign, pos.remaining)
                return {"closed": True, "reason": "tp2",
                        "exit_price": pos.take_profit2, "pnl": pos.realised}

        return None


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

        # TP1 — sortie 50% (ou 100% si tp1_close_all) à 0.7R
        if not pos.tp1_done:
            hit = price >= pos.take_profit1 if pos.direction == "long" else price <= pos.take_profit1
            if hit:
                close_ratio = 1.0 if pos.meta.get("tp1_close_all") else 0.5
                lots50 = round(min(pos.volume * close_ratio, pos.remaining), 2)
                tick = self._mt5.symbol_info_tick(self.symbol)
                fill_price = tick.bid if pos.direction == "long" else tick.ask
                self._send_partial_close(pos, lots50, fill_price)
                pos.realised += (fill_price - pos.entry) * sign * CONTRACT_SIZE * lots50
                pos.remaining = round(pos.remaining - lots50, 2)
                pos.tp1_done = True
                # BE : SL à l'entrée (+ marge optionnelle BE_BUFFER_R×R, 0 par défaut)
                be_buffer = strategy.BE_BUFFER_R * abs(pos.entry - pos.stop_loss)
                pos.stop_loss = pos.entry - be_buffer * sign
                self._update_sl(pos, pos.stop_loss)
                if pos.remaining < 0.01:
                    return {"closed": True, "reason": "tp1",
                            "exit_price": fill_price, "pnl": pos.realised}
                return {"closed": False, "reason": "tp1_partial",
                        "exit_price": fill_price, "pnl": pos.realised}

        # TP2 — sortie 50% restants à 1.0R
        if pos.tp1_done:
            hit_tp2 = price >= pos.take_profit2 if pos.direction == "long" else price <= pos.take_profit2
            if hit_tp2:
                tick = self._mt5.symbol_info_tick(self.symbol)
                fill_price = tick.bid if pos.direction == "long" else tick.ask
                self._send_partial_close(pos, pos.remaining, fill_price)
                pos.realised += (fill_price - pos.entry) * sign * CONTRACT_SIZE * pos.remaining
                pos.remaining = 0.0
                return {"closed": True, "reason": "tp2",
                        "exit_price": fill_price, "pnl": pos.realised}

        return None

    def close_position(self, pos: Position, reason: str = "manual",
                       price: Optional[float] = None) -> Dict[str, Any]:
        # `price` ignoré ici — le tick MT5 ci-dessous est déjà temps réel, contrairement
        # au cache M5 de PaperBroker.get_price() (voir son close_position).
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
                contract_size: float = 100.0, pip_size: float = 0.1,
                data_symbol: Optional[str] = None, price_scale: float = 1.0,
                paper_broker_cls: type = PaperBroker) -> BaseBroker:
    """Factory: returns an MT5 broker for live mode if available, else paper.

    paper_broker_cls : classe PaperBroker à instancier (ex. ESPaperBroker pour
    une gestion de sortie fidèle à la stratégie ES, différente de la Strat A).
    """
    if mode == "live":
        mt5 = MT5Broker(symbol)
        if mt5.connected():
            return mt5
        # fall back to paper if MT5 unavailable
    return paper_broker_cls(spread_pips, slippage_pips, symbol=symbol,
                            contract_size=contract_size, pip_size=pip_size,
                            data_symbol=data_symbol, price_scale=price_scale)
