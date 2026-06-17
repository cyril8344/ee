"""
portfolio_index.py
==================
Portfolio index rebalancing engine.

Standalone module — does not touch strategy.py, broker.py, or main.py
unless explicitly wired by the caller.

Workflow
--------
1. Define a TargetAllocation  (symbol → weight, must sum to 1.0)
2. Snapshot the current portfolio  (symbol → current market value in $)
3. Engine calculates drift per symbol
4. If any drift exceeds the threshold, it generates RebalanceOrders
5. Caller passes orders to the execution layer (BrokerAdapter)

Execution adapters provided
---------------------------
- PaperAdapter   : logs orders, no real fills
- MT5Adapter     : sends orders via MetaTrader5 (Windows only)
- GenericAdapter : base class to implement a custom broker (Alpaca, IBKR…)
"""

from __future__ import annotations

import json
import logging
import os
import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Data structures
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class TargetAllocation:
    """
    Desired portfolio weights.

    Example:
        TargetAllocation({"XAUUSD": 0.60, "EURUSD": 0.40})
    """
    weights: Dict[str, float]

    def __post_init__(self):
        total = sum(self.weights.values())
        if abs(total - 1.0) > 0.001:
            raise ValueError(f"Weights must sum to 1.0, got {total:.4f}")
        if any(w < 0 for w in self.weights.values()):
            raise ValueError("Negative weights are not allowed")

    @classmethod
    def from_json(cls, path: str) -> "TargetAllocation":
        with open(path, "r") as f:
            data = json.load(f)
        return cls(weights=data["weights"])

    @classmethod
    def from_dict(cls, d: dict) -> "TargetAllocation":
        return cls(weights=d)

    def symbols(self) -> List[str]:
        return list(self.weights.keys())


@dataclass
class PortfolioSnapshot:
    """
    Current state of the portfolio: how much capital is in each symbol.

    values_usd : symbol → current market value in USD
    cash_usd   : uninvested cash
    """
    values_usd: Dict[str, float]
    cash_usd: float = 0.0
    captured_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    @property
    def total_usd(self) -> float:
        return sum(self.values_usd.values()) + self.cash_usd

    def current_weights(self) -> Dict[str, float]:
        total = self.total_usd
        if total <= 0:
            return {s: 0.0 for s in self.values_usd}
        return {s: v / total for s, v in self.values_usd.items()}


@dataclass
class DriftReport:
    """Drift of each symbol: current weight minus target weight."""
    symbol: str
    target_weight: float
    current_weight: float
    drift: float              # current - target  (positive = overweight)
    drift_usd: float          # dollar amount to rebalance
    action: str               # "BUY" | "SELL" | "HOLD"


@dataclass
class RebalanceOrder:
    """A single rebalance order ready for execution."""
    symbol: str
    action: str               # "BUY" | "SELL"
    usd_amount: float         # dollar value to trade
    lots: float               # calculated lot size
    price: float              # latest market price (indicative)
    contract_size: float      # ounces / units per lot
    reason: str               # human-readable (e.g. "drift +8.3%")
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


@dataclass
class RebalanceResult:
    """Outcome of an execute() call."""
    executed: List[RebalanceOrder]
    skipped: List[Tuple[RebalanceOrder, str]]   # (order, reason)
    total_usd_traded: float
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


# ─────────────────────────────────────────────────────────────────────────────
# Contract specs (mirrors risk_manager.py)
# ─────────────────────────────────────────────────────────────────────────────

CONTRACT_SPECS: Dict[str, Dict] = {
    "XAUUSD": {"contract_size": 100.0,     "min_lot": 0.01, "lot_step": 0.01},
    "EURUSD": {"contract_size": 100_000.0, "min_lot": 0.01, "lot_step": 0.01},
    "GBPUSD": {"contract_size": 100_000.0, "min_lot": 0.01, "lot_step": 0.01},
    "BTCUSD": {"contract_size": 1.0,       "min_lot": 0.01, "lot_step": 0.01},
}

def _get_spec(symbol: str) -> Dict:
    return CONTRACT_SPECS.get(symbol, {"contract_size": 100_000.0, "min_lot": 0.01, "lot_step": 0.01})


def _round_lots(raw: float, spec: Dict) -> float:
    step = spec["lot_step"]
    min_lot = spec["min_lot"]
    rounded = round(round(raw / step) * step, 2)
    return max(rounded, min_lot) if rounded >= min_lot else 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Broker adapters
# ─────────────────────────────────────────────────────────────────────────────

class BrokerAdapter(ABC):
    """Abstract base — implement this to connect any broker."""

    @abstractmethod
    def get_price(self, symbol: str) -> Optional[float]:
        """Return latest bid/ask mid price for symbol, or None on failure."""

    @abstractmethod
    def send_order(self, order: RebalanceOrder) -> bool:
        """Execute order. Return True on success."""

    def get_portfolio_snapshot(
        self, symbols: List[str], capital: float
    ) -> PortfolioSnapshot:
        """
        Default implementation: read open positions to build snapshot.
        Override in subclasses that can query live positions.
        Returns a snapshot with `capital` split equally as a fallback.
        """
        equal = capital / len(symbols) if symbols else 0.0
        return PortfolioSnapshot(
            values_usd={s: equal for s in symbols},
            cash_usd=0.0,
        )


class PaperAdapter(BrokerAdapter):
    """Simulated broker — logs orders, never touches real money."""

    def __init__(self, prices: Optional[Dict[str, float]] = None):
        self._prices = prices or {}

    def set_prices(self, prices: Dict[str, float]) -> None:
        self._prices = prices

    def get_price(self, symbol: str) -> Optional[float]:
        return self._prices.get(symbol)

    def send_order(self, order: RebalanceOrder) -> bool:
        logger.info(
            "[PaperAdapter] %s %s | %.4f lots @ %.5f | $%.2f | %s",
            order.action, order.symbol, order.lots, order.price,
            order.usd_amount, order.reason,
        )
        return True


class MT5Adapter(BrokerAdapter):
    """
    MetaTrader 5 execution adapter.
    Only functional on Windows with the MetaTrader5 package installed and
    a running terminal logged in to a live/demo account.
    """

    def __init__(self, magic: int = 20240101, deviation: int = 20):
        self.magic = magic
        self.deviation = deviation
        self._mt5 = None
        self._available = self._init_mt5()

    def _init_mt5(self) -> bool:
        try:
            import MetaTrader5 as mt5  # type: ignore
            if not mt5.initialize():
                logger.error("MT5Adapter: mt5.initialize() failed: %s", mt5.last_error())
                return False
            self._mt5 = mt5
            logger.info("MT5Adapter: connected to MetaTrader5")
            return True
        except ImportError:
            logger.warning("MT5Adapter: MetaTrader5 package not installed")
            return False

    def get_price(self, symbol: str) -> Optional[float]:
        if not self._available:
            return None
        tick = self._mt5.symbol_info_tick(symbol)
        if tick is None:
            return None
        return (tick.bid + tick.ask) / 2.0

    def send_order(self, order: RebalanceOrder) -> bool:
        if not self._available:
            logger.error("MT5Adapter: not available")
            return False
        mt5 = self._mt5
        tick = mt5.symbol_info_tick(order.symbol)
        if tick is None:
            logger.error("MT5Adapter: no tick for %s", order.symbol)
            return False

        order_type = mt5.ORDER_TYPE_BUY if order.action == "BUY" else mt5.ORDER_TYPE_SELL
        price = tick.ask if order.action == "BUY" else tick.bid

        request = {
            "action":   mt5.TRADE_ACTION_DEAL,
            "symbol":   order.symbol,
            "volume":   order.lots,
            "type":     order_type,
            "price":    price,
            "deviation": self.deviation,
            "magic":    self.magic,
            "comment":  f"rebalance:{order.reason[:20]}",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        result = mt5.order_send(request)
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            logger.error("MT5Adapter: order failed retcode=%s", result.retcode)
            return False
        logger.info("MT5Adapter: %s %s %.2f lots OK (order #%s)",
                    order.action, order.symbol, order.lots, result.order)
        return True

    def get_portfolio_snapshot(
        self, symbols: List[str], capital: float
    ) -> PortfolioSnapshot:
        if not self._available:
            return super().get_portfolio_snapshot(symbols, capital)
        values: Dict[str, float] = {s: 0.0 for s in symbols}
        positions = self._mt5.positions_get()
        if positions:
            for pos in positions:
                sym = pos.symbol
                if sym in values:
                    # current_price × volume × contract_size
                    spec = _get_spec(sym)
                    values[sym] += pos.volume * spec["contract_size"] * pos.price_current
        account = self._mt5.account_info()
        cash = account.balance - sum(values.values()) if account else 0.0
        return PortfolioSnapshot(values_usd=values, cash_usd=max(cash, 0.0))


class GenericHTTPAdapter(BrokerAdapter):
    """
    Skeleton for REST-based brokers (Alpaca, OANDA, IBKR…).
    Fill in `_base_url`, `_headers`, and the two abstract methods.
    """

    def __init__(self, base_url: str, api_key: str):
        self._base_url = base_url.rstrip("/")
        self._headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

    def get_price(self, symbol: str) -> Optional[float]:
        # TODO: implement GET {base_url}/v2/latest_quote/{symbol}
        raise NotImplementedError("Implement get_price for your broker")

    def send_order(self, order: RebalanceOrder) -> bool:
        # TODO: implement POST {base_url}/v2/orders
        raise NotImplementedError("Implement send_order for your broker")


# ─────────────────────────────────────────────────────────────────────────────
# Core engine
# ─────────────────────────────────────────────────────────────────────────────

class PortfolioIndexEngine:
    """
    Rebalancing engine.

    Usage
    -----
        target = TargetAllocation({"XAUUSD": 0.60, "EURUSD": 0.40})
        adapter = PaperAdapter(prices={"XAUUSD": 3380.0, "EURUSD": 1.0850})
        engine = PortfolioIndexEngine(target, adapter, total_capital=10000.0)

        result = engine.rebalance()          # auto: check drift → execute
        report = engine.drift_report()       # inspect drift without trading
    """

    def __init__(
        self,
        target: TargetAllocation,
        broker: BrokerAdapter,
        total_capital: float = 10_000.0,
        drift_threshold: float = 0.05,   # rebalance only if drift > 5 %
        min_order_usd: float = 50.0,     # ignore tiny orders
    ):
        self.target = target
        self.broker = broker
        self.total_capital = total_capital
        self.drift_threshold = drift_threshold
        self.min_order_usd = min_order_usd
        self._lock = threading.Lock()
        self._history: List[RebalanceResult] = []

    # ── public API ──────────────────────────────────────────────────────────

    def update_target(self, new_weights: Dict[str, float]) -> None:
        """Hot-swap allocation without reinstantiating the engine."""
        with self._lock:
            self.target = TargetAllocation(new_weights)
        logger.info("PortfolioIndex: target updated → %s", new_weights)

    def snapshot(self) -> PortfolioSnapshot:
        """Query current portfolio value from broker."""
        return self.broker.get_portfolio_snapshot(
            self.target.symbols(), self.total_capital
        )

    def drift_report(
        self, snap: Optional[PortfolioSnapshot] = None
    ) -> List[DriftReport]:
        """
        Return per-symbol drift without executing anything.
        Pass a snapshot to avoid a broker round-trip.
        """
        snap = snap or self.snapshot()
        current_w = snap.current_weights()
        reports = []
        for sym, tgt_w in self.target.weights.items():
            cur_w = current_w.get(sym, 0.0)
            drift = cur_w - tgt_w
            drift_usd = drift * snap.total_usd
            if abs(drift) < 1e-6:
                action = "HOLD"
            elif drift > 0:
                action = "SELL"
            else:
                action = "BUY"
            reports.append(DriftReport(
                symbol=sym,
                target_weight=tgt_w,
                current_weight=cur_w,
                drift=drift,
                drift_usd=drift_usd,
                action=action,
            ))
        return reports

    def generate_orders(
        self, snap: Optional[PortfolioSnapshot] = None
    ) -> List[RebalanceOrder]:
        """
        Compute rebalancing orders. Does NOT execute.
        Only generates orders for symbols whose drift exceeds the threshold.
        """
        snap = snap or self.snapshot()
        reports = self.drift_report(snap)
        orders = []

        for r in reports:
            if r.action == "HOLD":
                continue
            if abs(r.drift) < self.drift_threshold:
                continue                          # within tolerance → skip
            usd_amount = abs(r.drift_usd)
            if usd_amount < self.min_order_usd:
                continue                          # too small → skip

            price = self.broker.get_price(r.symbol)
            if not price:
                logger.warning("generate_orders: no price for %s — skipping", r.symbol)
                continue

            spec = _get_spec(r.symbol)
            # lots = usd_amount / (price × contract_size)
            raw_lots = usd_amount / (price * spec["contract_size"])
            lots = _round_lots(raw_lots, spec)
            if lots <= 0:
                continue

            orders.append(RebalanceOrder(
                symbol=r.symbol,
                action=r.action,
                usd_amount=usd_amount,
                lots=lots,
                price=price,
                contract_size=spec["contract_size"],
                reason=f"drift {r.drift:+.1%}",
            ))

        return orders

    def rebalance(
        self, dry_run: bool = False
    ) -> RebalanceResult:
        """
        Full rebalance cycle: snapshot → drift → orders → execute.

        dry_run=True : generate orders but skip execution (useful for preview).
        """
        with self._lock:
            snap = self.snapshot()
            orders = self.generate_orders(snap)

            executed, skipped = [], []

            if not orders:
                logger.info("PortfolioIndex: portfolio balanced — nothing to do")
            else:
                for order in orders:
                    if dry_run:
                        logger.info(
                            "[DRY RUN] would %s %s %.4f lots ($%.2f)",
                            order.action, order.symbol, order.lots, order.usd_amount,
                        )
                        executed.append(order)
                        continue
                    ok = self.broker.send_order(order)
                    if ok:
                        executed.append(order)
                    else:
                        skipped.append((order, "broker rejected"))

            result = RebalanceResult(
                executed=executed,
                skipped=skipped,
                total_usd_traded=sum(o.usd_amount for o in executed),
            )
            self._history.append(result)
            return result

    def status(self) -> dict:
        """Serialisable status dict for the API."""
        try:
            snap = self.snapshot()
            reports = self.drift_report(snap)
            needs_rebalance = any(
                abs(r.drift) >= self.drift_threshold for r in reports
            )
        except Exception as e:
            return {"error": str(e)}

        return {
            "target_weights": self.target.weights,
            "current_weights": {r.symbol: round(r.current_weight, 4) for r in reports},
            "drift": {r.symbol: round(r.drift, 4) for r in reports},
            "needs_rebalance": needs_rebalance,
            "drift_threshold": self.drift_threshold,
            "total_capital_usd": round(snap.total_usd, 2),
            "last_rebalance": (
                self._history[-1].timestamp if self._history else None
            ),
            "rebalances_total": len(self._history),
        }


# ─────────────────────────────────────────────────────────────────────────────
# Scheduled rebalancer (background thread)
# ─────────────────────────────────────────────────────────────────────────────

class ScheduledRebalancer:
    """
    Runs `engine.rebalance()` on a fixed schedule.

    interval_hours : how often to check (default: every 24h)
    dry_run        : if True, never sends real orders
    """

    def __init__(
        self,
        engine: PortfolioIndexEngine,
        interval_hours: float = 24.0,
        dry_run: bool = False,
    ):
        self.engine = engine
        self.interval_seconds = interval_hours * 3600
        self.dry_run = dry_run
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self.last_result: Optional[RebalanceResult] = None

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._loop, daemon=True, name="rebalancer"
        )
        self._thread.start()
        logger.info("ScheduledRebalancer: started (interval=%.1fh, dry_run=%s)",
                    self.interval_seconds / 3600, self.dry_run)

    def stop(self) -> None:
        self._stop_event.set()

    def _loop(self) -> None:
        while not self._stop_event.wait(self.interval_seconds):
            try:
                logger.info("ScheduledRebalancer: running rebalance cycle…")
                self.last_result = self.engine.rebalance(dry_run=self.dry_run)
                n = len(self.last_result.executed)
                usd = self.last_result.total_usd_traded
                logger.info(
                    "ScheduledRebalancer: %d order(s) executed, $%.2f traded", n, usd
                )
            except Exception:
                logger.exception("ScheduledRebalancer: error during rebalance")
