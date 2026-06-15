"""
main.py
=======
FastAPI application for the XAU/USD scalping bot.

Responsibilities
----------------
- Run a background trading loop (paper by default) that:
    * resamples M5 -> M15/H1, computes indicators,
    * checks session / news / risk gates,
    * evaluates the strategy, opens/manages positions per market,
    * persists trades + equity to SQLite,
    * pushes live state to the dashboard over WebSocket.
- Expose REST endpoints for state, chart data, trades, settings, mode
  switching (with double confirmation for live) and backtests.

Run:
    uvicorn main:app --reload --port 8000
"""

from __future__ import annotations

import asyncio
import json
import os
import threading
import traceback
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from typing import Optional, Dict, Any, List

import pandas as pd
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel


# --------------------------------------------------------------------------- #
# Telegram notifications (optional — set TELEGRAM_BOT_TOKEN + TELEGRAM_CHAT_ID)
# --------------------------------------------------------------------------- #
def _send_telegram(message: str) -> None:
    token = os.environ.get("TELEGRAM_BOT_TOKEN", "")
    chat_id = os.environ.get("TELEGRAM_CHAT_ID", "")
    if not token or not chat_id:
        return
    try:
        payload = json.dumps({"chat_id": chat_id, "text": message, "parse_mode": "HTML"})
        req = urllib.request.Request(
            f"https://api.telegram.org/bot{token}/sendMessage",
            data=payload.encode(),
            headers={"Content-Type": "application/json"},
        )
        urllib.request.urlopen(req, timeout=5)
    except Exception:
        pass

import database as db
from risk_manager import RiskManager
from news_filter import NewsFilter
from broker import make_broker, Position
import strategy
from strategy import add_indicators, evaluate, snapshot, swing_levels, active_session
from backtest import BacktestConfig, run_backtest
from optimizer import OptimizeConfig, run_optimize


MARKET_CONFIG = {
    "XAUUSD": {
        "name": "XAU/USD",
        "atr_min": 0.8,
        "contract_size": 100.0,
        "spread_pips": 0.3,
        "slippage_pips": 0.1,
    },
    "EURUSD": {
        "name": "EUR/USD",
        "atr_min": 0.00030,
        "contract_size": 100000.0,
        "spread_pips": 0.2,
        "slippage_pips": 0.05,
    },
}


@dataclass
class MarketState:
    symbol: str
    config: dict
    broker: Any
    position: Optional[Any] = None
    last_signal: Optional[Dict[str, Any]] = None
    last_snapshot: Dict[str, Any] = field(default_factory=dict)


# --------------------------------------------------------------------------- #
# App + global state
# --------------------------------------------------------------------------- #
app = FastAPI(title="XAU/USD Scalping Bot", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class BotState:
    def __init__(self):
        db.init_db()
        self.settings = db.get_settings()
        self.risk = RiskManager()
        self.risk.sync_from_settings(self.settings)
        self.news = NewsFilter(window_minutes=60, currencies=("USD", "EUR"))
        self.alerts: List[Dict[str, Any]] = []
        self.bot_status = "EN VEILLE"     # ACTIF | EN VEILLE | BLOQUE
        self.lock = threading.Lock()

        active_markets = self.settings.get("active_markets", ["XAUUSD", "EURUSD"])
        mode = self.settings.get("mode", "paper")
        self.market_states: Dict[str, MarketState] = {}
        for sym in active_markets:
            cfg = MARKET_CONFIG.get(sym, MARKET_CONFIG["XAUUSD"])
            broker = make_broker(
                mode, sym,
                self.settings.get("spread_pips", cfg["spread_pips"]),
                self.settings.get("slippage_pips", cfg["slippage_pips"]),
                cfg["contract_size"],
            )
            self.market_states[sym] = MarketState(symbol=sym, config=cfg, broker=broker)

        self.pattern_weights: Dict = db.get_pattern_stats()
        self._hydrate_today()

    def _hydrate_today(self):
        today = db.today_utc()
        daily = db.get_or_create_daily(today, self.risk.capital)
        trades = db.get_trades_for_day(today, mode=self.settings.get("mode"))
        closed = [t for t in trades if t["status"] == "closed"]
        pnl = sum(t.get("pnl") or 0.0 for t in closed)
        self.risk.hydrate_day(
            trades_today=len(trades),
            pnl_today=pnl,
            start_equity=daily["start_equity"],
            blocked=bool(daily["blocked"]),
        )

    def push_alert(self, kind: str, message: str):
        self.alerts.append({
            "kind": kind, "message": message,
            "ts": datetime.now(timezone.utc).isoformat(),
        })
        self.alerts = self.alerts[-30:]


state = BotState()


# --------------------------------------------------------------------------- #
# WebSocket manager
# --------------------------------------------------------------------------- #
class WSManager:
    def __init__(self):
        self.active: List[WebSocket] = []

    async def connect(self, ws: WebSocket):
        await ws.accept()
        self.active.append(ws)

    def disconnect(self, ws: WebSocket):
        if ws in self.active:
            self.active.remove(ws)

    async def broadcast(self, payload: Dict[str, Any]):
        dead = []
        msg = json.dumps(payload, default=str)
        for ws in self.active:
            try:
                await ws.send_text(msg)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.disconnect(ws)


ws_manager = WSManager()


# --------------------------------------------------------------------------- #
# Market context builder
# --------------------------------------------------------------------------- #
def build_context(broker):
    """Return (m5, m15, h1) indicator-ready frames from the given broker feed."""
    m5_raw = broker.get_rates_m5(500)
    m5 = add_indicators(m5_raw)
    agg = {"open": "first", "high": "max", "low": "min",
           "close": "last", "volume": "sum"}
    m15 = add_indicators(
        m5_raw.resample("15min", label="right", closed="right").agg(agg).dropna())
    h1 = add_indicators(
        m5_raw.resample("60min", label="right", closed="right").agg(agg).dropna())
    return m5, m15, h1


def current_equity() -> float:
    eq = state.risk.capital
    for ms in state.market_states.values():
        if ms.position is not None:
            try:
                eq += ms.position.unrealised_pnl(ms.broker.get_price())
            except Exception:
                pass
    return eq


# --------------------------------------------------------------------------- #
# Trading loop (one decision per tick)
# --------------------------------------------------------------------------- #
def trading_tick() -> Dict[str, Any]:
    with state.lock:
        # Daily rollover
        today = db.today_utc()
        daily = db.get_or_create_daily(today, state.risk.capital)
        if state.risk.start_equity_today is None:
            state.risk.start_new_day(daily["start_equity"])

        now = datetime.now(timezone.utc)
        session = active_session(now)
        news_status = state.news.status(now)
        session_filter = state.settings.get("session_filter", True)

        state.pattern_weights = db.get_pattern_stats()
        any_active = False
        for ms in state.market_states.values():
            try:
                m5, m15, h1 = build_context(ms.broker)
                snap = snapshot(m5, m15, h1)
                ms.last_snapshot = snap

                # ---- Manage open position ----
                if ms.position is not None:
                    pos = ms.position
                    close_info = ms.broker.update_position(pos)
                    age_min = (now - pos.open_time).total_seconds() / 60.0
                    if close_info is None and age_min >= strategy.MAX_TRADE_MINUTES:
                        close_info = ms.broker.close_position(pos, "timeout")
                    if close_info and close_info.get("closed"):
                        _finalize_trade(ms, pos, close_info, now)
                        ms.position = None
                    elif close_info and close_info.get("reason") == "tp1_partial":
                        state.push_alert("info", f"[{ms.symbol}] TP1 atteint — 60% clôturé")

                # ---- Look for entry ----
                can_enter_session = (session is not None) or (not session_filter)
                if (ms.position is None and can_enter_session
                        and not state.risk.blocked and not news_status["blocked"]
                        and state.settings.get("bot_enabled", True)):
                    sig = evaluate(m5, m15, h1, now=now, check_session=session_filter,
                                   atr_min=ms.config["atr_min"],
                                   pattern_weights=state.pattern_weights)
                    if sig is not None:
                        ms.last_signal = sig.to_dict()
                        decision = state.risk.can_open_trade(
                            sig.entry, sig.stop_loss,
                            contract_size=ms.config["contract_size"],
                        )
                        if decision.allowed:
                            _open_trade(ms, sig, decision, now)
                        else:
                            state.push_alert("warn", f"[{ms.symbol}] Signal ignoré: {decision.reason}")

                if ms.position is not None:
                    any_active = True
            except Exception:
                traceback.print_exc()

        # ---- Determine overall bot status ----
        if state.risk.blocked:
            state.bot_status = "BLOQUE"
        elif news_status["blocked"]:
            state.bot_status = "BLOQUE"
        elif session is None and session_filter:
            state.bot_status = "EN VEILLE"
        elif any_active:
            state.bot_status = "ACTIF"
        elif session is None and not session_filter:
            state.bot_status = "ACTIF"
        else:
            state.bot_status = "ACTIF"

        return _public_state(session, news_status)


def _open_trade(ms: MarketState, sig, decision, now):
    pos = ms.broker.market_order(
        sig.direction, decision.volume, sig.stop_loss,
        sig.take_profit1, sig.take_profit2,
        session=sig.session, meta=sig.meta,
    )
    ms.position = pos
    state.risk.register_open()

    trade_id = db.insert_trade({
        "symbol": ms.symbol,
        "direction": sig.direction,
        "session": sig.session,
        "entry_time": pos.open_time.isoformat(),
        "entry_price": pos.entry,
        "stop_loss": sig.stop_loss,
        "take_profit1": sig.take_profit1,
        "take_profit2": sig.take_profit2,
        "volume": decision.volume,
        "risk_amount": decision.risk_amount,
        "status": "open",
        "mode": state.settings.get("mode", "paper"),
        "meta": {"ticket": pos.ticket, "reason": sig.reason, **sig.meta},
    })
    pos.meta["trade_id"] = trade_id
    db.update_daily(db.today_utc(), {"trade_count": state.risk.trades_today})
    arrow = "🟢 LONG" if sig.direction == "long" else "🔴 SHORT"
    msg = (f"{arrow} <b>{ms.config['name']} ouvert</b>\n"
           f"Entrée : {pos.entry:.5f}\n"
           f"SL : {sig.stop_loss:.5f}  TP1 : {sig.take_profit1:.5f}  TP2 : {sig.take_profit2:.5f}\n"
           f"Raison : {sig.reason}\nSession : {sig.session}")
    state.push_alert("entry", f"[{ms.symbol}] {arrow} ouvert @ {pos.entry:.5f} ({sig.reason})")
    threading.Thread(target=_send_telegram, args=(msg,), daemon=True).start()


def _finalize_trade(ms: MarketState, pos: Position, close_info: Dict[str, Any], now: datetime):
    pnl = float(close_info["pnl"])
    state.risk.register_close(pnl)

    # Update pattern performance stats
    triggers = pos.meta.get("triggers", [])
    if triggers:
        won = pnl > 0
        db.update_pattern_stats(triggers, won)
        state.pattern_weights = db.get_pattern_stats()

    duration = (now - pos.open_time).total_seconds() / 60.0
    trade_id = pos.meta.get("trade_id")
    start_eq = state.risk.start_equity_today or state.risk.capital
    if trade_id:
        db.update_trade(trade_id, {
            "exit_time": now.isoformat(),
            "exit_price": close_info["exit_price"],
            "pnl": round(pnl, 2),
            "pnl_pct": round(pnl / start_eq * 100.0, 3),
            "duration_min": round(duration, 1),
            "status": "closed",
            "exit_reason": close_info["reason"],
        })
    today = db.today_utc()
    daily = db.get_daily(today) or {"pnl": 0.0}
    db.update_daily(today, {
        "pnl": round((daily.get("pnl") or 0.0) + pnl, 2),
        "blocked": 1 if state.risk.blocked else 0,
    })
    db.add_equity_point(state.risk.capital, source="live")

    result = "✅ GAGNANT" if pnl >= 0 else "❌ PERDANT"
    state.push_alert("exit", f"[{ms.symbol}] {result} {pnl:+.2f}$ ({close_info['reason']})")
    msg = (f"{result} <b>{ms.config['name']} clôturé</b>\n"
           f"PnL : {pnl:+.2f}$  Durée : {round(duration, 1)} min\n"
           f"Raison : {close_info['reason']}")
    threading.Thread(target=_send_telegram, args=(msg,), daemon=True).start()
    if state.risk.blocked:
        state.push_alert("danger", "🛑 Stop journalier atteint — bot bloqué jusqu'à demain")
        threading.Thread(target=_send_telegram,
                         args=("🛑 <b>Stop journalier atteint</b> — bot bloqué jusqu'à demain",),
                         daemon=True).start()


# --------------------------------------------------------------------------- #
# Public state serialisation
# --------------------------------------------------------------------------- #
def _position_payload(ms: MarketState) -> Optional[Dict[str, Any]]:
    pos = ms.position
    if pos is None:
        return None
    try:
        price = ms.broker.get_price()
    except Exception:
        price = pos.entry
    upnl = pos.unrealised_pnl(price)
    age = (datetime.now(timezone.utc) - pos.open_time).total_seconds()
    remaining_sec = max(0, strategy.MAX_TRADE_MINUTES * 60 - int(age))

    if pos.direction == "long":
        denom1 = (pos.take_profit1 - pos.entry) or 1e-9
        denom2 = (pos.take_profit2 - pos.entry) or 1e-9
        prog1 = (price - pos.entry) / denom1
        prog2 = (price - pos.entry) / denom2
    else:
        denom1 = (pos.entry - pos.take_profit1) or 1e-9
        denom2 = (pos.entry - pos.take_profit2) or 1e-9
        prog1 = (pos.entry - price) / denom1
        prog2 = (pos.entry - price) / denom2

    return {
        "ticket": pos.ticket,
        "direction": pos.direction,
        "entry": round(pos.entry, 5),
        "price": round(price, 5),
        "stop_loss": round(pos.stop_loss, 5),
        "take_profit1": round(pos.take_profit1, 5),
        "take_profit2": round(pos.take_profit2, 5),
        "volume": pos.volume,
        "remaining": pos.remaining,
        "tp1_done": pos.tp1_done,
        "session": pos.session,
        "unrealised_pnl": round(upnl, 2),
        "open_time": pos.open_time.isoformat(),
        "age_seconds": int(age),
        "remaining_seconds": remaining_sec,
        "progress_tp1": round(max(-1.0, min(prog1, 1.5)), 3),
        "progress_tp2": round(max(-1.0, min(prog2, 1.5)), 3),
    }


def _public_state(session=None, news_status=None) -> Dict[str, Any]:
    if news_status is None:
        news_status = state.news.status()
    today = db.today_utc()
    daily = db.get_daily(today) or {"pnl": 0.0, "start_equity": state.risk.capital}
    day_pnl = daily.get("pnl") or 0.0
    start_eq = daily.get("start_equity") or state.risk.capital

    markets = {}
    for sym, ms in state.market_states.items():
        snap = ms.last_snapshot
        markets[sym] = {
            "symbol": sym,
            "name": ms.config["name"],
            "bias": snap.get("bias", "NEUTRE"),
            "session": snap.get("session", "Hors session"),
            "price": snap.get("price"),
            "indicators": {
                "rsi_m5": snap.get("rsi_m5"),
                "rsi_m15": snap.get("rsi_m15"),
                "atr_m5": snap.get("atr_m5"),
                "atr_avg": snap.get("atr_avg"),
                "atr_min": ms.config["atr_min"],
            },
            "position": _position_payload(ms),
            "last_signal": ms.last_signal,
        }

    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "bot_status": state.bot_status,
        "mode": state.settings.get("mode", "paper"),
        "day_pnl": round(day_pnl, 2),
        "day_pnl_pct": round(day_pnl / start_eq * 100.0, 3) if start_eq else 0.0,
        "trades_today": state.risk.trades_today,
        "max_trades_per_day": state.risk.max_trades_per_day,
        "risk": state.risk.status(),
        "news": news_status,
        "alerts": state.alerts[-8:],
        "settings": {
            "session_filter": state.settings.get("session_filter", True),
            "active_markets": state.settings.get("active_markets", ["XAUUSD", "EURUSD"]),
        },
        "markets": markets,
    }


# --------------------------------------------------------------------------- #
# Background loop
# --------------------------------------------------------------------------- #
@app.on_event("startup")
async def _startup():
    asyncio.create_task(_loop())


async def _loop():
    while True:
        try:
            payload = await asyncio.to_thread(trading_tick)
            await ws_manager.broadcast({"type": "state", "data": payload})
        except Exception:
            traceback.print_exc()
        await asyncio.sleep(5)


# --------------------------------------------------------------------------- #
# REST API
# --------------------------------------------------------------------------- #
@app.get("/api/health")
def health():
    brokers = {sym: {"name": ms.broker.name, "connected": ms.broker.connected()}
               for sym, ms in state.market_states.items()}
    return {"status": "ok", "brokers": brokers, "mode": state.settings.get("mode")}


@app.get("/api/state")
def get_state():
    try:
        return trading_tick()
    except Exception as e:
        return {**_public_state(), "error": str(e)}


@app.get("/api/chart")
def get_chart(tf: str = "M5", symbol: str = "XAUUSD"):
    """Candles + EMAs + swing S/R for the dashboard chart."""
    ms = state.market_states.get(symbol)
    if ms is None:
        raise HTTPException(status_code=404, detail=f"Unknown market: {symbol}")
    m5_raw = ms.broker.get_rates_m5(500)
    rule = {"M5": None, "M15": "15min", "H1": "60min"}.get(tf.upper())
    if rule:
        agg = {"open": "first", "high": "max", "low": "min",
               "close": "last", "volume": "sum"}
        base = m5_raw.resample(rule, label="right", closed="right").agg(agg).dropna()
    else:
        base = m5_raw
    df = add_indicators(base).tail(180)
    levels = swing_levels(add_indicators(base), lookback=50)

    candles = []
    for ts, row in df.iterrows():
        candles.append({
            "time": ts.isoformat(),
            "open": round(float(row["open"]), 5),
            "high": round(float(row["high"]), 5),
            "low": round(float(row["low"]), 5),
            "close": round(float(row["close"]), 5),
            "ema9": round(float(row["ema9"]), 5),
            "ema21": round(float(row["ema21"]), 5),
            "ema200": round(float(row["ema200"]), 5),
            "rsi": round(float(row["rsi"]), 1),
            "volume": float(row["volume"]),
        })

    markers = []
    for t in db.get_trades_for_day(db.today_utc(), mode=state.settings.get("mode")):
        if t.get("symbol", "XAUUSD") != symbol:
            continue
        markers.append({
            "time": t["entry_time"], "type": "entry",
            "direction": t["direction"], "price": t["entry_price"],
        })
        if t.get("exit_time"):
            markers.append({
                "time": t["exit_time"], "type": "exit",
                "price": t.get("exit_price"), "pnl": t.get("pnl"),
            })

    return {"timeframe": tf.upper(), "symbol": symbol, "candles": candles,
            "levels": levels, "markers": markers}


@app.get("/api/trades")
def get_trades(scope: str = "today"):
    mode = state.settings.get("mode")
    if scope == "today":
        trades = db.get_trades_for_day(db.today_utc(), mode=mode)
    else:
        trades = db.get_recent_trades(200, mode=mode)
    curve = db.get_equity_curve(source="live", limit=500)
    return {"trades": trades, "equity_curve": curve}


@app.get("/api/settings")
def read_settings():
    return state.settings


class SettingsPatch(BaseModel):
    capital: Optional[float] = None
    risk_per_trade_pct: Optional[float] = None
    confirm_risk_change: Optional[bool] = False
    max_trades_per_day: Optional[int] = None
    daily_stop_pct: Optional[float] = None
    bot_enabled: Optional[bool] = None
    spread_pips: Optional[float] = None
    slippage_pips: Optional[float] = None
    session_filter: Optional[bool] = None
    active_markets: Optional[List[str]] = None


@app.post("/api/settings")
def write_settings(patch: SettingsPatch):
    data = patch.dict(exclude_none=True)
    if "risk_per_trade_pct" in data:
        if not data.pop("confirm_risk_change", False):
            raise HTTPException(
                status_code=400,
                detail="Changing risk per trade requires confirm_risk_change=true",
            )
    data.pop("confirm_risk_change", None)
    with state.lock:
        state.settings = db.update_settings(data)
        state.risk.sync_from_settings(state.settings)
    return state.settings


class ModeSwitch(BaseModel):
    mode: str                 # 'paper' | 'live'
    confirm: bool = False
    confirm_again: bool = False


@app.post("/api/mode")
def switch_mode(req: ModeSwitch):
    if req.mode not in ("paper", "live"):
        raise HTTPException(status_code=400, detail="mode must be paper|live")
    if req.mode == "live" and not (req.confirm and req.confirm_again):
        raise HTTPException(
            status_code=400,
            detail="Switching to LIVE requires double confirmation "
                   "(confirm=true & confirm_again=true)",
        )
    with state.lock:
        any_open = any(ms.position is not None for ms in state.market_states.values())
        if any_open:
            raise HTTPException(status_code=409,
                                detail="Close all open positions before switching mode")
        state.settings = db.update_settings({"mode": req.mode})
        for sym, ms in state.market_states.items():
            cfg = ms.config
            ms.broker = make_broker(
                req.mode, sym,
                state.settings.get("spread_pips", cfg["spread_pips"]),
                state.settings.get("slippage_pips", cfg["slippage_pips"]),
                cfg["contract_size"],
            )
        state.push_alert("info", f"Mode basculé sur {req.mode.upper()}")
    first_ms = next(iter(state.market_states.values()))
    return {"mode": req.mode, "broker": first_ms.broker.name,
            "connected": first_ms.broker.connected()}


@app.post("/api/close")
def close_now(symbol: str = "XAUUSD"):
    with state.lock:
        ms = state.market_states.get(symbol)
        if ms is None:
            raise HTTPException(status_code=404, detail=f"Unknown market: {symbol}")
        if ms.position is None:
            raise HTTPException(status_code=404, detail=f"No open position for {symbol}")
        now = datetime.now(timezone.utc)
        info = ms.broker.close_position(ms.position, "manual")
        _finalize_trade(ms, ms.position, info, now)
        ms.position = None
    return {"closed": True, "symbol": symbol}


@app.post("/api/bot/toggle")
def toggle_bot():
    with state.lock:
        new_val = not state.settings.get("bot_enabled", True)
        state.settings = db.update_settings({"bot_enabled": new_val})
    return {"bot_enabled": new_val}


class BacktestRequest(BaseModel):
    start: str
    end: str
    capital: float = 10000.0
    risk_pct: float = 1.0
    spread_pips: float = 0.3
    slippage_pips: float = 0.1
    max_trades_per_day: int = 4
    daily_stop_pct: float = 2.0
    symbol: str = "XAUUSD"


@app.post("/api/backtest")
async def backtest(req: BacktestRequest):
    cfg = BacktestConfig(
        start=req.start, end=req.end, capital=req.capital,
        risk_pct=req.risk_pct, spread_pips=req.spread_pips,
        slippage_pips=req.slippage_pips,
        max_trades_per_day=req.max_trades_per_day,
        daily_stop_pct=req.daily_stop_pct,
        symbol=req.symbol,
    )
    result = await asyncio.to_thread(run_backtest, cfg)
    return result


class OptimizeRequest(BaseModel):
    start: str
    end: str
    symbol: str = "XAUUSD"
    capital: float = 10000.0
    risk_pct: float = 1.0
    spread_pips: float = 0.3
    slippage_pips: float = 0.1
    max_trades_per_day: int = 4
    daily_stop_pct: float = 2.0


@app.post("/api/optimize")
async def optimize(req: OptimizeRequest):
    cfg = OptimizeConfig(
        start=req.start, end=req.end, symbol=req.symbol,
        capital=req.capital, risk_pct=req.risk_pct,
        spread_pips=req.spread_pips, slippage_pips=req.slippage_pips,
        max_trades_per_day=req.max_trades_per_day,
        daily_stop_pct=req.daily_stop_pct,
    )
    result = await asyncio.to_thread(run_optimize, cfg)
    return result


class ApplyParamsRequest(BaseModel):
    adx_min: Optional[float] = None
    rsi_low: Optional[float] = None
    rsi_high: Optional[float] = None
    sl_atr_mult: Optional[float] = None
    sr_proximity: Optional[float] = None


@app.post("/api/optimize/apply")
def optimize_apply(req: ApplyParamsRequest):
    """Apply optimised strategy parameters in-memory (no DB persistence)."""
    applied: Dict[str, Any] = {}
    if req.adx_min is not None:
        strategy.ADX_MIN = req.adx_min
        applied["adx_min"] = req.adx_min
    if req.rsi_low is not None:
        strategy.RSI_LOW = req.rsi_low
        applied["rsi_low"] = req.rsi_low
    if req.rsi_high is not None:
        strategy.RSI_HIGH = req.rsi_high
        applied["rsi_high"] = req.rsi_high
    if req.sl_atr_mult is not None:
        strategy.SL_ATR_MULT = req.sl_atr_mult
        applied["sl_atr_mult"] = req.sl_atr_mult
    if req.sr_proximity is not None:
        strategy.SR_PROXIMITY_ATR = req.sr_proximity
        applied["sr_proximity"] = req.sr_proximity
    return {"applied": applied, "message": "Paramètres appliqués en mémoire (non persistés)"}


@app.get("/api/news")
def news():
    state.news.refresh()
    return state.news.status()


@app.get("/api/pattern-stats")
def pattern_stats():
    return db.get_pattern_stats()


@app.get("/api/data-provider")
def data_provider_status():
    """Which market-data providers are configured / currently usable."""
    import data_provider
    import os
    return {
        "configured": os.environ.get("XAU_DATA_PROVIDER", "auto"),
        "available": data_provider.available_providers(),
    }


# --------------------------------------------------------------------------- #
# WebSocket endpoint
# --------------------------------------------------------------------------- #
@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws_manager.connect(ws)
    try:
        try:
            await ws.send_text(json.dumps(
                {"type": "state", "data": await asyncio.to_thread(trading_tick)},
                default=str))
        except Exception:
            pass
        while True:
            await ws.receive_text()
    except WebSocketDisconnect:
        ws_manager.disconnect(ws)
    except Exception:
        ws_manager.disconnect(ws)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)
