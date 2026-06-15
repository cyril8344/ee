"""
main.py
=======
FastAPI application for the XAU/USD scalping bot.

Responsibilities
----------------
- Run a background trading loop (paper by default) that:
    * resamples M5 -> M15/H1, computes indicators,
    * checks session / news / risk gates,
    * evaluates the strategy, opens/manages a single position,
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
import threading
import traceback
from datetime import datetime, timezone, timedelta
from typing import Optional, Dict, Any, List

import pandas as pd
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import database as db
from risk_manager import RiskManager
from news_filter import NewsFilter
from broker import make_broker, Position
import strategy
from strategy import add_indicators, evaluate, snapshot, swing_levels, active_session
from backtest import BacktestConfig, run_backtest


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
        self.news = NewsFilter(window_minutes=30, currencies=("USD",))
        self.broker = make_broker(
            self.settings.get("mode", "paper"),
            self.settings.get("symbol", "XAUUSD"),
            self.settings.get("spread_pips", 0.3),
            self.settings.get("slippage_pips", 0.1),
        )
        self.position: Optional[Position] = None
        self.last_signal: Optional[Dict[str, Any]] = None
        self.last_snapshot: Dict[str, Any] = {}
        self.alerts: List[Dict[str, Any]] = []
        self.bot_status = "EN VEILLE"     # ACTIF | EN VEILLE | BLOQUE
        self.lock = threading.Lock()
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
def build_context():
    """Return (m5, m15, h1) indicator-ready frames from the broker feed."""
    m5_raw = state.broker.get_rates_m5(500)
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
    if state.position is not None:
        try:
            eq += state.position.unrealised_pnl(state.broker.get_price())
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

        m5, m15, h1 = build_context()
        snap = snapshot(m5, m15, h1)
        state.last_snapshot = snap

        now = datetime.now(timezone.utc)
        session = active_session(now)
        news_status = state.news.status(now)

        # ---- Manage open position ----
        if state.position is not None:
            pos = state.position
            close_info = state.broker.update_position(pos)
            # forced timeout (45 min)
            age_min = (now - pos.open_time).total_seconds() / 60.0
            if close_info is None and age_min >= strategy.MAX_TRADE_MINUTES:
                close_info = state.broker.close_position(pos, "timeout")
            if close_info and close_info.get("closed"):
                _finalize_trade(pos, close_info, now)
                state.position = None
            elif close_info and close_info.get("reason") == "tp1_partial":
                state.push_alert("info", "TP1 atteint — 60% clôturé")

        # ---- Determine bot status ----
        if state.risk.blocked:
            state.bot_status = "BLOQUE"
        elif news_status["blocked"]:
            state.bot_status = "BLOQUE"
        elif session is None:
            state.bot_status = "EN VEILLE"
        else:
            state.bot_status = "ACTIF"

        # ---- Look for entry ----
        if (state.position is None and session is not None
                and not state.risk.blocked and not news_status["blocked"]
                and state.settings.get("bot_enabled", True)):
            sig = evaluate(m5, m15, h1, now=now, check_session=True)
            if sig is not None:
                state.last_signal = sig.to_dict()
                decision = state.risk.can_open_trade(sig.entry, sig.stop_loss)
                if decision.allowed:
                    _open_trade(sig, decision, now)
                else:
                    state.push_alert("warn", f"Signal ignoré: {decision.reason}")

        return _public_state(snap, session, news_status)


def _open_trade(sig, decision, now):
    pos = state.broker.market_order(
        sig.direction, decision.volume, sig.stop_loss,
        sig.take_profit1, sig.take_profit2,
        session=sig.session, meta=sig.meta,
    )
    state.position = pos
    state.risk.register_open()

    trade_id = db.insert_trade({
        "symbol": state.settings.get("symbol", "XAUUSD"),
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
    state.push_alert("entry", f"{arrow} ouvert @ {pos.entry:.2f} ({sig.reason})")


def _finalize_trade(pos: Position, close_info: Dict[str, Any], now: datetime):
    pnl = float(close_info["pnl"])
    state.risk.register_close(pnl)
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
    # update daily + equity
    today = db.today_utc()
    daily = db.get_daily(today) or {"pnl": 0.0}
    db.update_daily(today, {
        "pnl": round((daily.get("pnl") or 0.0) + pnl, 2),
        "blocked": 1 if state.risk.blocked else 0,
    })
    db.add_equity_point(state.risk.capital, source="live")

    result = "✅ GAGNANT" if pnl >= 0 else "❌ PERDANT"
    state.push_alert("exit", f"{result} {pnl:+.2f}$ ({close_info['reason']})")
    if state.risk.blocked:
        state.push_alert("danger", "🛑 Stop journalier atteint — bot bloqué jusqu'à demain")


# --------------------------------------------------------------------------- #
# Public state serialisation
# --------------------------------------------------------------------------- #
def _position_payload() -> Optional[Dict[str, Any]]:
    pos = state.position
    if pos is None:
        return None
    try:
        price = state.broker.get_price()
    except Exception:
        price = pos.entry
    upnl = pos.unrealised_pnl(price)
    age = (datetime.now(timezone.utc) - pos.open_time).total_seconds()
    remaining_sec = max(0, strategy.MAX_TRADE_MINUTES * 60 - int(age))

    # progress toward TP1/TP2
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
        "entry": round(pos.entry, 3),
        "price": round(price, 3),
        "stop_loss": round(pos.stop_loss, 3),
        "take_profit1": round(pos.take_profit1, 3),
        "take_profit2": round(pos.take_profit2, 3),
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


def _public_state(snap=None, session=None, news_status=None) -> Dict[str, Any]:
    if snap is None:
        snap = state.last_snapshot
    if news_status is None:
        news_status = state.news.status()
    today = db.today_utc()
    daily = db.get_daily(today) or {"pnl": 0.0, "start_equity": state.risk.capital}
    day_pnl = daily.get("pnl") or 0.0
    start_eq = daily.get("start_equity") or state.risk.capital

    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "bias": snap.get("bias", "NEUTRE"),
        "session": snap.get("session", "Hors session"),
        "bot_status": state.bot_status,
        "mode": state.settings.get("mode", "paper"),
        "price": snap.get("price"),
        "day_pnl": round(day_pnl, 2),
        "day_pnl_pct": round(day_pnl / start_eq * 100.0, 3) if start_eq else 0.0,
        "trades_today": state.risk.trades_today,
        "max_trades_per_day": state.risk.max_trades_per_day,
        "risk": state.risk.status(),
        "indicators": {
            "rsi_m5": snap.get("rsi_m5"),
            "rsi_m15": snap.get("rsi_m15"),
            "atr_m5": snap.get("atr_m5"),
            "atr_avg": snap.get("atr_avg"),
            "atr_min": snap.get("atr_min"),
        },
        "news": news_status,
        "position": _position_payload(),
        "last_signal": state.last_signal,
        "alerts": state.alerts[-8:],
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
    return {"status": "ok", "broker": state.broker.name,
            "connected": state.broker.connected(),
            "mode": state.settings.get("mode")}


@app.get("/api/state")
def get_state():
    try:
        return trading_tick()
    except Exception as e:
        # Return last known state rather than 500ing the dashboard.
        return {**_public_state(), "error": str(e)}


@app.get("/api/chart")
def get_chart(tf: str = "M5"):
    """Candles + EMAs + swing S/R for the dashboard chart."""
    m5_raw = state.broker.get_rates_m5(500)
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
            "open": round(float(row["open"]), 3),
            "high": round(float(row["high"]), 3),
            "low": round(float(row["low"]), 3),
            "close": round(float(row["close"]), 3),
            "ema9": round(float(row["ema9"]), 3),
            "ema21": round(float(row["ema21"]), 3),
            "ema200": round(float(row["ema200"]), 3),
            "rsi": round(float(row["rsi"]), 1),
            "volume": float(row["volume"]),
        })

    # entry/exit markers from today's trades
    markers = []
    for t in db.get_trades_for_day(db.today_utc(), mode=state.settings.get("mode")):
        markers.append({
            "time": t["entry_time"], "type": "entry",
            "direction": t["direction"], "price": t["entry_price"],
        })
        if t.get("exit_time"):
            markers.append({
                "time": t["exit_time"], "type": "exit",
                "price": t.get("exit_price"), "pnl": t.get("pnl"),
            })

    return {"timeframe": tf.upper(), "candles": candles,
            "levels": levels, "markers": markers}


@app.get("/api/trades")
def get_trades(scope: str = "today"):
    mode = state.settings.get("mode")
    if scope == "today":
        trades = db.get_trades_for_day(db.today_utc(), mode=mode)
    else:
        trades = db.get_recent_trades(200, mode=mode)
    # intraday equity curve
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


@app.post("/api/settings")
def write_settings(patch: SettingsPatch):
    data = patch.dict(exclude_none=True)
    # Risk % change requires confirmation
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
        if state.position is not None:
            raise HTTPException(status_code=409,
                                detail="Close the open position before switching mode")
        state.settings = db.update_settings({"mode": req.mode})
        state.broker = make_broker(
            req.mode, state.settings.get("symbol", "XAUUSD"),
            state.settings.get("spread_pips", 0.3),
            state.settings.get("slippage_pips", 0.1),
        )
        state.push_alert("info", f"Mode basculé sur {req.mode.upper()}")
    return {"mode": req.mode, "broker": state.broker.name,
            "connected": state.broker.connected()}


@app.post("/api/close")
def close_now():
    with state.lock:
        if state.position is None:
            raise HTTPException(status_code=404, detail="No open position")
        now = datetime.now(timezone.utc)
        info = state.broker.close_position(state.position, "manual")
        _finalize_trade(state.position, info, now)
        state.position = None
    return {"closed": True}


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


@app.post("/api/backtest")
async def backtest(req: BacktestRequest):
    cfg = BacktestConfig(
        start=req.start, end=req.end, capital=req.capital,
        risk_pct=req.risk_pct, spread_pips=req.spread_pips,
        slippage_pips=req.slippage_pips,
        max_trades_per_day=req.max_trades_per_day,
        daily_stop_pct=req.daily_stop_pct,
    )
    result = await asyncio.to_thread(run_backtest, cfg)
    # persist backtest equity curve under its own source
    if "equity_curve" in result:
        # don't spam the DB; only summary is interesting to keep
        pass
    return result


@app.get("/api/news")
def news():
    state.news.refresh()
    return state.news.status()


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
        # send an immediate snapshot
        try:
            await ws.send_text(json.dumps(
                {"type": "state", "data": await asyncio.to_thread(trading_tick)},
                default=str))
        except Exception:
            pass
        while True:
            # keep the socket alive; ignore client messages
            await ws.receive_text()
    except WebSocketDisconnect:
        ws_manager.disconnect(ws)
    except Exception:
        ws_manager.disconnect(ws)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)
