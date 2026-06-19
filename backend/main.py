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

import sys
import os
# Ensure backend/ is on sys.path regardless of working directory (Railway fix)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

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
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
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
from macro_filter import MacroFilter
from broker import make_broker, Position
import strategy
from strategy import add_indicators, evaluate, snapshot, swing_levels, active_session, find_order_blocks
from backtest import BacktestConfig, run_backtest
from optimizer import OptimizeConfig, run_optimize
from auth import create_access_token, get_current_user, verify_credentials
import cot_report
import retail_sentiment
import realtime_feed
import correlations as corr_engine
import finnhub_feed as _fh_module
from agent_manager import AgentManager
import agent_memory


MARKET_CONFIG = {
    "XAUUSD": {
        "name": "XAU/USD",
        "atr_min": 0.8,
        "contract_size": 100.0,
        "spread_pips": 0.3,
        "slippage_pips": 0.1,
        "pip_size": 0.1,       # 1 pip XAU/USD = $0.10
    },
    "EURUSD": {
        "name": "EUR/USD",
        "atr_min": 0.00030,
        "contract_size": 100000.0,
        "spread_pips": 0.2,
        "slippage_pips": 0.05,
        "pip_size": 0.0001,    # 1 pip EUR/USD = 0.0001
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

# Built React frontend (produced by `npm run build` during nixpacks build phase)
_FRONTEND_DIST = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "frontend", "dist")
_ASSETS_DIR = os.path.join(_FRONTEND_DIST, "assets")
if os.path.isdir(_ASSETS_DIR):
    app.mount("/assets", StaticFiles(directory=_ASSETS_DIR), name="frontend-assets")


class BotState:
    def __init__(self):
        db.init_db()
        self.settings = db.get_settings()
        self.risk = RiskManager()
        self.risk.sync_from_settings(self.settings)
        self.news = NewsFilter(window_minutes=30, currencies=("USD", "EUR"))
        self.macro = MacroFilter()
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
                cfg.get("pip_size", 0.1),
            )
            self.market_states[sym] = MarketState(symbol=sym, config=cfg, broker=broker)

        self.pattern_weights: Dict = db.get_pattern_stats()
        self._hydrate_today()
        self._restore_open_positions()

        # Agent IA — perpetual optimisation
        self.agent = AgentManager(self)
        self.agent.load_saved_config()
        self.agent.start()

    def _restore_open_positions(self):
        """On restart, rebuild in-memory Position objects from DB open trades."""
        from broker import Position
        open_trades = db.get_open_trades()
        for t in open_trades:
            sym = t.get("symbol", "XAUUSD")
            ms = self.market_states.get(sym)
            if ms is None or ms.position is not None:
                continue
            try:
                # Inject trade_id into meta so _finalize_trade can update the
                # correct DB row (trade_id is only held in memory normally).
                recovered_meta = {**t.get("meta", {}), "trade_id": t["id"]}
                ms.position = Position(
                    ticket=t.get("meta", {}).get("ticket", t["id"]),
                    direction=t["direction"],
                    entry=float(t["entry_price"]),
                    volume=float(t["volume"]),
                    stop_loss=float(t["stop_loss"]),
                    take_profit1=float(t["take_profit1"]),
                    take_profit2=float(t["take_profit2"]),
                    open_time=datetime.fromisoformat(t["entry_time"]),
                    meta=recovered_meta,
                    session=t.get("session", ""),
                )
                logger.info("[BotState] position restaurée: %s %s @ %s (trade_id=%s)",
                            sym, t["direction"], t["entry_price"], t["id"])
            except Exception:
                traceback.print_exc()

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
                eq += ms.position.unrealised_pnl(ms.broker.get_price(), ms.broker.contract_size)
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
                snap = snapshot(m5, m15, h1, atr_min_override=ms.config["atr_min"])
                ms.last_snapshot = snap

                # Vérifier la fiabilité des données AVANT toute gestion de position.
                # Sur données synthétiques, ni les entrées ni la gestion TP/SL ne doivent
                # s'exécuter — les prix simulés sont aléatoires et fermeraient les positions
                # à des niveaux irréels.
                data_synthetic = hasattr(ms.broker, "is_synthetic") and ms.broker.is_synthetic()
                if data_synthetic:
                    # N'alerter qu'une fois par transition (pas à chaque tick de 5s)
                    if not ms.last_snapshot.get("_was_synthetic"):
                        state.push_alert("warn", f"[{ms.symbol}] Données synthétiques — gestion suspendue (retry dans 15s)")
                    ms.last_snapshot["_was_synthetic"] = True
                    if ms.position is not None:
                        any_active = True
                    continue
                # Retour aux données réelles — log une fois
                if ms.last_snapshot.get("_was_synthetic"):
                    state.push_alert("info", f"[{ms.symbol}] Données réelles restaurées")
                    ms.last_snapshot["_was_synthetic"] = False

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
                macro_blocked, macro_reason = state.macro.blocks_entry(ms.symbol, snap.get("bias", "NEUTRE"))
                if macro_blocked:
                    state.push_alert("warn", f"[{ms.symbol}] Macro bloqué: {macro_reason}")
                if (ms.position is None and can_enter_session
                        and not state.risk.blocked and not news_status["blocked"]
                        and not macro_blocked
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
        session=sig.session, meta=sig.meta, risk_amount=decision.risk_amount,
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
    # Persist trade_id back into meta in DB so recovery after restart works
    db.update_trade(trade_id, {"meta": json.dumps({
        "ticket": pos.ticket, "reason": sig.reason,
        "trade_id": trade_id, **sig.meta,
    })})
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

    # Record trade context for agent learning
    if trade_id:
        try:
            agent_memory.record_trade_context(trade_id, {
                "adx":     ms.last_snapshot.get("adx"),
                "atr":     ms.last_snapshot.get("atr_m5"),
                "atr_avg": ms.last_snapshot.get("atr_avg"),
                "rsi_m5":  ms.last_snapshot.get("rsi_m5"),
                "rsi_m15": ms.last_snapshot.get("rsi_m15"),
                "session": ms.last_snapshot.get("session"),
                "bias":    ms.last_signal.get("direction") if isinstance(ms.last_signal, dict) else ms.last_signal,
                "won":     pnl > 0,
            })
        except Exception:
            pass

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
    upnl = pos.unrealised_pnl(price, ms.broker.contract_size)
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
        "progress_tp1": 1.0 if pos.tp1_done else round(max(-1.0, min(prog1, 1.5)), 3),
        "progress_tp2": round(max(-1.0, min(prog2, 1.5)), 3),
        "risk_amount": round(pos.risk_amount, 2),
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
            "conditions": snap.get("conditions"),
            "data_provider": getattr(getattr(ms.broker, "data", None), "provider", None),
            "data_errors": {k: v for k, v in __import__("data_provider").get_last_errors().items()
                            if k.startswith(sym + ":")},
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
        "macro": state.macro.status(),
        "alerts": state.alerts[-8:],
        "settings": {
            "session_filter": state.settings.get("session_filter", True),
            "active_markets": state.settings.get("active_markets", ["XAUUSD", "EURUSD"]),
        },
        "markets": markets,
        "realtime": {
            "connected": realtime_feed.is_connected(),
            "xauusd_tick": realtime_feed.get_latest("XAU/USD"),
            "eurusd_tick": realtime_feed.get_latest("EUR/USD"),
        },
    }


# --------------------------------------------------------------------------- #
# Background loop
# --------------------------------------------------------------------------- #
@app.on_event("startup")
async def _startup():
    try:
        realtime_feed.start_feed()
    except Exception:
        traceback.print_exc()
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

# ---- Authentication --------------------------------------------------------
class LoginRequest(BaseModel):
    username: str
    password: str


@app.get("/")
def root():
    _idx = os.path.join(_FRONTEND_DIST, "index.html")
    if os.path.isfile(_idx):
        return FileResponse(_idx)
    return {"status": "ok", "service": "scalping-bot"}


@app.get("/api/auth-debug")
def auth_debug():
    """Temporary diagnostic — confirms what credentials Railway passes to the app.
    Does NOT expose the password (only its length) so it is safe to open in a
    browser. Remove once login is confirmed working."""
    u = os.environ.get("ADMIN_USERNAME")
    p = os.environ.get("ADMIN_PASSWORD")
    return {
        "admin_username_is_set": u is not None,
        "admin_username_value": u,            # username is not a secret
        "admin_password_is_set": p is not None,
        "admin_password_length": len(p) if p else 0,
        "using_defaults": u is None and p is None,
    }


@app.post("/api/login")
def login(req: LoginRequest):
    """Public endpoint — no auth required. Returns a JWT on success."""
    if not verify_credentials(req.username, req.password):
        raise HTTPException(status_code=401, detail="Invalid username or password")
    token = create_access_token({"sub": req.username})
    return {"access_token": token, "token_type": "bearer"}


# ---- Protected endpoints ---------------------------------------------------
@app.get("/api/health")
def health(_user: dict = Depends(get_current_user)):
    brokers = {sym: {"name": ms.broker.name, "connected": ms.broker.connected()}
               for sym, ms in state.market_states.items()}
    return {"status": "ok", "brokers": brokers, "mode": state.settings.get("mode")}


@app.get("/api/state")
def get_state(_user: dict = Depends(get_current_user)):
    try:
        return trading_tick()
    except Exception as e:
        return {**_public_state(), "error": str(e)}


@app.get("/api/chart")
def get_chart(tf: str = "M5", symbol: str = "XAUUSD", _user: dict = Depends(get_current_user)):
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

    obs = find_order_blocks(add_indicators(base))
    order_blocks = [{"type": ob["type"], "low": round(ob["low"], 5), "high": round(ob["high"], 5)} for ob in obs]

    return {"timeframe": tf.upper(), "symbol": symbol, "candles": candles,
            "levels": levels, "markers": markers, "order_blocks": order_blocks}


@app.get("/api/trades")
def get_trades(scope: str = "today", _user: dict = Depends(get_current_user)):
    mode = state.settings.get("mode")
    if scope == "today":
        trades = db.get_trades_for_day(db.today_utc(), mode=mode)
    else:
        trades = db.get_recent_trades(200, mode=mode)
    curve = db.get_equity_curve(source="live", limit=500)
    return {"trades": trades, "equity_curve": curve}


@app.get("/api/settings")
def read_settings(_user: dict = Depends(get_current_user)):
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
def write_settings(patch: SettingsPatch, _user: dict = Depends(get_current_user)):
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


@app.post("/api/reset-day")
def reset_day(_user: dict = Depends(get_current_user)):
    """Remet les compteurs journaliers à zéro et débloque le bot sans effacer l'historique."""
    with state.lock:
        today = db.today_utc()
        db.update_daily(today, {"blocked": 0})
        state.risk.start_new_day(state.risk.capital)
        state.bot_status = "ACTIF"
    state.push_alert("info", "Journée réinitialisée — bot débloqué, compteurs remis à zéro")
    return {"ok": True, "message": "Journée réinitialisée"}


class ModeSwitch(BaseModel):
    mode: str                 # 'paper' | 'live'
    confirm: bool = False
    confirm_again: bool = False


@app.post("/api/mode")
def switch_mode(req: ModeSwitch, _user: dict = Depends(get_current_user)):
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
                cfg.get("pip_size", 0.1),
            )
        state.push_alert("info", f"Mode basculé sur {req.mode.upper()}")
    first_ms = next(iter(state.market_states.values()))
    return {"mode": req.mode, "broker": first_ms.broker.name,
            "connected": first_ms.broker.connected()}


@app.post("/api/close")
def close_now(symbol: str = "XAUUSD", _user: dict = Depends(get_current_user)):
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
def toggle_bot(_user: dict = Depends(get_current_user)):
    with state.lock:
        new_val = not state.settings.get("bot_enabled", True)
        state.settings = db.update_settings({"bot_enabled": new_val})
    return {"bot_enabled": new_val}


@app.post("/api/risk/unblock")
def unblock_risk(_user: dict = Depends(get_current_user)):
    """Force-clear the daily risk block (admin override)."""
    with state.lock:
        state.risk.blocked = False
        state.risk.block_reason = ""
        today = db.today_utc()
        db.update_daily(today, {"blocked": 0})
        state.push_alert("info", "🔓 Blocage risk réinitialisé manuellement")
    return {"ok": True, "blocked": False}


@app.post("/api/test/signal")
def test_signal(symbol: str = "XAUUSD", direction: str = "long",
                _user: dict = Depends(get_current_user)):
    """Force a test trade in PAPER mode only. Verifies the full execution pipeline."""
    with state.lock:
        if state.settings.get("mode", "paper") != "paper":
            raise HTTPException(status_code=400, detail="Test signal only available in paper mode")
        ms = state.market_states.get(symbol)
        if ms is None:
            raise HTTPException(status_code=404, detail=f"Unknown market: {symbol}")
        if ms.position is not None:
            raise HTTPException(status_code=409, detail="Position already open — close it first")
        if direction not in ("long", "short"):
            raise HTTPException(status_code=400, detail="direction must be 'long' or 'short'")

        try:
            m5, m15, h1 = build_context(ms.broker)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Could not load market data: {e}")

        cur = m5.iloc[-1]
        entry = float(cur["close"])
        atr_val = float(cur["atr"]) if not pd.isna(cur.get("atr", float("nan"))) else ms.config["atr_min"] * 2
        risk = atr_val * 1.2

        if direction == "long":
            sl = entry - risk
            tp1 = entry + risk
            tp2 = entry + 2.5 * risk
        else:
            sl = entry + risk
            tp1 = entry - risk
            tp2 = entry - 2.5 * risk

        from strategy import Signal
        now = datetime.now(timezone.utc)
        sig = Signal(
            direction=direction, bias="LONG" if direction == "long" else "SHORT",
            session="Test", entry=entry, stop_loss=sl,
            take_profit1=tp1, take_profit2=tp2,
            atr=atr_val, reason="test_signal",
            risk_distance=risk, timestamp=now,
            meta={"triggers": ["test_signal"], "rsi_m5": 50.0, "rsi_m15": 50.0},
        )
        decision = state.risk.can_open_trade(entry, sl, ms.config["contract_size"])
        if not decision.allowed:
            raise HTTPException(status_code=400, detail=f"Risk manager refused: {decision.reason}")

        _open_trade(ms, sig, decision, now)

    return {
        "status": "ok",
        "symbol": symbol,
        "direction": direction,
        "entry": round(entry, 3),
        "stop_loss": round(sl, 3),
        "take_profit1": round(tp1, 3),
        "message": f"Position TEST ouverte en paper @ {entry:.3f}",
    }


class BacktestRequest(BaseModel):
    start: str
    end: str
    capital: float = 1000.0
    risk_pct: float = 5.0
    spread_pips: float = 0.3
    slippage_pips: float = 0.1
    max_trades_per_day: int = 4
    daily_stop_pct: float = 100.0
    symbol: str = "XAUUSD"


@app.post("/api/backtest")
async def backtest(req: BacktestRequest, _user: dict = Depends(get_current_user)):
    cfg = BacktestConfig(
        start=req.start, end=req.end, capital=req.capital,
        risk_pct=req.risk_pct, spread_pips=req.spread_pips,
        slippage_pips=req.slippage_pips,
        max_trades_per_day=req.max_trades_per_day,
        daily_stop_pct=req.daily_stop_pct,
        symbol=req.symbol,
    )
    try:
        result = await asyncio.to_thread(run_backtest, cfg)
        return result
    except Exception as exc:
        traceback.print_exc()
        return {"error": str(exc) or "Erreur interne du backtest"}


class OptimizeRequest(BaseModel):
    start: str
    end: str
    symbol: str = "XAUUSD"
    capital: float = 1000.0
    risk_pct: float = 5.0
    spread_pips: float = 0.3
    slippage_pips: float = 0.1
    max_trades_per_day: int = 4
    daily_stop_pct: float = 100.0


@app.post("/api/optimize")
async def optimize(req: OptimizeRequest, _user: dict = Depends(get_current_user)):
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
def optimize_apply(req: ApplyParamsRequest, _user: dict = Depends(get_current_user)):
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


@app.get("/api/news-feed")
def news_feed(_user: dict = Depends(get_current_user)):
    """Return upcoming economic events and latest forex news from Finnhub.

    Response shape:
        {
            "upcoming_events": [{"time": "HH:MM", "event": "...",
                                  "currency": "USD", "impact": "high"}, ...],
            "latest_news":     [{"headline": "...", "datetime": <epoch>,
                                  "url": "..."}, ...10 items...]
        }
    Falls back to empty lists when Finnhub is not configured.
    """
    try:
        feed = _fh_module.get_feed()
        upcoming = feed.get_upcoming_events(hours_ahead=24)
        latest = feed.get_forex_news(limit=10)
    except Exception:
        upcoming = []
        latest = []
    return {"upcoming_events": upcoming, "latest_news": latest}


@app.get("/api/macro")
def macro_status():
    return state.macro.status()


@app.get("/api/pattern-stats")
def pattern_stats():
    return db.get_pattern_stats()


@app.post("/api/pattern-stats/reset")
def reset_pattern_stats(keep_symbol: Optional[str] = None,
                        _user: dict = Depends(get_current_user)):
    """
    Remet les poids de patterns à neutre.
    keep_symbol (ex: 'XAUUSD'): reconstruit les stats depuis l'historique
    de ce symbole, efface le reste. Utile pour purger les trades EUR/USD
    défectueux sans perdre l'apprentissage sur l'or.
    """
    replayed = db.reset_pattern_stats(rebuild_from_symbol=keep_symbol)
    state.pattern_weights = db.get_pattern_stats()
    if keep_symbol:
        state.push_alert("info",
            f"Stats patterns réinitialisées — {replayed} trades {keep_symbol} rejoués, "
            f"apprentissage EUR/USD effacé")
    else:
        state.push_alert("info", "Statistiques des patterns réinitialisées — poids remis à neutre")
    return {"ok": True, "replayed": replayed,
            "message": f"Pattern stats réinitialisées ({replayed} trades rejoués)"}


@app.get("/api/agent")
async def get_agent_status(user=Depends(get_current_user)):
    """Return current Agent IA status (running, last/next run, sharpe, params)."""
    return state.agent.status()


@app.get("/api/agent/history")
async def get_agent_history(user=Depends(get_current_user)):
    """Return the list of past Agent IA optimization runs (newest first)."""
    return state.agent.history()


# ── Portfolio index ─────────────────────────────────────────────────────────
from portfolio_index import (
    PortfolioIndexEngine, TargetAllocation, PaperAdapter, ScheduledRebalancer
)

_portfolio_engine: Optional[PortfolioIndexEngine] = None
_portfolio_rebalancer: Optional[ScheduledRebalancer] = None


def _get_portfolio_engine() -> PortfolioIndexEngine:
    global _portfolio_engine
    if _portfolio_engine is None:
        # Default equal-weight across active markets
        symbols = list(MARKET_CONFIG.keys())
        w = 1.0 / len(symbols)
        target = TargetAllocation({s: w for s in symbols})
        broker = PaperAdapter()
        _portfolio_engine = PortfolioIndexEngine(
            target, broker, total_capital=10_000.0
        )
    return _portfolio_engine


class PortfolioTargetRequest(BaseModel):
    weights: dict
    dry_run: bool = False


@app.get("/api/portfolio")
async def get_portfolio_status(user=Depends(get_current_user)):
    """Current portfolio drift vs target allocation."""
    return _get_portfolio_engine().status()


@app.post("/api/portfolio/target")
async def set_portfolio_target(req: PortfolioTargetRequest, user=Depends(get_current_user)):
    """Update target allocation and optionally trigger immediate rebalance."""
    engine = _get_portfolio_engine()
    engine.update_target(req.weights)
    if not req.dry_run:
        result = engine.rebalance(dry_run=False)
        return {
            "message": "Target updated and rebalanced",
            "executed": len(result.executed),
            "total_usd_traded": result.total_usd_traded,
        }
    result = engine.rebalance(dry_run=True)
    return {
        "message": "Target updated (dry run — no orders sent)",
        "would_execute": len(result.executed),
        "preview": [
            {"symbol": o.symbol, "action": o.action,
             "lots": o.lots, "usd_amount": round(o.usd_amount, 2)}
            for o in result.executed
        ],
    }


@app.post("/api/portfolio/rebalance")
async def trigger_rebalance(user=Depends(get_current_user)):
    """Manually trigger a rebalance cycle."""
    result = _get_portfolio_engine().rebalance(dry_run=False)
    return {
        "executed": len(result.executed),
        "skipped": len(result.skipped),
        "total_usd_traded": round(result.total_usd_traded, 2),
        "orders": [
            {"symbol": o.symbol, "action": o.action,
             "lots": o.lots, "usd": round(o.usd_amount, 2), "reason": o.reason}
            for o in result.executed
        ],
    }


@app.get("/api/data-provider")
def data_provider_status():
    """Which market-data providers are configured / currently usable."""
    import data_provider
    import os

    # Current provider per market (from cached broker state)
    current = {}
    for sym, ms in state.market_states.items():
        p = getattr(getattr(ms.broker, "data", None), "provider", None)
        current[sym] = p

    # Test a live fetch for each market to see what actually works
    test_results = {}
    for sym in state.market_states:
        try:
            _, prov = data_provider.get_m5(bars=10, symbol=sym)
            test_results[sym] = {"provider": prov, "ok": True}
        except Exception as e:
            test_results[sym] = {"provider": "error", "ok": False, "error": str(e)}

    return {
        "configured": os.environ.get("XAU_DATA_PROVIDER", "auto"),
        "available": data_provider.available_providers(),
        "current_per_market": current,
        "live_test": test_results,
        "twelvedata_key_set": bool(os.environ.get("TWELVEDATA_API_KEY", "").strip()),
        "last_errors": data_provider.get_last_errors(),
    }


@app.get("/api/cot")
def get_cot(_user: dict = Depends(get_current_user)):
    """
    Return the latest CFTC Commitments of Traders data for Gold (XAUUSD).
    Cached for 6 hours — data is published weekly on Fridays.
    """
    try:
        return cot_report.get_cot_data()
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"COT data unavailable: {exc}")


@app.get("/api/sentiment")
def get_sentiment(_user: dict = Depends(get_current_user)):
    """
    Return retail trader sentiment for XAUUSD and EURUSD.
    Primary source: Myfxbook community outlook (scraping).
    Fallback: COT-derived proxy for XAUUSD, static 50/50 for EURUSD.
    Cached for 15 minutes.
    """
    try:
        return retail_sentiment.get_sentiment()
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Sentiment data unavailable: {exc}")


@app.get("/api/fed")
def get_fed(_user: dict = Depends(get_current_user)):
    """Fed rate direction, real interest rates, and central bank gold positioning."""
    try:
        from fred_feed import get_gold_macro_bias
        return get_gold_macro_bias()
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"FRED data unavailable: {exc}")


@app.get("/api/correlations")
def get_correlations(_user: dict = Depends(get_current_user)):
    """
    Return 20-period rolling Pearson correlation of XAU/USD daily returns
    against key correlated/anti-correlated assets.
    Cached for 30 minutes — data updates slowly (daily close prices).
    """
    try:
        return corr_engine.get_correlations()
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Correlation data unavailable: {exc}")


# ── RL Agent endpoints ───────────────────────────────────────────────────────
from rl_trainer import RLTrainer as _RLTrainer

_rl_trainers: Dict[str, _RLTrainer] = {}


def _get_rl_trainer(symbol: str = "XAUUSD") -> _RLTrainer:
    if symbol not in _rl_trainers:
        trainer = _RLTrainer(symbol=symbol)
        trainer.start_auto()
        _rl_trainers[symbol] = trainer
    return _rl_trainers[symbol]


# Pre-warm RL trainers for active markets on startup
def _init_rl_trainers():
    for sym in state.market_states:
        _get_rl_trainer(sym)

threading.Thread(target=_init_rl_trainers, daemon=True, name="rl-init").start()


@app.get("/api/rl")
async def get_rl_status(symbol: str = "XAUUSD", user=Depends(get_current_user)):
    """RL agent status: training state, paper trading metrics, promotion flag."""
    return _get_rl_trainer(symbol).status()


@app.post("/api/rl/train")
async def start_rl_training(symbol: str = "XAUUSD", user=Depends(get_current_user)):
    """Launch RL training in the background (non-blocking)."""
    trainer = _get_rl_trainer(symbol)
    if trainer._training:
        raise HTTPException(status_code=409, detail="Training already in progress")
    trainer.train(blocking=False)
    return {"message": f"Training started for {symbol} ({trainer.backend()})"}


@app.post("/api/rl/paper")
async def start_rl_paper(symbol: str = "XAUUSD", user=Depends(get_current_user)):
    """Start live paper-trading loop for the RL agent."""
    trainer = _get_rl_trainer(symbol)
    if not trainer._agent:
        raise HTTPException(status_code=400, detail="No model — train first")
    trainer.run_paper_loop()
    return {"message": f"Paper loop started for {symbol}"}


@app.post("/api/rl/validate")
async def validate_rl(symbol: str = "XAUUSD", user=Depends(get_current_user)):
    """Evaluate current RL model on held-out data and return metrics."""
    trainer = _get_rl_trainer(symbol)
    metrics = await asyncio.to_thread(trainer.validate)
    if metrics is None:
        raise HTTPException(status_code=400, detail="No model or no validation data")
    return vars(metrics)


@app.get("/api/rl/history")
async def get_rl_history(symbol: str = "XAUUSD", user=Depends(get_current_user)):
    """Return the list of past RL training sessions (newest first)."""
    return _get_rl_trainer(symbol).history()


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


# SPA fallback — any unknown path serves index.html so React Router can handle it
@app.get("/{full_path:path}")
def spa_fallback(full_path: str):
    _idx = os.path.join(_FRONTEND_DIST, "index.html")
    if os.path.isfile(_idx):
        return FileResponse(_idx)
    raise HTTPException(status_code=404, detail="Not found")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)
