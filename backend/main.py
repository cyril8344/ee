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
import logging
import os
import threading
import traceback
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime, date, timezone, timedelta
from typing import Optional, Dict, Any, List

logger = logging.getLogger("main")

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
from risk_manager import RiskManager, RiskDecision
from news_filter import NewsFilter
from macro_filter import MacroFilter
from broker import make_broker, Position
import strategy
from strategy import add_indicators, evaluate, snapshot, swing_levels, active_session, find_order_blocks
import feature_logger as _feat_log
from backtest import BacktestConfig, run_backtest
from optimizer import OptimizeConfig, run_optimize
from auth import create_access_token, get_current_user, verify_credentials
import cot_report
import retail_sentiment
import realtime_feed
import correlations as corr_engine
import finnhub_feed as _fh_module
from agent_manager import AgentManager
from live_agent import LiveAdaptiveAgent
import agent_memory
from ml_gate import OnlineLogisticRegression, AdaptiveThresholds
import pretrain as _pretrain_module
from llm_gate import LLMGate
from researcher_agent import ResearcherAgent
from adaptive_agent import AdaptiveAgent


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
        "default_strategy": "eurusd_simple",  # EMA pullback + OB + patterns
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
    adaptive: Optional[Any] = None   # AdaptiveThresholds instance
    ml_gate: Optional[Any] = None    # OnlineLogisticRegression instance (per symbol)
    circuit_breaker_until: Optional[datetime] = None
    recent_results: List[bool] = field(default_factory=list)
    last_close_time: Optional[datetime] = None  # horodatage de la dernière fermeture


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
        # Sanitiser les valeurs corrompues (ex: daily_stop_pct à des millions % après amorçage)
        _sanitize = {}
        if float(self.settings.get("daily_stop_pct", 2.0)) > 10.0:
            _sanitize["daily_stop_pct"] = 2.0
        if int(self.settings.get("max_trades_per_day", 4)) > 10:
            _sanitize["max_trades_per_day"] = 4
        if _sanitize:
            self.settings = db.update_settings(_sanitize)
            logger.warning("[BotState] settings corrompus sanitisés: %s", _sanitize)

        # En mode amorçage : forcer bot_enabled=True (sans corrompre daily_stop_pct)
        if strategy.BOOTSTRAP_MODE:
            if not self.settings.get("bot_enabled", True):
                self.settings = db.update_settings({"bot_enabled": True})
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
            ms = MarketState(symbol=sym, config=cfg, broker=broker)
            ms.adaptive = AdaptiveThresholds(
                atr_min_default=cfg["atr_min"], symbol=sym
            )
            ms.ml_gate = OnlineLogisticRegression(symbol=sym)
            self.market_states[sym] = ms

        self.pattern_weights: Dict = db.get_pattern_stats()
        self._hydrate_today()
        self._restore_open_positions()

        # Agent IA — perpetual optimisation (backtest-based, EURUSD only)
        self.agent = AgentManager(self)
        self.agent.load_saved_config()
        self.agent.start()

        # Agent live adaptatif — apprend uniquement des vrais trades paper XAUUSD
        self.live_agent = LiveAdaptiveAgent(symbol="XAUUSD")

        # LLM gate — validation contextuelle des signaux (désactivé si ANTHROPIC_API_KEY absent)
        self.llm_gate = LLMGate()

        # Chercheur de paramètres — optimise RSI/ADX en arrière-plan hors session
        self.researcher = ResearcherAgent(capital=self.risk.capital)

        # Agent adaptatif autonome — analyse les trades et ajuste strategy.* à chaud
        self.adaptive = AdaptiveAgent()

        # En BOOTSTRAP_MODE : débloquer le risk au démarrage (stop journalier non pertinent)
        if strategy.BOOTSTRAP_MODE and self.risk.blocked:
            self.risk.blocked = False
            self.risk.block_reason = ""
            db.update_daily(db.today_utc(), {"blocked": 0})

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
    """Return (m5, m15, h1, h4) indicator-ready frames from the given broker feed."""
    m5_raw = broker.get_rates_m5(500)
    m5 = add_indicators(m5_raw)
    agg = {"open": "first", "high": "max", "low": "min",
           "close": "last", "volume": "sum"}
    m15 = add_indicators(
        m5_raw.resample("15min", label="right", closed="right").agg(agg).dropna())
    h1 = add_indicators(
        m5_raw.resample("60min", label="right", closed="right").agg(agg).dropna())
    h4 = add_indicators(
        m5_raw.resample("240min", label="right", closed="right").agg(agg).dropna())
    return m5, m15, h1, h4


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
# Boucle d'apprentissage autonome — helpers
# --------------------------------------------------------------------------- #
def _launch_auto_pretrain(label: str = "auto", window_months: int = 3, reset: bool = True) -> None:
    """Lance un pretrain en arrière-plan sur les `window_months` derniers mois."""
    if _pretrain_module.get_progress()["running"]:
        logger.info("[AutoPretrain:%s] pretrain déjà en cours — ignoré", label)
        return
    end_d   = date.today().isoformat()
    start_d = (date.today() - timedelta(days=window_months * 30)).isoformat()
    capital = state.risk.capital

    def _done():
        state.push_alert("info", f"[AutoApprenant] Pretrain '{label}' terminé — ML Gate rafraîchie")

    _pretrain_module.launch_pretrain(
        start_d, end_d, symbol="XAUUSD", reset=reset,
        capital=capital, on_complete=_done,
    )
    logger.info("[AutoPretrain:%s] lancé %s → %s", label, start_d, end_d)


def _build_llm_snap(ms, sig, m5, m15, h1) -> Dict[str, Any]:
    """Construit le snapshot de marché pour la LLM gate."""
    cur     = m5.iloc[-1]  if len(m5)  > 0 else {}
    m15_cur = m15.iloc[-1] if len(m15) > 0 else {}
    h1_cur  = h1.iloc[-1]  if len(h1)  > 0 else {}
    close     = float(getattr(cur,     "close",  0) or 0)
    ema200    = float(getattr(h1_cur,  "ema200", 0) or 0)
    atr_h1    = float(getattr(h1_cur,  "atr",    1) or 1)
    ema200_dist = (close - ema200) / max(atr_h1, 0.001) if ema200 else 0.0
    vwap      = float(getattr(cur, "vwap", close) or close)
    return {
        "session":       sig.session,
        "rsi_m5":        float(sig.meta.get("rsi_m5",  50)),
        "rsi_m15":       float(sig.meta.get("rsi_m15", 50)),
        "atr":           sig.atr,
        "adx_h1":        float(getattr(h1_cur, "adx", 0) or 0),
        "bias":          sig.bias,
        "vwap_side":     1 if close >= vwap else 0,
        "patterns":      list(sig.meta.get("triggers", [])),
        "pattern_weight": float(sig.meta.get("weight_sum", 0.0)),
        "ml_score":      float(sig.meta.get("ml_prob") or 0.5),
        "ema200_dist":   round(ema200_dist, 3),
    }


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
                m5, m15, h1, h4 = build_context(ms.broker)
                snap = snapshot(m5, m15, h1, atr_min_override=ms.config["atr_min"],
                               pattern_weights=state.pattern_weights,
                               ml_gate=ms.ml_gate,
                               adaptive_thresholds=ms.adaptive)
                ms.last_snapshot = snap

                # Pour les marchés Strategy B, calculer les conditions ICT en temps réel
                if ms.config.get("default_strategy", "A") == "B":
                    try:
                        from strategy_ict import (_h1_bias, _find_order_blocks,
                                                   _in_ob, ADX_MIN_H1, _h1_sr_levels,
                                                   SR_ZONE_ATR_H1)
                        _bias_ict  = _h1_bias(h1)
                        _atr_ict   = float(m5.iloc[-1].get("atr", 0) or 0)
                        _dir_ict   = _bias_ict or "LONG"
                        _obs_long  = _find_order_blocks(m5, "LONG",  _atr_ict)
                        _obs_short = _find_order_blocks(m5, "SHORT", _atr_ict)
                        _obs_dir   = _obs_long if _bias_ict == "LONG" else _obs_short
                        _cur_ict   = m5.iloc[-1]
                        _in_ob_now = any(_in_ob(float(_cur_ict["low"]), float(_cur_ict["high"]), ob) for ob in _obs_dir)
                        _adx_h1    = float(h1.iloc[-1].get("adx", 0) or 0) if len(h1) > 0 else 0.0
                        _adx_ok    = _adx_h1 >= ADX_MIN_H1
                        _h1_sr     = _h1_sr_levels(h1)
                        _price_ict = float(_cur_ict["close"])
                        _h1_atr    = float(h1.iloc[-1].get("atr", _atr_ict) or _atr_ict) if len(h1) > 0 else _atr_ict
                        _zone_tol  = SR_ZONE_ATR_H1 * _h1_atr
                        _near_res  = any(0 < (r - _price_ict) < _zone_tol for r in _h1_sr["resistance"])
                        _near_sup  = any(0 < (_price_ict - s) < _zone_tol for s in _h1_sr["support"])
                        _sr_active = _near_res or _near_sup

                        _blocking = None
                        if _bias_ict is None:
                            _blocking = "bias_neutre"
                        elif not _adx_ok and not _sr_active:
                            _blocking = "adx_h1_trop_bas"
                        elif not _obs_dir and not _sr_active:
                            _blocking = "aucun_ob_detecte"
                        elif not _in_ob_now and not _sr_active:
                            _blocking = "prix_hors_ob"

                        snap["ict_conditions"] = {
                            "h1_bias":        _bias_ict or "NEUTRE",
                            "adx_h1":         round(_adx_h1, 1),
                            "adx_ok":         _adx_ok,
                            "ob_count_long":  len(_obs_long),
                            "ob_count_short": len(_obs_short),
                            "in_ob_zone":     _in_ob_now,
                            "sr_active":      _sr_active,
                            "sr_zone":        "resistance" if _near_res else ("support" if _near_sup else None),
                            "blocking_reason": _blocking,
                            "obs": [{"type": "bullish" if _dir_ict == "LONG" else "bearish",
                                     "low": round(ob["low"], 5), "high": round(ob["high"], 5)}
                                    for ob in _obs_dir[:5]],
                        }
                    except Exception:
                        pass


                def _set_loop_gate(reason: str):
                    """Surcharge blocking_reason dans conditions pour debug dashboard."""
                    c = ms.last_snapshot.get("conditions")
                    if isinstance(c, dict) and not c.get("blocking_reason"):
                        c["blocking_reason"] = reason

                # BOOTSTRAP_MODE : lever les blocages capital/daily-stop uniquement
                # — max_trades_per_day est respecté pour éviter les doublons
                if strategy.BOOTSTRAP_MODE:
                    if state.risk.blocked:
                        state.risk.blocked = False
                        state.risk.block_reason = ""
                        db.update_daily(db.today_utc(), {"blocked": 0})
                    if ms.circuit_breaker_until is not None:
                        ms.circuit_breaker_until = None
                        ms.recent_results.clear()

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
                    _set_loop_gate("données_synthétiques")
                    if ms.position is not None:
                        any_active = True
                    continue
                # Retour aux données réelles — log une fois
                if ms.last_snapshot.get("_was_synthetic"):
                    state.push_alert("info", f"[{ms.symbol}] Données réelles restaurées")
                    ms.last_snapshot["_was_synthetic"] = False

                # ---- Manage open position ----
                _just_closed = False
                if ms.position is not None:
                    pos = ms.position
                    close_info = ms.broker.update_position(pos)
                    age_min = (now - pos.open_time).total_seconds() / 60.0
                    if close_info is None and age_min >= strategy.MAX_TRADE_MINUTES:
                        close_info = ms.broker.close_position(pos, "timeout")
                    if close_info and close_info.get("closed"):
                        _finalize_trade(ms, pos, close_info, now)
                        ms.position = None
                        ms.last_close_time = now
                        _just_closed = True  # pas de ré-entrée dans la même itération
                    elif close_info and close_info.get("reason") == "tp1_partial":
                        state.push_alert("info", f"[{ms.symbol}] TP1 atteint — 60% clôturé")

                # ---- Look for entry ----
                can_enter_session = (session is not None) or (not session_filter)
                macro_blocked, macro_reason = state.macro.blocks_entry(ms.symbol, snap.get("bias", "NEUTRE"))
                if macro_blocked:
                    state.push_alert("warn", f"[{ms.symbol}] Macro bloqué: {macro_reason}")
                # Circuit breaker — lever la pause si le délai est écoulé
                if ms.circuit_breaker_until is not None and now >= ms.circuit_breaker_until:
                    ms.circuit_breaker_until = None
                    ms.recent_results.clear()
                    state.push_alert("info", f"[{ms.symbol}] Circuit breaker levé — reprise du trading")

                # Calculer la raison de blocage boucle externe (indépendant de evaluate())
                if ms.position is None:
                    if not can_enter_session:
                        _set_loop_gate("hors_session")
                    elif state.risk.blocked:
                        _set_loop_gate(f"risk: {state.risk.block_reason}")
                    elif news_status["blocked"]:
                        _set_loop_gate("actualités")
                    elif macro_blocked:
                        _set_loop_gate(f"macro: {macro_reason}")
                    elif ms.circuit_breaker_until is not None:
                        _set_loop_gate("circuit_breaker")
                    elif not state.settings.get("bot_enabled", True):
                        _set_loop_gate("bot_désactivé")
                    elif state.risk.trades_today >= state.risk.max_trades_per_day:
                        _set_loop_gate(f"max_trades: {state.risk.trades_today}/{state.risk.max_trades_per_day}")

                # Cooldown 300s (1 bougie M5) après fermeture — évite la ré-entrée sur le même signal
                _CLOSE_COOLDOWN_SECS = 300
                _in_cooldown = (
                    ms.last_close_time is not None
                    and (now - ms.last_close_time).total_seconds() < _CLOSE_COOLDOWN_SECS
                )
                _bs = strategy.BOOTSTRAP_MODE
                _daily_limit_ok = state.risk.trades_today < state.risk.max_trades_per_day
                if (ms.position is None and not _just_closed and not _in_cooldown
                        and _daily_limit_ok and can_enter_session
                        and (_bs or (
                            not state.risk.blocked
                            and not news_status["blocked"]
                            and not macro_blocked
                            and ms.circuit_breaker_until is None
                            and state.settings.get("bot_enabled", True)
                        ))):
                    # Strategy is fixed per symbol in MARKET_CONFIG (not user-switchable)
                    sym_strategy = ms.config.get("default_strategy", "A")
                    if sym_strategy == "B":
                        from strategy_ict import evaluate_ict as _eval_ict
                        sig = _eval_ict(m5, m15, h1, now=now,
                                        check_session=session_filter,
                                        atr_min=ms.config["atr_min"])
                    elif sym_strategy == "eurusd_simple":
                        from strategy import evaluate_eurusd as _eval_eurusd
                        sig = _eval_eurusd(m5, m15, h1, now=now,
                                           check_session=session_filter,
                                           atr_min=ms.config["atr_min"],
                                           pattern_weights=state.pattern_weights,
                                           ml_gate=ms.ml_gate)
                    else:
                        _rlog: Dict[str, Any] = {}
                        sig = evaluate(m5, m15, h1, h4=h4, now=now, check_session=session_filter,
                                       atr_min=ms.config["atr_min"],
                                       pattern_weights=state.pattern_weights,
                                       ml_gate=ms.ml_gate,
                                       adaptive_thresholds=ms.adaptive,
                                       _reject_log=_rlog)
                        if sig is None and _rlog:
                            ms.last_snapshot["reject_log"] = _rlog
                            _set_loop_gate("evaluate: " + list(_rlog.keys())[0])
                            logger.info("[%s] evaluate() rejet: %s", ms.symbol, _rlog)
                        # LLM gate — validation contextuelle (hors BOOTSTRAP_MODE)
                        if sig is not None and state.llm_gate.enabled and not strategy.BOOTSTRAP_MODE:
                            _snap_llm = _build_llm_snap(ms, sig, m5, m15, h1)
                            _llm = state.llm_gate.analyze(_snap_llm, sig.direction)
                            if _llm["action"] == "HOLD" or _llm["confidence"] < 0.55:
                                sig = None
                                _set_loop_gate(f"llm: {_llm['reason'][:40]}")
                    if sig is not None:
                        ms.last_signal = sig.to_dict()
                        decision = state.risk.can_open_trade(
                            sig.entry, sig.stop_loss,
                            contract_size=ms.config["contract_size"],
                        )
                        # BOOTSTRAP_MODE : si bloqué uniquement par taille de lot / capital,
                        # forcer lot minimum 0.01 pour collecter les données ML.
                        # NE PAS contourner max_trades_per_day ni daily_stop (évite les 10× même trade).
                        if not decision.allowed and strategy.BOOTSTRAP_MODE:
                            sl_dist = abs(sig.entry - sig.stop_loss)
                            if (sl_dist > 0
                                    and state.risk.trades_today < state.risk.max_trades_per_day
                                    and not state.risk.blocked):
                                risk_amt = 0.01 * sl_dist * ms.config["contract_size"]
                                decision = RiskDecision(True, "bootstrap_min_lot",
                                                        volume=0.01, risk_amount=risk_amt,
                                                        stop_distance=sl_dist)
                        if decision.allowed:
                            _open_trade(ms, sig, decision, now)
                        else:
                            _set_loop_gate(f"risk_trade: {decision.reason}")
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

        # Chercheur de paramètres — appel périodique hors session sans position active
        state.researcher.maybe_run(has_active_position=any_active)
        # Agent adaptatif — analyse et ajuste les paramètres toutes les 6h
        state.adaptive.maybe_run(has_active_position=any_active)

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

    # Logging ML — stratégie B uniquement
    if sig.meta.get("strategy") == "ICT_B":
        sess_start_h = {"London": 7, "NY": 13}.get(sig.session, 7)
        session_hour = (ts.hour + ts.minute / 60.0) - sess_start_h
        _feat_log.log_entry(trade_id, sig.meta,
                            ts_utc=pos.open_time.isoformat(),
                            session=sig.session,
                            session_hour=round(session_hour, 2))
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
    won = pnl > 0

    # Agent live adaptatif XAUUSD — apprend de chaque trade réel
    if ms.symbol == "XAUUSD":
        try:
            state.live_agent.on_trade_closed(won=won, pnl=pnl,
                                             features=pos.meta.get("ml_features"))
            # Transition BOOTSTRAP → filtres calibrés : lancer pretrain automatique
            if state.live_agent.consume_bootstrap_exit():
                from live_agent import BOOTSTRAP_EXIT_TRADES
                state.push_alert("info",
                    f"[AutoApprenant] {BOOTSTRAP_EXIT_TRADES} trades live — BOOTSTRAP désactivé, pretrain lancé")
                _launch_auto_pretrain(label="post_bootstrap", window_months=3)
        except Exception:
            pass

    # Circuit breaker — WR glissant sur les 15 derniers trades
    ms.recent_results.append(won)
    if len(ms.recent_results) > 15:
        ms.recent_results = ms.recent_results[-15:]
    if len(ms.recent_results) >= 10:
        rolling_wr = sum(ms.recent_results) / len(ms.recent_results)
        if rolling_wr < 0.38 and ms.circuit_breaker_until is None:
            ms.circuit_breaker_until = now + timedelta(hours=2)
            state.push_alert("warn", f"[{ms.symbol}] Circuit breaker: WR {rolling_wr:.0%} sur {len(ms.recent_results)} trades → pause 2h")

    if triggers:
        db.update_pattern_stats(triggers, won)
        state.pattern_weights = db.get_pattern_stats()

    # Update ML gate and adaptive thresholds with entry context features
    ml_features = pos.meta.get("ml_features")
    if ml_features:
        try:
            ms.ml_gate.update(ml_features, won)
        except Exception:
            pass
        try:
            ms.adaptive.update(ml_features, pos.entry, won)
        except Exception:
            pass

    duration = (now - pos.open_time).total_seconds() / 60.0
    trade_id = pos.meta.get("trade_id")

    # Logging ML — stratégie B uniquement
    if trade_id and pos.meta.get("strategy") == "ICT_B":
        risk_amount = pos.meta.get("risk_amount") or abs(pos.entry - pos.stop_loss) * pos.volume * 100.0
        _feat_log.log_exit(trade_id, pnl=pnl, risk_amount=risk_amount,
                           exit_reason=close_info.get("reason", ""),
                           duration_min=duration)
    start_eq = state.risk.start_equity_today or state.risk.capital
    _exit_patch = {
        "exit_time": now.isoformat(),
        "exit_price": close_info["exit_price"],
        "pnl": round(pnl, 2),
        "pnl_pct": round(pnl / start_eq * 100.0, 3),
        "duration_min": round(duration, 1),
        "status": "closed",
        "exit_reason": close_info["reason"],
    }
    if trade_id:
        _updated = db.update_trade(trade_id, _exit_patch)
        if not _updated:
            # La ligne a été supprimée par un Reset pendant que le trade était ouvert.
            # On ré-insère une ligne complète pour que l'historique reste cohérent.
            db.insert_trade({
                "symbol": ms.symbol,
                "direction": pos.direction,
                "session": pos.meta.get("session", ""),
                "entry_time": pos.open_time.isoformat(),
                "exit_time": now.isoformat(),
                "entry_price": pos.entry,
                "exit_price": close_info["exit_price"],
                "stop_loss": pos.stop_loss,
                "take_profit1": pos.take_profit1,
                "take_profit2": pos.take_profit2,
                "volume": pos.volume,
                "risk_amount": pos.meta.get("risk_amount"),
                "pnl": round(pnl, 2),
                "pnl_pct": round(pnl / start_eq * 100.0, 3),
                "duration_min": round(duration, 1),
                "status": "closed",
                "exit_reason": close_info["reason"],
                "mode": state.settings.get("mode", "paper"),
                "meta": pos.meta,
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

    _sym_to_td = {"XAUUSD": "XAU/USD", "EURUSD": "EUR/USD"}
    markets = {}
    for sym, ms in state.market_states.items():
        snap = ms.last_snapshot
        # Préférer le prix du feed WebSocket temps réel au close OHLCV (stale jusqu'à 5 min)
        rt_tick = realtime_feed.get_latest(_sym_to_td.get(sym, sym))
        live_price = rt_tick["price"] if rt_tick else snap.get("price")
        markets[sym] = {
            "symbol": sym,
            "name": ms.config["name"],
            "bias": snap.get("bias", "NEUTRE"),
            "session": snap.get("session", "Hors session"),
            "price": live_price,
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
            "ict_conditions": snap.get("ict_conditions"),
            "reject_log": snap.get("reject_log"),
            "ml_gate": ms.ml_gate.status() if ms.ml_gate else {},
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
        "live_agent": state.live_agent.status(),
        "llm_gate": state.llm_gate.status(),
        "researcher": state.researcher.status(),
        "adaptive": state.adaptive.status(),
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
    # Restaurer les heures bloquées depuis les settings persistés
    saved_hours = state.settings.get("bad_hours_cet")
    if saved_hours is not None:
        strategy.BAD_HOURS_CET = set(saved_hours)
    try:
        realtime_feed.start_feed()
    except Exception:
        traceback.print_exc()
    asyncio.create_task(_loop())
    asyncio.create_task(_price_tick())
    asyncio.create_task(_learning_loop())


async def _price_tick():
    """Boucle légère toutes les secondes : met à jour le prix temps réel depuis le feed
    WebSocket Twelve Data (non-bloquant) et vérifie TP/SL sur les positions ouvertes."""
    _sym_to_td = {"XAUUSD": "XAU/USD", "EURUSD": "EUR/USD"}
    while True:
        await asyncio.sleep(1)
        try:
            with state.lock:
                for sym, ms in state.market_states.items():
                    # Lire le prix depuis le feed WebSocket — opération non-bloquante
                    tick = realtime_feed.get_latest(_sym_to_td.get(sym, sym))
                    if tick is None:
                        continue
                    rt_price = tick["price"]
                    # Mettre à jour le prix pour tous les symboles (avec ou sans position)
                    ms.last_snapshot["price"] = rt_price

                    pos = ms.position
                    if pos is None:
                        continue
                    # Vérifier TP/SL avec le prix temps réel
                    direction = pos.direction
                    if direction == "long":
                        if rt_price <= pos.stop_loss:
                            close_info = ms.broker.close_position(pos, "sl_realtime")
                            if close_info and close_info.get("closed"):
                                now = datetime.now(timezone.utc)
                                _finalize_trade(ms, pos, close_info, now)
                                ms.position = None
                                ms.last_close_time = now
                        elif rt_price >= pos.take_profit2:
                            close_info = ms.broker.close_position(pos, "tp2_realtime")
                            if close_info and close_info.get("closed"):
                                now = datetime.now(timezone.utc)
                                _finalize_trade(ms, pos, close_info, now)
                                ms.position = None
                                ms.last_close_time = now
                    else:
                        if rt_price >= pos.stop_loss:
                            close_info = ms.broker.close_position(pos, "sl_realtime")
                            if close_info and close_info.get("closed"):
                                now = datetime.now(timezone.utc)
                                _finalize_trade(ms, pos, close_info, now)
                                ms.position = None
                                ms.last_close_time = now
                        elif rt_price <= pos.take_profit2:
                            close_info = ms.broker.close_position(pos, "tp2_realtime")
                            if close_info and close_info.get("closed"):
                                now = datetime.now(timezone.utc)
                                _finalize_trade(ms, pos, close_info, now)
                                ms.position = None
                                ms.last_close_time = now
        except Exception:
            traceback.print_exc()


async def _loop():
    while True:
        try:
            payload = await asyncio.to_thread(trading_tick)
            await ws_manager.broadcast({"type": "state", "data": payload})
        except Exception:
            traceback.print_exc()
        await asyncio.sleep(2)


async def _learning_loop():
    """Pretrain hebdomadaire automatique chaque dimanche 22h CET — garde la ML Gate fraîche."""
    import pytz
    _CET = pytz.timezone("Europe/Paris")
    while True:
        try:
            now_cet = datetime.now(timezone.utc).astimezone(_CET)
            # Dimanche = weekday 6. Calculer les secondes jusqu'au prochain dimanche 22h00 CET.
            days_to_sunday = (6 - now_cet.weekday()) % 7
            target = now_cet.replace(hour=22, minute=0, second=0, microsecond=0)
            if days_to_sunday == 0 and now_cet >= target:
                days_to_sunday = 7   # ce dimanche est déjà passé → prochain
            target += timedelta(days=days_to_sunday)
            sleep_secs = (target - now_cet).total_seconds()
            logger.info("[LearningLoop] prochain pretrain hebdo dans %.1fh (dimanche 22h CET)", sleep_secs / 3600)
            await asyncio.sleep(max(sleep_secs, 3600))   # au moins 1h entre les vérifications
        except asyncio.CancelledError:
            return
        except Exception:
            traceback.print_exc()
            await asyncio.sleep(3600)
            continue

        with state.lock:
            state.push_alert("info", "[AutoApprenant] Pretrain hebdomadaire — 3 mois glissants, ML Gate rafraîchie")
            _launch_auto_pretrain(label="hebdomadaire", window_months=3, reset=False)


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
    # Fetch more raw M5 bars so EMA200 has enough warmup before the displayed window.
    # M5: 2500 bars → EMA200 needs 200, leaves 2300 warmup before the last 180 shown.
    # M15: 3600 M5 bars → ~1200 M15 bars → ~1020 warmup bars before the last 180 shown.
    # H1: 5000 M5 bars → ~416 H1 bars → ~236 warmup bars before the last 180 shown.
    _RAW_BARS = {"M5": 2500, "M15": 3600, "H1": 5000}
    m5_raw = ms.broker.get_rates_m5(_RAW_BARS.get(tf.upper(), 2500))
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

    # Si le dernier bar OHLCV est périmé (>6 min), ajouter une bougie live
    # pour que le graphique affiche toujours l'heure courante même si le
    # provider de données a du retard (ex : yfinance GC=F, rate-limit TD).
    if candles and tf.upper() == "M5":
        last_ts = pd.Timestamp(candles[-1]["time"])
        if last_ts.tzinfo is None:
            last_ts = last_ts.tz_localize("UTC")
        now_utc = datetime.now(timezone.utc)
        staleness_secs = (now_utc - last_ts.to_pydatetime()).total_seconds()
        if staleness_secs > 360:
            _sym_to_td = {"XAUUSD": "XAU/USD", "EURUSD": "EUR/USD"}
            rt = realtime_feed.get_latest(_sym_to_td.get(symbol, symbol))
            if rt and rt.get("price"):
                live_p = round(float(rt["price"]), 5 if symbol == "EURUSD" else 2)
                now_floor = now_utc.replace(second=0, microsecond=0)
                now_floor = now_floor.replace(minute=(now_floor.minute // 5) * 5)
                candles.append({
                    "time": now_floor.isoformat(),
                    "open": live_p, "high": live_p, "low": live_p, "close": live_p,
                    "ema9": candles[-1]["ema9"],
                    "ema21": candles[-1]["ema21"],
                    "ema200": candles[-1]["ema200"],
                    "rsi": candles[-1]["rsi"],
                    "volume": 0.0,
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


@app.get("/api/trades/report")
def get_trade_report_endpoint(_user: dict = Depends(get_current_user)):
    return db.get_trade_report(limit=1000)


@app.get("/api/trades")
def get_trades(scope: str = "today", _user: dict = Depends(get_current_user)):
    mode = state.settings.get("mode")
    if scope == "today":
        trades = db.get_trades_for_day(db.today_utc(), mode=mode)
    else:
        trades = db.get_recent_trades(200, mode=mode)
    curve = db.get_equity_curve(source="live", limit=500)
    return {"trades": trades, "equity_curve": curve}


@app.post("/api/ai-report")
def get_ai_report(_user: dict = Depends(get_current_user)):
    """Génère un rapport d'analyse IA sur la situation du bot."""
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        raise HTTPException(400, detail="ANTHROPIC_API_KEY non configuré")
    try:
        import anthropic as _anthropic
        client = _anthropic.Anthropic(api_key=api_key)

        trade_report = db.get_trade_report(limit=500)
        researcher_status = state.researcher.status()
        risk_status = state.risk.status()
        llm_status = state.llm_gate.status()

        researcher_text = ""
        if researcher_status.get("results_total", 0) > 0:
            best = researcher_status.get("best_params") or {}
            top3 = researcher_status.get("top3") or []
            researcher_text = (
                f"\nCHERCHEUR DE PARAMÈTRES ({researcher_status['results_total']} expériences) :\n"
                f"Meilleurs params appliqués : {json.dumps(best)}\n"
                f"Top 3 :\n" +
                "\n".join(
                    f"  score={r.get('score',0):.3f} PF={r.get('profit_factor',0):.2f} "
                    f"WR={r.get('win_rate',0)*100:.0f}% n={r.get('total_trades',0)} → {r.get('params')}"
                    for r in top3
                )
            )

        llm_text = ""
        if llm_status.get("enabled"):
            llm_text = (
                f"\nLLM GATE : {llm_status.get('total_calls',0)} signaux analysés, "
                f"{llm_status.get('passed',0)} passés, {llm_status.get('held',0)} bloqués "
                f"(pass rate {llm_status.get('pass_rate',0)*100:.0f}%)"
            )

        prompt = (
            "Tu es un expert en trading algorithmique, spécialisé dans le scalping XAU/USD (Or).\n"
            "Analyse les données ci-dessous du bot et fournis en français :\n"
            "1. **Bilan de performance** : comment se porte le bot (WR, PF, PnL) ?\n"
            "2. **Points forts et points faibles** identifiés dans les données\n"
            "3. **Conseils concrets** pour améliorer les performances\n"
            "4. **Tendances** : heures/sessions/directions qui marchent ou non\n\n"
            f"{trade_report.get('llm_summary', 'Pas encore de trades.')}\n"
            f"{researcher_text}\n"
            f"{llm_text}\n\n"
            f"ÉTAT RISQUE : capital={risk_status.get('capital')}$ "
            f"| trades aujourd'hui={risk_status.get('trades_today')}/{risk_status.get('max_trades_per_day')} "
            f"| PnL jour={risk_status.get('realised_pnl_today')}$\n\n"
            "Sois direct, précis et actionnable. Maximum 400 mots."
        )

        resp = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=800,
            messages=[{"role": "user", "content": prompt}],
        )
        text = resp.content[0].text.strip()
        return {
            "report": text,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "trades_total": trade_report["stats"]["total"],
        }
    except Exception as exc:
        raise HTTPException(500, detail=str(exc))


@app.get("/api/adaptive-agent/status")
def adaptive_status(_user: dict = Depends(get_current_user)):
    """Retourne l'historique des runs de l'agent adaptatif."""
    return state.adaptive.status()


@app.post("/api/adaptive-agent/run")
def adaptive_run_now(_user: dict = Depends(get_current_user)):
    """Force un run immédiat de l'agent adaptatif."""
    result = state.adaptive.run_now()
    if "error" in result:
        raise HTTPException(400, detail=result["error"])
    return result


@app.post("/api/admin/reset-history")
def reset_all_history(_user: dict = Depends(get_current_user)):
    """Supprime TOUS les trades et remet les stats à zéro. Irréversible."""
    with state.lock:
        result = db.reset_all_trades()
        # Réinitialiser l'état en mémoire
        state.risk.hydrate_day(
            trades_today=0, pnl_today=0.0,
            start_equity=state.risk.capital,
            blocked=False,
        )
        state.risk.start_equity_today = state.risk.capital
    return result


@app.post("/api/admin/cleanup-duplicates")
def cleanup_duplicate_trades(_user: dict = Depends(get_current_user)):
    """Supprime les trades en double (même symbol+direction+entry_price dans la même minute).
    Garde le premier, supprime les suivants. Utile après un bug de double-entrée BOOTSTRAP."""
    with state.lock:
        result = db.delete_duplicate_trades()
        # Resynchroniser le P&L du jour après nettoyage
        today = db.today_utc()
        trades_today = db.get_trades_for_day(today, mode=state.settings.get("mode"))
        closed_today = [t for t in trades_today if t["status"] == "closed"]
        pnl_today = round(sum(t.get("pnl") or 0.0 for t in closed_today), 2)
        db.update_daily(today, {"pnl": pnl_today, "trade_count": len(trades_today)})
        state.risk.hydrate_day(
            trades_today=len(trades_today),
            pnl_today=pnl_today,
            start_equity=state.risk.start_equity_today or state.risk.capital,
            blocked=state.risk.blocked,
        )
    result["pnl_today"] = pnl_today
    result["trades_today"] = len(trades_today)
    return result


@app.delete("/api/trades/{trade_id}")
def delete_trade_by_id(trade_id: int, _user: dict = Depends(get_current_user)):
    """Supprime manuellement un trade de l'historique et resynchronise le P&L du jour."""
    with state.lock:
        deleted = db.delete_trade(trade_id)
        if not deleted:
            raise HTTPException(404, detail="Trade non trouvé")
        # Resynchroniser le P&L et trade_count du jour depuis la DB
        today = db.today_utc()
        trades_today = db.get_trades_for_day(today, mode=state.settings.get("mode"))
        closed_today = [t for t in trades_today if t["status"] == "closed"]
        pnl_today = round(sum(t.get("pnl") or 0.0 for t in closed_today), 2)
        db.update_daily(today, {"pnl": pnl_today, "trade_count": len(trades_today)})
        state.risk.hydrate_day(
            trades_today=len(trades_today),
            pnl_today=pnl_today,
            start_equity=state.risk.start_equity_today or state.risk.capital,
            blocked=state.risk.blocked,
        )
    return {"deleted": trade_id, "pnl_today": pnl_today, "trades_today": len(trades_today)}


@app.get("/api/strategy/blocked-hours")
def get_blocked_hours(_user: dict = Depends(get_current_user)):
    """Retourne la liste des heures CET actuellement bloquées."""
    return {"blocked_hours": sorted(getattr(strategy, "BAD_HOURS_CET", set()))}


@app.post("/api/strategy/blocked-hours/{hour}")
def toggle_blocked_hour(hour: int, _user: dict = Depends(get_current_user)):
    """Active/désactive le blocage d'une heure CET (0-23). Persisté en base."""
    if not (0 <= hour <= 23):
        raise HTTPException(400, detail="Heure invalide (0-23)")
    with state.lock:
        bad = set(getattr(strategy, "BAD_HOURS_CET", set()))
        if hour in bad:
            bad.discard(hour)
            action = "unblocked"
        else:
            bad.add(hour)
            action = "blocked"
        strategy.BAD_HOURS_CET = bad
        db.update_settings({"bad_hours_cet": sorted(bad)})
        state.settings = db.get_settings()
    return {"blocked_hours": sorted(bad), "action": action, "hour": hour}


@app.post("/api/risk/reset-daily")
def reset_daily_counter(_user: dict = Depends(get_current_user)):
    """Recalcule le compteur de trades du jour et le P&L depuis la DB.
    Utile après suppression manuelle de trades ou changement de max_trades_per_day."""
    with state.lock:
        today = db.today_utc()
        trades_today = db.get_trades_for_day(today, mode=state.settings.get("mode"))
        closed_today = [t for t in trades_today if t["status"] == "closed"]
        pnl_today = round(sum(t.get("pnl") or 0.0 for t in closed_today), 2)
        db.update_daily(today, {"pnl": pnl_today, "trade_count": len(trades_today)})
        state.risk.hydrate_day(
            trades_today=len(trades_today),
            pnl_today=pnl_today,
            start_equity=state.risk.start_equity_today or state.risk.capital,
            blocked=state.risk.blocked,
        )
    return {"trades_today": len(trades_today), "pnl_today": pnl_today}


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
    strategy: Optional[str] = None   # "A" (défaut) | "B" (ICT/SMC)


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
            m5, m15, h1, h4 = build_context(ms.broker)
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
    capital: float = 10_000.0
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


@app.post("/api/backtest/ict")
async def backtest_ict(req: BacktestRequest, _user: dict = Depends(get_current_user)):
    cfg = BacktestConfig(
        start=req.start, end=req.end, capital=req.capital,
        risk_pct=req.risk_pct, spread_pips=req.spread_pips,
        slippage_pips=req.slippage_pips,
        max_trades_per_day=req.max_trades_per_day,
        daily_stop_pct=req.daily_stop_pct,
        symbol=req.symbol,
        strategy_mode="ict",
    )
    try:
        result = await asyncio.to_thread(run_backtest, cfg)
        return result
    except Exception as exc:
        traceback.print_exc()
        return {"error": str(exc) or "Erreur interne du backtest ICT"}


class OptimizeRequest(BaseModel):
    start: str
    end: str
    symbol: str = "XAUUSD"
    capital: float = 10_000.0
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


# ── Pré-entraînement ─────────────────────────────────────────────────────────
class PretrainRequest(BaseModel):
    start:         str = "2024-01-01"
    end:           str = "2024-12-31"
    symbol:        str = "XAUUSD"
    atr_min:       Optional[float] = None
    reset:         bool = True
    capital:       float = 10_000.0
    risk_pct:      float = 5.0
    strategy_mode: str = "A"


@app.post("/api/pretrain")
def start_pretrain(req: PretrainRequest, _user: dict = Depends(get_current_user)):
    """Lance le pré-entraînement en arrière-plan."""
    prog = _pretrain_module.get_progress()
    if prog["running"]:
        return {"ok": False, "message": "Pré-entraînement déjà en cours", "progress": prog}
    _ms = state.market_states.get(req.symbol)
    _on_complete = _ms.ml_gate._load if _ms and _ms.ml_gate else None
    _pretrain_module.launch_pretrain(
        start=req.start, end=req.end,
        symbol=req.symbol, atr_min=req.atr_min, reset=req.reset,
        capital=req.capital, risk_pct=req.risk_pct,
        strategy_mode=req.strategy_mode,
        on_complete=_on_complete,
    )
    return {"ok": True, "message": "Pré-entraînement lancé", "progress": _pretrain_module.get_progress()}


# ── Walk-forward ──────────────────────────────────────────────────────────────
_wf_state: Dict[str, Any] = {"running": False, "window": 0, "n_splits": 4, "result": None, "error": None}
_wf_lock  = threading.Lock()


class WalkForwardRequest(BaseModel):
    start: str
    end: str
    n_splits: int = 4
    symbol: str = "XAUUSD"
    capital: float = 10_000.0
    risk_pct: float = 5.0
    strategy_mode: str = "A"


@app.post("/api/pretrain/walkforward")
def start_walkforward(req: WalkForwardRequest, _user: dict = Depends(get_current_user)):
    """Lance un walk-forward sur n_splits fenêtres indépendantes."""
    with _wf_lock:
        if _wf_state["running"] or _pretrain_module.get_progress()["running"]:
            return {"ok": False, "message": "Un pré-entraînement est déjà en cours"}
        _wf_state.update(running=True, window=0, n_splits=req.n_splits, result=None, error=None)

    def _run():
        try:
            r = _pretrain_module.run_walk_forward(
                start=req.start, end=req.end,
                n_splits=req.n_splits, symbol=req.symbol,
                capital=req.capital, risk_pct=req.risk_pct,
                strategy_mode=req.strategy_mode,
            )
            with _wf_lock:
                _wf_state.update(running=False, window=req.n_splits, result=r)
        except Exception as exc:
            with _wf_lock:
                _wf_state.update(running=False, error=str(exc))

    threading.Thread(target=_run, daemon=True, name="walkforward").start()
    return {"ok": True, "message": f"Walk-forward lancé ({req.n_splits} fenêtres)"}


@app.get("/api/pretrain/walkforward")
def get_walkforward(_user: dict = Depends(get_current_user)):
    with _wf_lock:
        return dict(_wf_state)


# ── Optimisation Bayésienne (Optuna) ──────────────────────────────────────────
_optuna_state: Dict[str, Any] = {
    "running": False, "progress": 0, "n_trials": 0, "best_score": 0.0,
    "result": None, "error": None,
}
_optuna_lock = threading.Lock()


class OptunaBayesRequest(BaseModel):
    start: str
    end: str
    n_trials: int = 30
    n_splits: int = 3
    symbol: str = "XAUUSD"
    capital: float = 10_000.0
    risk_pct: float = 5.0


@app.post("/api/optimize/bayesian")
def start_bayesian_optimize(req: OptunaBayesRequest, _user: dict = Depends(get_current_user)):
    """Optimisation Bayésienne des seuils stratégie via walk-forward."""
    with _optuna_lock:
        if _optuna_state["running"]:
            return {"ok": False, "message": "Optimisation déjà en cours"}
        _optuna_state.update(
            running=True, progress=0, n_trials=req.n_trials,
            best_score=0.0, result=None, error=None,
        )

    def _cb(done: int, total: int, score: float) -> None:
        with _optuna_lock:
            _optuna_state.update(
                progress=done,
                best_score=max(_optuna_state["best_score"], score),
            )

    def _run():
        try:
            from optimizer import run_optuna_optimize
            r = run_optuna_optimize(
                start=req.start, end=req.end,
                n_trials=req.n_trials, n_splits=req.n_splits,
                symbol=req.symbol, capital=req.capital, risk_pct=req.risk_pct,
                progress_cb=_cb,
            )
            with _optuna_lock:
                _optuna_state.update(running=False, result=r)
        except Exception as exc:
            with _optuna_lock:
                _optuna_state.update(running=False, error=str(exc))

    threading.Thread(target=_run, daemon=True, name="optuna").start()
    return {"ok": True, "message": f"Optimisation Bayésienne lancée ({req.n_trials} essais)"}


@app.get("/api/optimize/bayesian")
def get_bayesian_optimize(_user: dict = Depends(get_current_user)):
    with _optuna_lock:
        return dict(_optuna_state)


# ── Multi-period pretrain ─────────────────────────────────────────────────────
_multi_state: dict = {"running": False, "current": 0, "total": 3, "results": [], "error": None}
_multi_lock = threading.Lock()


@app.post("/api/pretrain/multi")
def start_multi_pretrain(req: PretrainRequest, _user: dict = Depends(get_current_user)):
    """Lance le pré-entraînement sur 3 périodes de 6 mois consécutives."""
    with _multi_lock:
        if _multi_state["running"] or _pretrain_module.get_progress()["running"]:
            return {"ok": False, "message": "Un pré-entraînement est déjà en cours"}
        _multi_state.update(running=True, current=0, total=3, results=[], error=None)

    from datetime import date, timedelta
    today = date.today()

    def _period(months_end: int, months_start: int) -> tuple[str, str]:
        end   = today - timedelta(days=months_end * 30)
        start = today - timedelta(days=months_start * 30)
        return start.isoformat(), end.isoformat()

    periods = [_period(0, 6), _period(6, 12), _period(12, 18)]
    def _label(start, end): return f"{start[:7]} → {end[:7]}"
    labels  = [_label(s, e) for s, e in periods]

    def _run():
        results = []
        try:
            for i, ((start, end), label) in enumerate(zip(periods, labels)):
                with _multi_lock:
                    _multi_state["current"] = i + 1
                r = _pretrain_module.run_pretrain(
                    start=start, end=end,
                    symbol=req.symbol, atr_min=req.atr_min,
                    reset=(i == 0 and req.reset),
                    capital=req.capital, risk_pct=req.risk_pct,
                    strategy_mode=req.strategy_mode,
                )
                n = r.get("n_trades", 0)
                n_sl = r.get("false_stops", {}).get("n_sl_direct", 0)
                diag = r.get("indicator_diagnostic", {})
                _sl  = diag.get("SL_direct", {}) or {}
                _tp2 = diag.get("TP2", {}) or {}
                results.append({
                    "label":         label,
                    "start":         start,
                    "end":           end,
                    "n_trades":      n,
                    "win_rate":      r.get("win_rate", 0),
                    "profit_factor": r.get("profit_factor", 0),
                    "net_pnl":       r.get("net_pnl", 0),
                    "sl_direct_pct": round(n_sl / max(n, 1) * 100, 1),
                    "avg_win":       r.get("avg_win", 0),
                    "avg_loss":      r.get("avg_loss", 0),
                    "equity_curve":  [{"equity": p["equity"]} for p in r.get("equity_curve", [])],
                    "diag_sl":  {"n": _sl.get("n", 0), "rsi_m5": _sl.get("rsi_m5"), "adx_h1": _sl.get("adx_h1"), "atr": _sl.get("atr"), "london_pct": _sl.get("london_pct")},
                    "diag_tp2": {"n": _tp2.get("n", 0), "rsi_m5": _tp2.get("rsi_m5"), "adx_h1": _tp2.get("adx_h1"), "atr": _tp2.get("atr"), "london_pct": _tp2.get("london_pct")},
                    "wr_by_hour":    r.get("wr_by_hour", {}),
                })
        except Exception as exc:
            with _multi_lock:
                _multi_state["error"] = str(exc)
        finally:
            with _multi_lock:
                _multi_state.update(running=False, results=results)

    threading.Thread(target=_run, daemon=True, name="pretrain-multi").start()
    return {"ok": True, "message": "Multi-périodes lancé"}


@app.get("/api/pretrain/multi")
def get_multi_pretrain(_user: dict = Depends(get_current_user)):
    """Statut et résultats du multi-period pretrain."""
    with _multi_lock:
        return dict(_multi_state)


@app.get("/api/pretrain/status")
def pretrain_status(_user: dict = Depends(get_current_user)):
    """Progression du pré-entraînement en cours."""
    return _pretrain_module.get_progress()


@app.get("/api/pretrain/trades")
def pretrain_trades(
    filter: str = "all",
    offset: int = 0,
    limit: int = 50,
    _user: dict = Depends(get_current_user),
):
    """Log détaillé des trades du dernier pré-entraînement, avec pagination."""
    prog = _pretrain_module.get_progress()
    result = prog.get("last_result") or {}
    log = result.get("trades_log", [])
    if filter == "losses":
        log = [t for t in log if not t["won"]]
    elif filter == "wins":
        log = [t for t in log if t["won"]]
    return {
        "total":  len(log),
        "offset": offset,
        "limit":  limit,
        "trades": log[offset: offset + limit],
    }


@app.get("/api/pretrain/stats")
def pretrain_stats(_user: dict = Depends(get_current_user)):
    """Stats agrégées du dernier pré-entraînement pour diagnostic stratégie."""
    from collections import defaultdict
    import statistics as _st

    prog = _pretrain_module.get_progress()
    result = prog.get("last_result") or {}
    log = result.get("trades_log", [])
    if not log:
        return {"error": "Pas de trades disponibles — lancez un pré-entraînement"}

    total = len(log)
    losses = [t for t in log if not t["won"]]
    wins   = [t for t in log if t["won"]]

    # ── Décomposition par raison de sortie ──────────────────────────────────
    buckets: dict = defaultdict(lambda: {"count": 0, "pnl": 0.0, "mae_r": [], "mfe_r": []})
    for t in log:
        b = buckets[t["exit_reason"]]
        b["count"] += 1
        b["pnl"]   += t["pnl"]
        b["mae_r"].append(t["mae_r"])
        b["mfe_r"].append(t["mfe_r"])

    exit_reasons = {
        er: {
            "count":     d["count"],
            "pct":       round(d["count"] / total * 100, 1),
            "avg_pnl":   round(d["pnl"] / d["count"], 2),
            "avg_mae_r": round(_st.mean(d["mae_r"]), 2) if d["mae_r"] else 0.0,
            "avg_mfe_r": round(_st.mean(d["mfe_r"]), 2) if d["mfe_r"] else 0.0,
        }
        for er, d in sorted(buckets.items(), key=lambda x: -x[1]["count"])
    }

    # ── WR par session × direction ───────────────────────────────────────────
    sd: dict = defaultdict(lambda: {"n": 0, "wins": 0})
    for t in log:
        key = f"{t.get('session', '?')}_{t['direction']}"
        sd[key]["n"]    += 1
        sd[key]["wins"] += int(t["won"])

    by_session_dir = {
        k: {"n": v["n"], "wins": v["wins"], "wr": round(v["wins"] / v["n"], 3)}
        for k, v in sorted(sd.items())
    }

    # ── Patterns les plus fréquents dans pertes vs gains ────────────────────
    pat_loss: dict = defaultdict(int)
    pat_win:  dict = defaultdict(int)
    for t in log:
        bucket = pat_win if t["won"] else pat_loss
        for p in (t.get("patterns") or []):
            bucket[p] += 1

    # ── Near-wins & lucky wins ───────────────────────────────────────────────
    near_wins_pct  = round(sum(1 for t in losses if t["mfe_r"] >= 0.5) / len(losses) * 100, 1) if losses else 0.0
    lucky_wins_pct = round(sum(1 for t in wins   if t["mae_r"] >= 0.5) / len(wins)   * 100, 1) if wins   else 0.0

    # ── False stops (SL direct → prix atteint TP1 dans les 10 bougies) ────────
    false_stops_data = result.get("false_stops", {})
    false_stops_pct  = false_stops_data.get("pct_false_stops", 0.0)

    # ── False breakevens (sl_after_tp1 → prix atteint TP2 dans les 20 bougies)
    false_be_data = result.get("false_breakevens", {})
    false_be_pct  = false_be_data.get("pct_false_bes", None)
    false_be_n    = false_be_data.get("n_sl_after_tp1", 0)

    # ── Couverture réelle des données ──────────────────────────────────────────
    data_cov = result.get("data_coverage", {})

    return {
        "total":               total,
        "n_wins":              len(wins),
        "n_losses":            len(losses),
        "exit_reasons":        exit_reasons,
        "by_session_dir":      by_session_dir,
        "top_patterns_losses": sorted(pat_loss.items(), key=lambda x: -x[1])[:6],
        "top_patterns_wins":   sorted(pat_win.items(),  key=lambda x: -x[1])[:6],
        "near_wins_pct":       near_wins_pct,
        "lucky_wins_pct":      lucky_wins_pct,
        "false_stops_pct":     false_stops_pct,
        "false_be_pct":        false_be_pct,
        "false_be_n":          false_be_n,
        "data_coverage":       data_cov,
        "indicator_diagnostic": result.get("indicator_diagnostic", {}),
        "wr_by_pattern":        result.get("wr_by_pattern", {}),
        "wr_by_session":        result.get("wr_by_session", {}),
        "wr_by_hour":           result.get("wr_by_hour", {}),
        "rejection_counts":     result.get("rejection_counts", {}),
    }


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


@app.get("/api/live-agent")
async def get_live_agent_status(user=Depends(get_current_user)):
    """Statut de l'agent adaptatif live XAUUSD (paramètres actuels, WR glissant, ajustements)."""
    return state.live_agent.status()


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
