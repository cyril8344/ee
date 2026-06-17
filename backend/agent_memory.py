"""
agent_memory.py
===============
Records trade context (market regime, indicator snapshot) alongside each
trade result so the perpetual learning agent can extract "lessons".

New SQLite tables (added to the existing DB):
  trade_context   — market conditions at trade open
  agent_lessons   — aggregated win rates per regime bucket

Market regime classification:
  - trend_strength : "strong" (ADX>28) / "moderate" (ADX 20-28) / "weak" (<20)
  - volatility     : "high" (ATR > 1.5×avg) / "normal" / "low" (<0.7×avg)
  - session        : "london" / "new_york" / "overlap" / "asian" / "off"
  - bias           : "LONG" / "SHORT" / "NEUTRAL"
"""

from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import database as db


# --------------------------------------------------------------------------- #
# Table creation
# --------------------------------------------------------------------------- #

def init_memory_tables(conn: sqlite3.Connection) -> None:
    """Create trade_context and agent_lessons tables if they don't exist yet."""
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS trade_context (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            trade_id        INTEGER NOT NULL,
            recorded_at     TEXT NOT NULL,
            adx             REAL,
            atr             REAL,
            atr_avg         REAL,
            rsi_m5          REAL,
            rsi_m15         REAL,
            session         TEXT,
            bias            TEXT,
            spread          REAL,
            vix             REAL,
            trend_strength  TEXT,   -- "strong" | "moderate" | "weak"
            volatility      TEXT,   -- "high" | "normal" | "low"
            won             INTEGER -- 1 = win, 0 = loss
        );

        CREATE TABLE IF NOT EXISTS agent_lessons (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            regime_key      TEXT NOT NULL UNIQUE,
            trend_strength  TEXT,
            volatility      TEXT,
            session         TEXT,
            bias            TEXT,
            trade_count     INTEGER NOT NULL DEFAULT 0,
            win_count       INTEGER NOT NULL DEFAULT 0,
            win_rate        REAL NOT NULL DEFAULT 0.0,
            updated_at      TEXT NOT NULL
        );
    """)


def _ensure_tables() -> None:
    """Ensure memory tables exist (idempotent)."""
    with db.get_conn() as conn:
        init_memory_tables(conn)


# --------------------------------------------------------------------------- #
# Regime helpers
# --------------------------------------------------------------------------- #

def _classify_trend(adx: Optional[float]) -> str:
    if adx is None:
        return "unknown"
    if adx > 28:
        return "strong"
    if adx >= 20:
        return "moderate"
    return "weak"


def _classify_volatility(atr: Optional[float], atr_avg: Optional[float]) -> str:
    if atr is None or atr_avg is None or atr_avg == 0:
        return "unknown"
    ratio = atr / atr_avg
    if ratio > 1.5:
        return "high"
    if ratio < 0.7:
        return "low"
    return "normal"


def _regime_key(trend: str, volatility: str, session: str, bias: str) -> str:
    return f"{trend}|{volatility}|{session}|{bias}"


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #

def record_trade_context(trade_id: int, context: Dict[str, Any]) -> None:
    """
    Insert a row into trade_context for the given trade.

    context keys (all optional):
        adx, atr, atr_avg, rsi_m5, rsi_m15, session, bias, spread, vix, won
    """
    _ensure_tables()

    adx = context.get("adx")
    atr = context.get("atr")
    atr_avg = context.get("atr_avg")
    session = (context.get("session") or "off").lower()
    bias = (context.get("bias") or "NEUTRAL").upper()
    won = 1 if context.get("won") else 0

    trend_strength = _classify_trend(adx)
    volatility = _classify_volatility(atr, atr_avg)
    regime = _regime_key(trend_strength, volatility, session, bias)
    now = datetime.now(timezone.utc).isoformat()

    with db.get_conn() as conn:
        init_memory_tables(conn)
        conn.execute("""
            INSERT INTO trade_context
                (trade_id, recorded_at, adx, atr, atr_avg, rsi_m5, rsi_m15,
                 session, bias, spread, vix, trend_strength, volatility, won)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            trade_id, now,
            adx, atr, atr_avg,
            context.get("rsi_m5"), context.get("rsi_m15"),
            session, bias,
            context.get("spread"), context.get("vix"),
            trend_strength, volatility, won,
        ))

        # Update aggregated lessons
        conn.execute("""
            INSERT INTO agent_lessons
                (regime_key, trend_strength, volatility, session, bias,
                 trade_count, win_count, win_rate, updated_at)
            VALUES (?, ?, ?, ?, ?, 1, ?, ?, ?)
            ON CONFLICT(regime_key) DO UPDATE SET
                trade_count = trade_count + 1,
                win_count   = win_count + excluded.win_count,
                win_rate    = CAST(win_count + excluded.win_count AS REAL)
                              / (trade_count + 1),
                updated_at  = excluded.updated_at
        """, (
            regime, trend_strength, volatility, session, bias,
            won, float(won), now,
        ))


def get_lessons() -> List[Dict[str, Any]]:
    """
    Return aggregated win rates per regime bucket, ordered by trade count desc.
    Each dict has: regime_key, trend_strength, volatility, session, bias,
                   trade_count, win_count, win_rate.
    """
    _ensure_tables()
    with db.get_conn() as conn:
        init_memory_tables(conn)
        rows = conn.execute("""
            SELECT regime_key, trend_strength, volatility, session, bias,
                   trade_count, win_count, win_rate
            FROM agent_lessons
            ORDER BY trade_count DESC
        """).fetchall()
    return [dict(r) for r in rows]


def get_best_regime(min_trades: int = 10) -> Optional[Dict[str, Any]]:
    """Return the regime bucket with highest win rate (minimum min_trades trades)."""
    _ensure_tables()
    with db.get_conn() as conn:
        init_memory_tables(conn)
        row = conn.execute("""
            SELECT regime_key, trend_strength, volatility, session, bias,
                   trade_count, win_count, win_rate
            FROM agent_lessons
            WHERE trade_count >= ?
            ORDER BY win_rate DESC
            LIMIT 1
        """, (min_trades,)).fetchone()
    return dict(row) if row else None


def get_worst_regime(min_trades: int = 10) -> Optional[Dict[str, Any]]:
    """Return the regime bucket with lowest win rate (minimum min_trades trades)."""
    _ensure_tables()
    with db.get_conn() as conn:
        init_memory_tables(conn)
        row = conn.execute("""
            SELECT regime_key, trend_strength, volatility, session, bias,
                   trade_count, win_count, win_rate
            FROM agent_lessons
            WHERE trade_count >= ?
            ORDER BY win_rate ASC
            LIMIT 1
        """, (min_trades,)).fetchone()
    return dict(row) if row else None
