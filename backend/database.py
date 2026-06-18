"""
database.py
===========
SQLite persistence layer for the XAU/USD scalping bot.

Tables
------
- settings        : single-row key/value bot configuration
- trades          : every closed (and open) trade
- equity_points   : intraday / historical equity curve samples
- daily_stats     : aggregated per-day stats (pnl, trade count, blocked flag)

The module exposes a small, dependency-free API used by the rest of the
backend.  All timestamps are stored as ISO-8601 strings in UTC.
"""

import os
import sqlite3
import json
from datetime import datetime, timezone, date
from contextlib import contextmanager
from typing import Optional, List, Dict, Any

# Use /data (Railway persistent volume) when available, fall back to local
_DEFAULT_DB = (
    "/data/xau_bot.db"
    if os.path.isdir("/data")
    else os.path.join(os.path.dirname(os.path.abspath(__file__)), "xau_bot.db")
)
DB_PATH = os.environ.get("XAU_DB_PATH", _DEFAULT_DB)


# --------------------------------------------------------------------------- #
# Connection helpers
# --------------------------------------------------------------------------- #
@contextmanager
def get_conn():
    """Context manager yielding a configured sqlite3 connection."""
    conn = sqlite3.connect(DB_PATH, detect_types=sqlite3.PARSE_DECLTYPES)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA foreign_keys=ON;")
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# --------------------------------------------------------------------------- #
# Schema
# --------------------------------------------------------------------------- #
DEFAULT_SETTINGS = {
    "capital": 1000.0,
    "risk_per_trade_pct": 5.0,     # 5% = 50€ sur 1000€ de capital
    "max_trades_per_day": 4,
    "daily_stop_pct": 100.0,       # -100% = -1000€ (pratiquement désactivé)
    "mode": "paper",               # 'paper' | 'live'
    "symbol": "XAUUSD",
    "active_markets": ["XAUUSD", "EURUSD"],
    "spread_pips": 0.3,
    "slippage_pips": 0.1,
    "bot_enabled": True,
}


def init_db() -> None:
    """Create tables and seed default settings if needed."""
    with get_conn() as conn:
        c = conn.cursor()
        c.executescript(
            """
            CREATE TABLE IF NOT EXISTS settings (
                id          INTEGER PRIMARY KEY CHECK (id = 1),
                data        TEXT NOT NULL,
                updated_at  TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS trades (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol        TEXT NOT NULL DEFAULT 'XAUUSD',
                direction     TEXT NOT NULL,           -- 'long' | 'short'
                session       TEXT,                    -- 'London' | 'NewYork'
                entry_time    TEXT NOT NULL,
                exit_time     TEXT,
                entry_price   REAL NOT NULL,
                exit_price    REAL,
                stop_loss     REAL NOT NULL,
                take_profit1  REAL NOT NULL,
                take_profit2  REAL NOT NULL,
                volume        REAL NOT NULL,           -- lots
                risk_amount   REAL NOT NULL,           -- $ risked
                pnl           REAL,                    -- realised $ (net of costs)
                pnl_pct       REAL,
                duration_min  REAL,
                status        TEXT NOT NULL DEFAULT 'open',  -- open|closed
                exit_reason   TEXT,                    -- tp1|tp2|sl|timeout|manual
                mode          TEXT NOT NULL DEFAULT 'paper',
                meta          TEXT                     -- JSON extras
            );

            CREATE TABLE IF NOT EXISTS equity_points (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                ts          TEXT NOT NULL,
                equity      REAL NOT NULL,
                source      TEXT NOT NULL DEFAULT 'live'  -- live|backtest
            );

            CREATE TABLE IF NOT EXISTS daily_stats (
                day           TEXT PRIMARY KEY,          -- YYYY-MM-DD (UTC)
                start_equity  REAL NOT NULL,
                pnl           REAL NOT NULL DEFAULT 0,
                trade_count   INTEGER NOT NULL DEFAULT 0,
                blocked       INTEGER NOT NULL DEFAULT 0,
                updated_at    TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS pattern_stats (
                pattern     TEXT PRIMARY KEY,
                trades      INTEGER DEFAULT 0,
                wins        INTEGER DEFAULT 0,
                updated_at  TEXT
            );
            """
        )

        # Seed settings
        row = c.execute("SELECT id FROM settings WHERE id = 1").fetchone()
        if row is None:
            c.execute(
                "INSERT INTO settings (id, data, updated_at) VALUES (1, ?, ?)",
                (json.dumps(DEFAULT_SETTINGS), _utcnow_iso()),
            )
        else:
            # One-time migrations sur active_markets pour les bases existantes.
            data_row = c.execute("SELECT data FROM settings WHERE id = 1").fetchone()
            existing = json.loads(data_row[0]) if data_row else {}
            am = existing.get("active_markets")
            changed = False
            if am == ["XAUUSD"]:
                am = ["XAUUSD", "EURUSD"]
                changed = True
            # Retire l'argent (XAG/USD) : on se concentre d'abord sur or + euro/dollar.
            if isinstance(am, list) and "XAGUSD" in am:
                am = [m for m in am if m != "XAGUSD"]
                changed = True
            if changed:
                existing["active_markets"] = am
                c.execute(
                    "UPDATE settings SET data = ?, updated_at = ? WHERE id = 1",
                    (json.dumps(existing), _utcnow_iso()),
                )


# --------------------------------------------------------------------------- #
# Settings
# --------------------------------------------------------------------------- #
def get_settings() -> Dict[str, Any]:
    with get_conn() as conn:
        row = conn.execute("SELECT data FROM settings WHERE id = 1").fetchone()
        if row is None:
            return dict(DEFAULT_SETTINGS)
        data = json.loads(row["data"])
        # Backfill any newly-added default keys.
        merged = dict(DEFAULT_SETTINGS)
        merged.update(data)
        return merged


def update_settings(patch: Dict[str, Any]) -> Dict[str, Any]:
    current = get_settings()
    current.update(patch)
    with get_conn() as conn:
        conn.execute(
            "UPDATE settings SET data = ?, updated_at = ? WHERE id = 1",
            (json.dumps(current), _utcnow_iso()),
        )
    return current


# --------------------------------------------------------------------------- #
# Trades
# --------------------------------------------------------------------------- #
def insert_trade(trade: Dict[str, Any]) -> int:
    fields = [
        "symbol", "direction", "session", "entry_time", "exit_time",
        "entry_price", "exit_price", "stop_loss", "take_profit1",
        "take_profit2", "volume", "risk_amount", "pnl", "pnl_pct",
        "duration_min", "status", "exit_reason", "mode", "meta",
    ]
    values = []
    for f in fields:
        v = trade.get(f)
        if f == "meta" and isinstance(v, (dict, list)):
            v = json.dumps(v)
        values.append(v)
    placeholders = ", ".join("?" for _ in fields)
    with get_conn() as conn:
        cur = conn.execute(
            f"INSERT INTO trades ({', '.join(fields)}) VALUES ({placeholders})",
            values,
        )
        return cur.lastrowid


def update_trade(trade_id: int, patch: Dict[str, Any]) -> None:
    if not patch:
        return
    sets = []
    values = []
    for k, v in patch.items():
        if k == "meta" and isinstance(v, (dict, list)):
            v = json.dumps(v)
        sets.append(f"{k} = ?")
        values.append(v)
    values.append(trade_id)
    with get_conn() as conn:
        conn.execute(
            f"UPDATE trades SET {', '.join(sets)} WHERE id = ?", values
        )


def _row_to_dict(row: sqlite3.Row) -> Dict[str, Any]:
    d = dict(row)
    if d.get("meta"):
        try:
            d["meta"] = json.loads(d["meta"])
        except (json.JSONDecodeError, TypeError):
            pass
    return d


def get_open_trade(mode: Optional[str] = None) -> Optional[Dict[str, Any]]:
    q = "SELECT * FROM trades WHERE status = 'open'"
    params: List[Any] = []
    if mode:
        q += " AND mode = ?"
        params.append(mode)
    q += " ORDER BY id DESC LIMIT 1"
    with get_conn() as conn:
        row = conn.execute(q, params).fetchone()
        return _row_to_dict(row) if row else None


def get_trades_for_day(day: str, mode: Optional[str] = None) -> List[Dict[str, Any]]:
    """day = 'YYYY-MM-DD' (UTC).  Matches on entry_time prefix."""
    q = "SELECT * FROM trades WHERE substr(entry_time, 1, 10) = ?"
    params: List[Any] = [day]
    if mode:
        q += " AND mode = ?"
        params.append(mode)
    q += " ORDER BY entry_time ASC"
    with get_conn() as conn:
        rows = conn.execute(q, params).fetchall()
        return [_row_to_dict(r) for r in rows]


def get_open_trades() -> List[Dict[str, Any]]:
    """Return all trades currently marked as open (for position recovery on restart)."""
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT * FROM trades WHERE status = 'open' ORDER BY id ASC"
        ).fetchall()
        return [_row_to_dict(r) for r in rows]


def get_recent_trades(limit: int = 100, mode: Optional[str] = None) -> List[Dict[str, Any]]:
    q = "SELECT * FROM trades"
    params: List[Any] = []
    if mode:
        q += " WHERE mode = ?"
        params.append(mode)
    q += " ORDER BY id DESC LIMIT ?"
    params.append(limit)
    with get_conn() as conn:
        rows = conn.execute(q, params).fetchall()
        return [_row_to_dict(r) for r in rows]


# --------------------------------------------------------------------------- #
# Equity
# --------------------------------------------------------------------------- #
def add_equity_point(equity: float, source: str = "live", ts: Optional[str] = None) -> None:
    with get_conn() as conn:
        conn.execute(
            "INSERT INTO equity_points (ts, equity, source) VALUES (?, ?, ?)",
            (ts or _utcnow_iso(), float(equity), source),
        )


def get_equity_curve(source: str = "live", limit: int = 1000) -> List[Dict[str, Any]]:
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT ts, equity FROM equity_points WHERE source = ? "
            "ORDER BY id DESC LIMIT ?",
            (source, limit),
        ).fetchall()
        return [dict(r) for r in reversed(rows)]


# --------------------------------------------------------------------------- #
# Daily stats
# --------------------------------------------------------------------------- #
def get_or_create_daily(day: str, start_equity: float) -> Dict[str, Any]:
    with get_conn() as conn:
        row = conn.execute(
            "SELECT * FROM daily_stats WHERE day = ?", (day,)
        ).fetchone()
        if row:
            return dict(row)
        conn.execute(
            "INSERT INTO daily_stats (day, start_equity, pnl, trade_count, "
            "blocked, updated_at) VALUES (?, ?, 0, 0, 0, ?)",
            (day, float(start_equity), _utcnow_iso()),
        )
        row = conn.execute(
            "SELECT * FROM daily_stats WHERE day = ?", (day,)
        ).fetchone()
        return dict(row)


def update_daily(day: str, patch: Dict[str, Any]) -> None:
    if not patch:
        return
    sets, values = [], []
    for k, v in patch.items():
        sets.append(f"{k} = ?")
        values.append(v)
    sets.append("updated_at = ?")
    values.append(_utcnow_iso())
    values.append(day)
    with get_conn() as conn:
        conn.execute(
            f"UPDATE daily_stats SET {', '.join(sets)} WHERE day = ?", values
        )


def get_daily(day: str) -> Optional[Dict[str, Any]]:
    with get_conn() as conn:
        row = conn.execute(
            "SELECT * FROM daily_stats WHERE day = ?", (day,)
        ).fetchone()
        return dict(row) if row else None


def today_utc() -> str:
    return date.today().isoformat()


# --------------------------------------------------------------------------- #
# Pattern stats
# --------------------------------------------------------------------------- #
def get_pattern_stats() -> Dict[str, Dict]:
    """Return {pattern: {trades, wins, win_rate, weight}} for all patterns."""
    with get_conn() as conn:
        rows = conn.execute("SELECT pattern, trades, wins FROM pattern_stats").fetchall()
    result = {}
    for row in rows:
        pattern, trades, wins = row["pattern"], row["trades"], row["wins"]
        win_rate = (wins + 2) / (trades + 4)  # Laplace smoothing
        # weight: 50% win_rate -> 1.0, 70% -> 1.8, 30% -> 0.3 (capped)
        weight = max(0.3, min(2.0, win_rate * 2.0)) if trades >= 5 else 1.0
        result[pattern] = {
            "trades": trades,
            "wins": wins,
            "win_rate": round((wins / trades * 100) if trades > 0 else 50.0, 1),
            "weight": round(weight, 3),
        }
    return result


def reset_pattern_stats() -> None:
    """Delete all pattern statistics (reset learning to neutral weights)."""
    with get_conn() as conn:
        conn.execute("DELETE FROM pattern_stats")


def update_pattern_stats(patterns: List[str], won: bool) -> None:
    """Increment trades (+1) and optionally wins (+1) for each pattern."""
    if not patterns:
        return
    now = datetime.utcnow().isoformat()
    with get_conn() as conn:
        for p in patterns:
            conn.execute("""
                INSERT INTO pattern_stats (pattern, trades, wins, updated_at)
                VALUES (?, 1, ?, ?)
                ON CONFLICT(pattern) DO UPDATE SET
                    trades = trades + 1,
                    wins = wins + ?,
                    updated_at = ?
            """, (p, 1 if won else 0, now, 1 if won else 0, now))


if __name__ == "__main__":
    init_db()
    print(f"Database initialised at {DB_PATH}")
    print("Settings:", json.dumps(get_settings(), indent=2))
