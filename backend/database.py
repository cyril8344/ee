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
    "risk_per_trade_pct": 2.0,     # 2% = 20€ sur 1000€ de capital
    "max_trades_per_day": 4,
    "daily_stop_pct": 8.0,         # -8% stop journalier (4 pertes max avant arrêt)
    "mode": "paper",               # 'paper' | 'live'
    "symbol": "XAUUSD",
    "active_markets": ["XAUUSD", "EURUSD"],
    "spread_pips": 0.3,
    "slippage_pips": 0.1,
    "bot_enabled": True,
}

INSTRUMENT_DEFAULTS: Dict[str, Dict[str, Any]] = {
    "XAUUSD": {
        "bot_enabled": True,
        "max_trades_per_day": 4,
        "risk_pct": 5.0,
        "daily_stop_pct": 2.0,
        "spread_pips": 0.3,
        "slippage_pips": 0.1,
        "bad_hours_cet": [8, 10, 14],
    },
    "EURUSD": {
        "bot_enabled": True,
        "max_trades_per_day": 4,
        "risk_pct": 2.0,
        "daily_stop_pct": 5.0,
        "spread_pips": 0.2,
        "slippage_pips": 0.05,
        "bad_hours_cet": [8, 14],
    },
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

            CREATE TABLE IF NOT EXISTS ml_gate (
                symbol      TEXT PRIMARY KEY,
                data        TEXT NOT NULL,
                updated_at  TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS adaptive_thresholds (
                symbol      TEXT PRIMARY KEY,
                data        TEXT NOT NULL,
                updated_at  TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS ohlcv_cache (
                key       TEXT PRIMARY KEY,
                symbol    TEXT NOT NULL,
                start     TEXT NOT NULL,
                end       TEXT NOT NULL,
                data      BLOB NOT NULL,
                saved_at  TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS live_agent (
                symbol     TEXT PRIMARY KEY,
                params     TEXT NOT NULL,
                trade_log  TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS instrument_settings (
                symbol     TEXT PRIMARY KEY,
                data       TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            """
        )

        # Migrate ml_gate table from singleton (id=1) to per-symbol schema
        try:
            cols = [r[1] for r in c.execute("PRAGMA table_info(ml_gate)").fetchall()]
            if "id" in cols and "symbol" not in cols:
                old = c.execute("SELECT data, updated_at FROM ml_gate WHERE id = 1").fetchone()
                c.execute("DROP TABLE ml_gate")
                c.execute("""
                    CREATE TABLE ml_gate (
                        symbol      TEXT PRIMARY KEY,
                        data        TEXT NOT NULL,
                        updated_at  TEXT NOT NULL
                    )
                """)
                if old:
                    c.execute(
                        "INSERT INTO ml_gate (symbol, data, updated_at) VALUES ('XAUUSD', ?, ?)",
                        old,
                    )
        except Exception:
            pass

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
# Instrument settings (per-symbol)
# --------------------------------------------------------------------------- #
def get_instrument_settings(symbol: str) -> Dict[str, Any]:
    defaults = dict(INSTRUMENT_DEFAULTS.get(symbol, INSTRUMENT_DEFAULTS["XAUUSD"]))
    with get_conn() as conn:
        row = conn.execute(
            "SELECT data FROM instrument_settings WHERE symbol = ?", (symbol,)
        ).fetchone()
        if row:
            data = json.loads(row["data"])
            defaults.update(data)
    return defaults


def update_instrument_settings(symbol: str, patch: Dict[str, Any]) -> Dict[str, Any]:
    current = get_instrument_settings(symbol)
    current.update(patch)
    now = _utcnow_iso()
    with get_conn() as conn:
        conn.execute(
            "INSERT INTO instrument_settings (symbol, data, updated_at) VALUES (?, ?, ?)"
            " ON CONFLICT(symbol) DO UPDATE SET data = excluded.data, updated_at = excluded.updated_at",
            (symbol, json.dumps(current), now),
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


def update_trade(trade_id: int, patch: Dict[str, Any]) -> bool:
    """Retourne True si la ligne a été trouvée et mise à jour, False sinon (ex: supprimée par reset)."""
    if not patch:
        return True
    sets = []
    values = []
    for k, v in patch.items():
        if k == "meta" and isinstance(v, (dict, list)):
            v = json.dumps(v)
        sets.append(f"{k} = ?")
        values.append(v)
    values.append(trade_id)
    with get_conn() as conn:
        cur = conn.execute(
            f"UPDATE trades SET {', '.join(sets)} WHERE id = ?", values
        )
        return cur.rowcount > 0


def delete_trade(trade_id: int) -> bool:
    with get_conn() as conn:
        cur = conn.execute("DELETE FROM trades WHERE id = ?", (trade_id,))
        return cur.rowcount > 0


def delete_duplicate_trades() -> Dict[str, Any]:
    """Supprime les trades en double : même symbol+direction+entry_price dans la même minute.
    Garde le premier inséré (id le plus bas), supprime les autres.
    Retourne le nombre de doublons supprimés et leurs ids."""
    with get_conn() as conn:
        # Trouver les groupes avec plus d'un trade identique (même minute arrondie)
        rows = conn.execute("""
            SELECT symbol, direction, entry_price,
                   substr(entry_time, 1, 16) AS minute,
                   GROUP_CONCAT(id ORDER BY id) AS ids,
                   COUNT(*) AS cnt
            FROM trades
            GROUP BY symbol, direction, entry_price, substr(entry_time, 1, 16)
            HAVING COUNT(*) > 1
        """).fetchall()

        deleted_ids = []
        for row in rows:
            ids = [int(x) for x in row["ids"].split(",")]
            to_delete = ids[1:]  # garder le premier (id le plus bas)
            for tid in to_delete:
                conn.execute("DELETE FROM trades WHERE id = ?", (tid,))
                deleted_ids.append(tid)

    return {"deleted": len(deleted_ids), "ids": deleted_ids}


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


def get_trades_for_day_by_symbol(day: str, symbol: str, mode: Optional[str] = None) -> List[Dict[str, Any]]:
    """Same as get_trades_for_day but filtered by symbol."""
    q = "SELECT * FROM trades WHERE substr(entry_time, 1, 10) = ? AND symbol = ?"
    params: List[Any] = [day, symbol]
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


def reset_all_trades() -> Dict[str, Any]:
    """Supprime TOUS les trades et remet daily_stats à zéro. Irréversible."""
    with get_conn() as conn:
        count = conn.execute("SELECT COUNT(*) FROM trades").fetchone()[0]
        conn.execute("DELETE FROM trades")
        conn.execute("DELETE FROM daily_stats")
    return {"deleted": count}


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


def reset_pattern_stats(rebuild_from_symbol: Optional[str] = None) -> int:
    """
    Efface toutes les statistiques de patterns.
    Si rebuild_from_symbol est fourni (ex: 'XAUUSD'), rejoue les trades fermés
    de ce symbole pour reconstruire les stats — le reste reste à neutre.
    Retourne le nombre de trades rejoués.
    """
    with get_conn() as conn:
        conn.execute("DELETE FROM pattern_stats")

    if not rebuild_from_symbol:
        return 0

    # Rejouer les trades fermés du symbole voulu
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT meta, pnl FROM trades WHERE symbol = ? AND status = 'closed'",
            (rebuild_from_symbol,)
        ).fetchall()

    replayed = 0
    for row in rows:
        try:
            meta = json.loads(row["meta"]) if isinstance(row["meta"], str) else (row["meta"] or {})
            triggers = meta.get("triggers", [])
            pnl = row["pnl"] or 0.0
            if triggers:
                update_pattern_stats(triggers, won=pnl > 0)
                replayed += 1
        except Exception:
            pass
    return replayed


def update_pattern_stats(patterns: List[str], won: bool) -> None:
    """Increment trades (+1) and optionally wins (+1) for each pattern."""
    if not patterns:
        return
    now = datetime.now(timezone.utc).isoformat()
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


def save_ml_weights(weights: list, bias_w: float, n_samples: int,
                    consecutive_losses: int = 0, symbol: str = "XAUUSD",
                    live_source: bool = False) -> None:
    """Persist ML gate weights to DB (per symbol)."""
    data = json.dumps({
        "weights": weights, "bias_w": bias_w, "n_samples": n_samples,
        "consecutive_losses": consecutive_losses,
        "live_source": live_source,
    })
    now  = _utcnow_iso()
    with get_conn() as conn:
        conn.execute("""
            INSERT INTO ml_gate (symbol, data, updated_at) VALUES (?, ?, ?)
            ON CONFLICT(symbol) DO UPDATE SET data = ?, updated_at = ?
        """, (symbol, data, now, data, now))


def save_adaptive_thresholds(symbol: str, data: dict) -> None:
    payload = json.dumps(data)
    now     = _utcnow_iso()
    with get_conn() as conn:
        conn.execute("""
            INSERT INTO adaptive_thresholds (symbol, data, updated_at) VALUES (?, ?, ?)
            ON CONFLICT(symbol) DO UPDATE SET data = ?, updated_at = ?
        """, (symbol, payload, now, payload, now))


def load_adaptive_thresholds(symbol: str) -> dict:
    with get_conn() as conn:
        row = conn.execute(
            "SELECT data FROM adaptive_thresholds WHERE symbol = ?", (symbol,)
        ).fetchone()
    if row is None:
        return {}
    try:
        return json.loads(row[0])
    except Exception:
        return {}


def load_ml_weights(symbol: str = "XAUUSD") -> dict:
    """Load ML gate weights from DB for a given symbol. Returns {} if not found."""
    with get_conn() as conn:
        row = conn.execute("SELECT data FROM ml_gate WHERE symbol = ?", (symbol,)).fetchone()
    if row is None:
        return {}
    try:
        return json.loads(row[0])
    except Exception:
        return {}


# --------------------------------------------------------------------------- #
# OHLCV cache (persiste dans le volume Railway /data)
# --------------------------------------------------------------------------- #
_OHLCV_CACHE_TTL_HOURS = 168  # 7 jours — données historiques immuables


def ohlcv_cache_load(key: str) -> Optional[bytes]:
    with get_conn() as conn:
        row = conn.execute(
            "SELECT data, saved_at FROM ohlcv_cache WHERE key = ?", (key,)
        ).fetchone()
    if row is None:
        return None
    saved = datetime.fromisoformat(row["saved_at"])
    if saved.tzinfo is None:
        saved = saved.replace(tzinfo=timezone.utc)
    age_h = (datetime.now(timezone.utc) - saved).total_seconds() / 3600
    if age_h > _OHLCV_CACHE_TTL_HOURS:
        return None
    return row["data"]


def ohlcv_cache_save(key: str, symbol: str, start: str, end: str, data: bytes) -> None:
    now = _utcnow_iso()
    with get_conn() as conn:
        conn.execute(
            """
            INSERT INTO ohlcv_cache (key, symbol, start, end, data, saved_at)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(key) DO UPDATE SET data = ?, saved_at = ?
            """,
            (key, symbol, start, end, data, now, data, now),
        )


def live_agent_load(symbol: str) -> Optional[dict]:
    with get_conn() as conn:
        row = conn.execute(
            "SELECT params, trade_log FROM live_agent WHERE symbol = ?", (symbol,)
        ).fetchone()
    if row is None:
        return None
    return {"params": json.loads(row["params"]), "trade_log": json.loads(row["trade_log"])}


def live_agent_save(symbol: str, params: dict, trade_log: list) -> None:
    now = _utcnow_iso()
    with get_conn() as conn:
        conn.execute(
            """
            INSERT INTO live_agent (symbol, params, trade_log, updated_at)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(symbol) DO UPDATE SET params = ?, trade_log = ?, updated_at = ?
            """,
            (symbol, json.dumps(params), json.dumps(trade_log), now,
             json.dumps(params), json.dumps(trade_log), now),
        )


# --------------------------------------------------------------------------- #
# Trade report (pour dashboard + agent IA)
# --------------------------------------------------------------------------- #
def get_trade_report(limit: int = 500, symbol: str | None = None) -> Dict[str, Any]:
    """Rapport complet de l'historique pour le dashboard et l'agent IA."""
    try:
        import pytz as _pytz
        _CET = _pytz.timezone("Europe/Paris")
    except Exception:
        _CET = None

    from collections import defaultdict

    with get_conn() as conn:
        if symbol:
            rows = conn.execute(
                "SELECT * FROM trades WHERE status = 'closed' AND symbol = ? ORDER BY entry_time ASC LIMIT ?",
                (symbol, limit),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM trades WHERE status = 'closed' ORDER BY entry_time ASC LIMIT ?",
                (limit,),
            ).fetchall()

    trades_list = [_row_to_dict(r) for r in rows]

    for t in trades_list:
        try:
            raw = t.get("entry_time", "")
            ts = datetime.fromisoformat(raw.replace("Z", "+00:00"))
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)
            ts_cet = ts.astimezone(_CET) if _CET else ts
            t["date_cet"] = ts_cet.strftime("%Y-%m-%d")
            t["hour_cet"] = ts_cet.hour
            t["datetime_cet"] = ts_cet.strftime("%Y-%m-%d %H:%M")
        except Exception:
            t["date_cet"] = "?"
            t["hour_cet"] = None
            t["datetime_cet"] = "?"

    closed = [t for t in trades_list if t.get("pnl") is not None]
    total = len(closed)
    wins_list = [t for t in closed if (t["pnl"] or 0) > 0]
    loss_list = [t for t in closed if (t["pnl"] or 0) <= 0]
    gross_profit = sum(t["pnl"] for t in wins_list)
    gross_loss = abs(sum(t["pnl"] for t in loss_list)) if loss_list else 0.0
    total_pnl = sum(t["pnl"] for t in closed)

    stats = {
        "total": total,
        "wins": len(wins_list),
        "losses": len(loss_list),
        "win_rate": round(len(wins_list) / total * 100, 1) if total > 0 else 0.0,
        "profit_factor": round(gross_profit / gross_loss, 2) if gross_loss > 0 else 0.0,
        "total_pnl": round(total_pnl, 2),
        "avg_win": round(gross_profit / len(wins_list), 2) if wins_list else 0.0,
        "avg_loss": round(-gross_loss / len(loss_list), 2) if loss_list else 0.0,
    }

    _by_hour: Dict[int, Dict] = defaultdict(lambda: {"n": 0, "wins": 0, "pnl": 0.0})
    _by_sess: Dict[str, Dict] = defaultdict(lambda: {"n": 0, "wins": 0, "pnl": 0.0})
    _by_dir:  Dict[str, Dict] = defaultdict(lambda: {"n": 0, "wins": 0, "pnl": 0.0})
    _by_day:  Dict[str, Dict] = defaultdict(lambda: {"n": 0, "wins": 0, "pnl": 0.0})

    for t in closed:
        pnl = t.get("pnl") or 0.0
        won = int(pnl > 0)

        h = t.get("hour_cet")
        if h is not None:
            _by_hour[h]["n"] += 1
            _by_hour[h]["wins"] += won
            _by_hour[h]["pnl"] += pnl

        sess = t.get("session") or "Autre"
        _by_sess[sess]["n"] += 1
        _by_sess[sess]["wins"] += won
        _by_sess[sess]["pnl"] += pnl

        d = t.get("direction") or "?"
        _by_dir[d]["n"] += 1
        _by_dir[d]["wins"] += won
        _by_dir[d]["pnl"] += pnl

        day = t.get("date_cet", "?")
        _by_day[day]["n"] += 1
        _by_day[day]["wins"] += won
        _by_day[day]["pnl"] += pnl

    def _agg(mapping: dict) -> dict:
        out = {}
        for k in sorted(mapping.keys(), key=str):
            v = mapping[k]
            out[str(k)] = {
                "n": v["n"],
                "wins": v["wins"],
                "wr": round(v["wins"] / v["n"] * 100, 1) if v["n"] > 0 else 0.0,
                "pnl": round(v["pnl"], 2),
            }
        return out

    by_hour = _agg(_by_hour)
    by_session = _agg(_by_sess)
    by_direction = _agg(_by_dir)
    by_day_full = _agg(_by_day)
    # Garder seulement les 30 derniers jours
    by_day = dict(list(by_day_full.items())[-30:])

    lines = [
        f"RAPPORT HISTORIQUE TRADES – {total} trades clôturés",
        f"WR global: {stats['win_rate']}% | PF: {stats['profit_factor']} | PnL total: {stats['total_pnl']}$",
        f"Gain moyen: +{stats['avg_win']}$ | Perte moyenne: {stats['avg_loss']}$",
        "",
        "WR PAR HEURE CET:",
    ]
    for h, v in by_hour.items():
        hi = int(h)
        tag = "London" if 8 <= hi < 12 else ("NY" if 14 <= hi < 18 else "hors-session")
        lines.append(f"  {h}h ({tag}): WR={v['wr']}% n={v['n']} pnl={v['pnl']}$")
    lines += ["", "WR PAR SESSION:"]
    for s, v in by_session.items():
        lines.append(f"  {s}: WR={v['wr']}% n={v['n']} pnl={v['pnl']}$")
    lines += ["", "WR PAR DIRECTION:"]
    for d, v in by_direction.items():
        lines.append(f"  {d.upper()}: WR={v['wr']}% n={v['n']} pnl={v['pnl']}$")
    lines += ["", "30 DERNIERS JOURS:"]
    for day, v in by_day.items():
        lines.append(f"  {day}: WR={v['wr']}% n={v['n']} pnl={v['pnl']}$")

    return {
        "stats": stats,
        "by_hour": by_hour,
        "by_session": by_session,
        "by_direction": by_direction,
        "by_day": by_day,
        "trades": trades_list,
        "llm_summary": "\n".join(lines),
    }


def get_weekly_report(week_offset: int = 0, symbol: str | None = None) -> Dict[str, Any]:
    """
    Rapport de la semaine courante (ou N semaines en arrière si week_offset < 0).
    Retourne stats WR/PF/PnL + breakdown par jour, session, direction, exit_reason.
    """
    from collections import defaultdict

    try:
        import pytz as _pytz
        _CET = _pytz.timezone("Europe/Paris")
    except Exception:
        _CET = None

    # Calcul lundi/dimanche de la semaine cible (en UTC)
    now_utc = datetime.now(timezone.utc)
    days_since_monday = now_utc.weekday()
    monday = (now_utc - timedelta(days=days_since_monday)).replace(
        hour=0, minute=0, second=0, microsecond=0
    ) + timedelta(weeks=week_offset)
    sunday = monday + timedelta(days=6, hours=23, minutes=59, seconds=59)

    monday_str = monday.strftime("%Y-%m-%d")
    sunday_str = sunday.strftime("%Y-%m-%d")
    week_label = f"{monday.strftime('%d/%m')} – {sunday.strftime('%d/%m/%Y')}"

    with get_conn() as conn:
        if symbol:
            rows = conn.execute(
                "SELECT * FROM trades WHERE status = 'closed' AND symbol = ?"
                " AND substr(entry_time, 1, 10) >= ? AND substr(entry_time, 1, 10) <= ?"
                " ORDER BY entry_time ASC",
                (symbol, monday_str, sunday_str),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM trades WHERE status = 'closed'"
                " AND substr(entry_time, 1, 10) >= ? AND substr(entry_time, 1, 10) <= ?"
                " ORDER BY entry_time ASC",
                (monday_str, sunday_str),
            ).fetchall()

    trades = [_row_to_dict(r) for r in rows]

    for t in trades:
        try:
            ts = datetime.fromisoformat(t["entry_time"].replace("Z", "+00:00"))
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)
            ts_cet = ts.astimezone(_CET) if _CET else ts
            t["date_cet"] = ts_cet.strftime("%Y-%m-%d")
            t["weekday"] = ts_cet.strftime("%A")  # Monday, Tuesday…
            t["hour_cet"] = ts_cet.hour
        except Exception:
            t["date_cet"] = "?"
            t["weekday"] = "?"
            t["hour_cet"] = None

    closed = [t for t in trades if t.get("pnl") is not None]
    total = len(closed)
    wins = [t for t in closed if (t["pnl"] or 0) > 0]
    losses = [t for t in closed if (t["pnl"] or 0) <= 0]

    gross_profit = sum(t["pnl"] for t in wins)
    gross_loss = abs(sum(t["pnl"] for t in losses)) if losses else 0.0
    total_pnl = sum(t["pnl"] for t in closed)

    # Breakdown exit_reason
    exit_counts: Dict[str, int] = defaultdict(int)
    for t in closed:
        reason = t.get("exit_reason") or "?"
        exit_counts[reason] += 1

    sl_direct = exit_counts.get("sl", 0)
    tp2_count = exit_counts.get("tp2", 0)
    tp1_count = exit_counts.get("tp1", 0)
    timeout_count = exit_counts.get("timeout", 0)

    # Breakdown par jour de semaine
    _by_day: Dict[str, Dict] = defaultdict(lambda: {"n": 0, "wins": 0, "pnl": 0.0})
    _by_sess: Dict[str, Dict] = defaultdict(lambda: {"n": 0, "wins": 0, "pnl": 0.0})
    _by_dir: Dict[str, Dict] = defaultdict(lambda: {"n": 0, "wins": 0, "pnl": 0.0})

    for t in closed:
        pnl = t.get("pnl") or 0.0
        won = int(pnl > 0)

        day_key = t.get("date_cet", "?")
        _by_day[day_key]["n"] += 1
        _by_day[day_key]["wins"] += won
        _by_day[day_key]["pnl"] += pnl

        sess = t.get("session") or "Autre"
        _by_sess[sess]["n"] += 1
        _by_sess[sess]["wins"] += won
        _by_sess[sess]["pnl"] += pnl

        d = (t.get("direction") or "?").upper()
        _by_dir[d]["n"] += 1
        _by_dir[d]["wins"] += won
        _by_dir[d]["pnl"] += pnl

    def _agg(mapping: dict) -> dict:
        out = {}
        for k in sorted(mapping.keys(), key=str):
            v = mapping[k]
            out[str(k)] = {
                "n": v["n"],
                "wins": v["wins"],
                "wr": round(v["wins"] / v["n"] * 100, 1) if v["n"] > 0 else 0.0,
                "pnl": round(v["pnl"], 2),
            }
        return out

    return {
        "week_label": week_label,
        "week_offset": week_offset,
        "stats": {
            "total": total,
            "wins": len(wins),
            "losses": len(losses),
            "win_rate": round(len(wins) / total * 100, 1) if total > 0 else 0.0,
            "profit_factor": round(gross_profit / gross_loss, 2) if gross_loss > 0 else 0.0,
            "total_pnl": round(total_pnl, 2),
            "avg_win": round(gross_profit / len(wins), 2) if wins else 0.0,
            "avg_loss": round(-gross_loss / len(losses), 2) if losses else 0.0,
        },
        "exit_reasons": {
            "sl_direct": sl_direct,
            "tp1_only": tp1_count,
            "tp2": tp2_count,
            "timeout": timeout_count,
            "sl_direct_pct": round(sl_direct / total * 100, 1) if total > 0 else 0.0,
            "tp2_pct": round(tp2_count / total * 100, 1) if total > 0 else 0.0,
        },
        "by_day": _agg(_by_day),
        "by_session": _agg(_by_sess),
        "by_direction": _agg(_by_dir),
    }


def get_monthly_report(month_offset: int = 0, symbol: str | None = None) -> Dict[str, Any]:
    """
    Rapport mensuel. month_offset=0 = mois courant, -1 = mois précédent, etc.
    """
    from collections import defaultdict

    try:
        import pytz as _pytz
        _CET = _pytz.timezone("Europe/Paris")
    except Exception:
        _CET = None

    now_utc = datetime.now(timezone.utc)
    year = now_utc.year
    month = now_utc.month + month_offset
    while month <= 0:
        month += 12
        year -= 1
    while month > 12:
        month -= 12
        year += 1

    import calendar
    first_day = datetime(year, month, 1, 0, 0, 0, tzinfo=timezone.utc)
    last_day_num = calendar.monthrange(year, month)[1]
    last_day = datetime(year, month, last_day_num, 23, 59, 59, tzinfo=timezone.utc)
    first_str = first_day.strftime("%Y-%m-%d")
    last_str = last_day.strftime("%Y-%m-%d")
    month_label = first_day.strftime("%B %Y")

    with get_conn() as conn:
        if symbol:
            rows = conn.execute(
                "SELECT * FROM trades WHERE status = 'closed' AND symbol = ?"
                " AND substr(entry_time, 1, 10) >= ? AND substr(entry_time, 1, 10) <= ?"
                " ORDER BY entry_time ASC",
                (symbol, first_str, last_str),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM trades WHERE status = 'closed'"
                " AND substr(entry_time, 1, 10) >= ? AND substr(entry_time, 1, 10) <= ?"
                " ORDER BY entry_time ASC",
                (first_str, last_str),
            ).fetchall()

    trades = [_row_to_dict(r) for r in rows]

    for t in trades:
        try:
            ts = datetime.fromisoformat(t["entry_time"].replace("Z", "+00:00"))
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)
            ts_cet = ts.astimezone(_CET) if _CET else ts
            t["date_cet"] = ts_cet.strftime("%Y-%m-%d")
            t["hour_cet"] = ts_cet.hour
            # Semaine ISO du mois (1-5)
            t["week_of_month"] = f"Sem {((ts_cet.day - 1) // 7) + 1}"
        except Exception:
            t["date_cet"] = "?"
            t["hour_cet"] = None
            t["week_of_month"] = "?"

    closed = [t for t in trades if t.get("pnl") is not None]
    total = len(closed)
    wins = [t for t in closed if (t["pnl"] or 0) > 0]
    losses = [t for t in closed if (t["pnl"] or 0) <= 0]

    gross_profit = sum(t["pnl"] for t in wins)
    gross_loss = abs(sum(t["pnl"] for t in losses)) if losses else 0.0
    total_pnl = sum(t["pnl"] for t in closed)

    exit_counts: Dict[str, int] = defaultdict(int)
    for t in closed:
        exit_counts[t.get("exit_reason") or "?"] += 1

    sl_direct = exit_counts.get("sl", 0)
    tp2_count = exit_counts.get("tp2", 0)
    tp1_count = exit_counts.get("tp1", 0)
    timeout_count = exit_counts.get("timeout", 0)

    _by_week: Dict[str, Dict] = defaultdict(lambda: {"n": 0, "wins": 0, "pnl": 0.0})
    _by_sess: Dict[str, Dict] = defaultdict(lambda: {"n": 0, "wins": 0, "pnl": 0.0})
    _by_dir: Dict[str, Dict] = defaultdict(lambda: {"n": 0, "wins": 0, "pnl": 0.0})
    _by_day: Dict[str, Dict] = defaultdict(lambda: {"n": 0, "wins": 0, "pnl": 0.0})

    for t in closed:
        pnl = t.get("pnl") or 0.0
        won = int(pnl > 0)

        wk = t.get("week_of_month", "?")
        _by_week[wk]["n"] += 1
        _by_week[wk]["wins"] += won
        _by_week[wk]["pnl"] += pnl

        sess = t.get("session") or "Autre"
        _by_sess[sess]["n"] += 1
        _by_sess[sess]["wins"] += won
        _by_sess[sess]["pnl"] += pnl

        d = (t.get("direction") or "?").upper()
        _by_dir[d]["n"] += 1
        _by_dir[d]["wins"] += won
        _by_dir[d]["pnl"] += pnl

        day = t.get("date_cet", "?")
        _by_day[day]["n"] += 1
        _by_day[day]["wins"] += won
        _by_day[day]["pnl"] += pnl

    def _agg(mapping: dict) -> dict:
        out = {}
        for k in sorted(mapping.keys(), key=str):
            v = mapping[k]
            out[str(k)] = {
                "n": v["n"],
                "wins": v["wins"],
                "wr": round(v["wins"] / v["n"] * 100, 1) if v["n"] > 0 else 0.0,
                "pnl": round(v["pnl"], 2),
            }
        return out

    return {
        "month_label": month_label,
        "month_offset": month_offset,
        "stats": {
            "total": total,
            "wins": len(wins),
            "losses": len(losses),
            "win_rate": round(len(wins) / total * 100, 1) if total > 0 else 0.0,
            "profit_factor": round(gross_profit / gross_loss, 2) if gross_loss > 0 else 0.0,
            "total_pnl": round(total_pnl, 2),
            "avg_win": round(gross_profit / len(wins), 2) if wins else 0.0,
            "avg_loss": round(-gross_loss / len(losses), 2) if losses else 0.0,
        },
        "exit_reasons": {
            "sl_direct": sl_direct,
            "tp1_only": tp1_count,
            "tp2": tp2_count,
            "timeout": timeout_count,
            "sl_direct_pct": round(sl_direct / total * 100, 1) if total > 0 else 0.0,
            "tp2_pct": round(tp2_count / total * 100, 1) if total > 0 else 0.0,
        },
        "by_week": _agg(_by_week),
        "by_session": _agg(_by_sess),
        "by_direction": _agg(_by_dir),
        "by_day": _agg(_by_day),
        "trades": closed,
    }


if __name__ == "__main__":
    init_db()
    print(f"Database initialised at {DB_PATH}")
    print("Settings:", json.dumps(get_settings(), indent=2))
