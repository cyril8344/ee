"""Tests for database.get_closed_trades() — added alongside
diag_real_trades_by_day_volatility() to feed real (not pretrain-simulated) trade
history into diagnostics that need it."""
import uuid
from datetime import datetime, timezone

import database as db


def _unique_symbol(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:8]}"


def _insert_closed_trade(symbol: str, entry_time: str, exit_reason: str):
    return db.insert_trade({
        "symbol": symbol, "direction": "long", "session": "London",
        "entry_time": entry_time, "exit_time": entry_time,
        "entry_price": 2000.0, "exit_price": 2001.0, "stop_loss": 1995.0,
        "take_profit1": 2001.4, "take_profit2": 2003.6,
        "volume": 0.1, "risk_amount": 10.0, "pnl": 10.0, "pnl_pct": 0.5,
        "duration_min": 10.0, "status": "closed", "exit_reason": exit_reason,
        "mode": "paper", "meta": {},
    })


def test_get_closed_trades_returns_only_closed_for_symbol_with_date_cet():
    db.init_db()
    symbol = _unique_symbol("TESTSYM_CLOSED")
    entry_time = datetime(2024, 6, 3, 9, 6, tzinfo=timezone.utc).isoformat()
    _insert_closed_trade(symbol, entry_time, "early_exit")

    other_symbol = _unique_symbol("TESTSYM_OTHER")
    _insert_closed_trade(other_symbol, entry_time, "tp2")

    trades = db.get_closed_trades(symbol=symbol)
    assert len(trades) == 1
    assert trades[0]["exit_reason"] == "early_exit"
    assert trades[0]["date_cet"] != "?"
    assert trades[0]["entry_ts_utc"] is not None


def test_get_closed_trades_excludes_open_trades():
    db.init_db()
    symbol = _unique_symbol("TESTSYM_OPEN")
    trade_id = _insert_closed_trade(symbol, datetime.now(timezone.utc).isoformat(), "tp1")
    db.update_trade(trade_id, {"status": "open", "exit_reason": None})

    trades = db.get_closed_trades(symbol=symbol)
    assert trades == []
