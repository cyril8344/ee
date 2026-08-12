"""Regression test for the retroactive audit added alongside the trading_tick()
concurrency fix (PR #336): find_trades_in_override_windows() must correctly flag
trades whose entry_time falls inside a logged strategy_override_windows window,
and must not flag trades outside any window or inside a still-open window.
"""
import uuid
from datetime import datetime, timedelta, timezone

import database as db


def _unique_symbol(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:8]}"


def _insert_trade_at(symbol: str, entry_time: str):
    return db.insert_trade({
        "symbol": symbol, "direction": "long", "session": "London",
        "entry_time": entry_time, "exit_time": entry_time,
        "entry_price": 2000.0, "exit_price": 2001.0, "stop_loss": 1995.0,
        "take_profit1": 2001.4, "take_profit2": 2003.6,
        "volume": 0.1, "risk_amount": 10.0, "pnl": 10.0, "pnl_pct": 0.5,
        "duration_min": 10.0, "status": "closed", "exit_reason": "tp1",
        "mode": "paper", "meta": {},
    })


def _make_window(symbol: str, started_at: str, ended_at: str | None) -> int:
    """Crée une fenêtre avec des horaires explicites (contourne les timestamps
    auto-now(), quasi simultanés dans un test, pour avoir une fenêtre large et
    déterministe à comparer)."""
    window_id = db.log_override_window_start(symbol)
    with db.get_conn() as conn:
        conn.execute(
            "UPDATE strategy_override_windows SET started_at = ?, ended_at = ? WHERE id = ?",
            (started_at, ended_at, window_id),
        )
    return window_id


def test_trade_inside_closed_window_is_flagged():
    db.init_db()
    symbol = _unique_symbol("TESTSYM_OVERRIDE")
    now = datetime.now(timezone.utc)
    started_at = (now - timedelta(seconds=10)).isoformat()
    ended_at = (now + timedelta(seconds=10)).isoformat()
    _make_window(symbol, started_at, ended_at)

    inside_time = now.isoformat()
    _insert_trade_at(symbol, inside_time)

    result = db.find_trades_in_override_windows(symbol=symbol)
    assert result["windows_logged"] >= 1
    at_risk_entries = [t["entry_time"] for t in result["trades_at_risk"]]
    assert inside_time in at_risk_entries


def test_trade_outside_any_window_is_not_flagged():
    db.init_db()
    symbol = _unique_symbol("TESTSYM_NOOVERRIDE")
    now = datetime.now(timezone.utc)
    started_at = (now - timedelta(seconds=10)).isoformat()
    ended_at = (now + timedelta(seconds=10)).isoformat()
    _make_window(symbol, started_at, ended_at)

    far_future_time = (now + timedelta(days=3650)).isoformat()
    _insert_trade_at(symbol, far_future_time)

    result = db.find_trades_in_override_windows(symbol=symbol)
    at_risk_entries = [t["entry_time"] for t in result["trades_at_risk"]]
    assert far_future_time not in at_risk_entries


def test_trade_during_still_open_window_is_not_flagged():
    """Une fenêtre avec ended_at=NULL (run en cours, ou crash sans libération propre)
    ne doit jamais être traitée comme une plage [start, end] fermée — pas de borne
    de fin fiable à comparer pour l'instant. Horodatage dans un futur lointain pour
    ne pas dépendre de l'absence de fenêtres fermées d'autres tests dans la même
    seconde (voir find_trades_in_override_windows : volontairement pas filtré par
    symbole de fenêtre, donc une fenêtre d'un autre test peut légitimement chevaucher
    "maintenant")."""
    db.init_db()
    symbol = _unique_symbol("TESTSYM_OPENWINDOW")
    far_future = datetime.now(timezone.utc) + timedelta(days=3650)
    db.log_override_window_start(symbol)  # jamais terminée (ended_at reste NULL)

    trade_time = far_future.isoformat()
    _insert_trade_at(symbol, trade_time)

    result = db.find_trades_in_override_windows(symbol=symbol)
    assert result["open_windows"] >= 1
    at_risk_entries = [t["entry_time"] for t in result["trades_at_risk"]]
    assert trade_time not in at_risk_entries
