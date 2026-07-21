"""Tests for the trade-volume diagnostics query in database.py."""
from datetime import datetime, timedelta, timezone

import pytest

import database as db


@pytest.fixture(autouse=True)
def _fresh_db(tmp_path, monkeypatch):
    """Point the DB at a throwaway file so tests don't see each other's trades."""
    monkeypatch.setattr(db, "DB_PATH", str(tmp_path / "test.db"))
    db.init_db()
    yield


def _insert_closed_trade(entry_time: datetime, symbol: str = "XAUUSD", pnl: float = 10.0):
    db.insert_trade({
        "symbol": symbol,
        "direction": "long",
        "session": "London",
        "entry_time": entry_time.isoformat(),
        "exit_time": (entry_time + timedelta(minutes=20)).isoformat(),
        "entry_price": 2000.0,
        "exit_price": 2005.0,
        "stop_loss": 1995.0,
        "take_profit1": 2003.5,
        "take_profit2": 2009.0,
        "volume": 0.1,
        "risk_amount": 50.0,
        "pnl": pnl,
        "pnl_pct": 1.0,
        "duration_min": 20.0,
        "status": "closed",
        "exit_reason": "tp2",
        "mode": "paper",
        "meta": None,
    })


def _this_monday_utc() -> datetime:
    now = datetime.now(timezone.utc)
    return (now - timedelta(days=now.weekday())).replace(hour=9, minute=0, second=0, microsecond=0)


def test_by_weekday_always_includes_monday_to_friday_even_with_zero_trades():
    monday = _this_monday_utc()
    _insert_closed_trade(monday)  # only Monday has a trade

    report = db.get_trade_volume_report(symbol="XAUUSD", weeks=4)

    assert set(report["by_weekday"].keys()) == {
        "Lundi", "Mardi", "Mercredi", "Jeudi", "Vendredi",
    }
    assert report["by_weekday"]["Lundi"]["n"] == 1
    assert report["by_weekday"]["Mardi"]["n"] == 0


def test_by_week_surfaces_zero_trade_weeks_as_gaps():
    monday = _this_monday_utc()
    _insert_closed_trade(monday)  # trade only in the current week
    # Skip the previous week entirely -> should still appear with n=0

    report = db.get_trade_volume_report(symbol="XAUUSD", weeks=4)

    assert report["weeks_covered"] == 4
    assert len(report["by_week"]) == 4
    assert report["weeks_with_zero_trades"] == 3
    counts = [v["n"] for v in report["by_week"].values()]
    assert counts[-1] == 1  # most recent week (current) has the trade
    assert sum(counts) == 1


def test_symbol_filter_isolates_trades():
    monday = _this_monday_utc()
    _insert_closed_trade(monday, symbol="XAUUSD")
    _insert_closed_trade(monday, symbol="EURUSD")

    xau_report = db.get_trade_volume_report(symbol="XAUUSD", weeks=2)
    eur_report = db.get_trade_volume_report(symbol="EURUSD", weeks=2)

    assert xau_report["total_trades"] == 1
    assert eur_report["total_trades"] == 1


def test_avg_and_extremes_across_weeks():
    monday = _this_monday_utc()
    last_monday = monday - timedelta(weeks=1)
    _insert_closed_trade(monday)
    _insert_closed_trade(monday + timedelta(days=1))
    _insert_closed_trade(last_monday)

    report = db.get_trade_volume_report(symbol="XAUUSD", weeks=2)

    assert report["total_trades"] == 3
    assert report["max_trades_in_a_week"] == 2
    assert report["min_trades_in_a_week"] == 1
    assert report["avg_trades_per_week"] == pytest.approx(1.5)
