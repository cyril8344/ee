"""Regression test: get_weekly_report()/get_monthly_report() must account for
every real exit_reason value (sl, sl_realtime, sl_after_tp1, tp1, tp2, timeout,
early_exit, ...), not just the 4 originally hardcoded ("sl"/"tp2"/"tp1"/
"timeout") — otherwise most closed trades (typically "sl_after_tp1", the
breakeven exit after TP1, which is a very common outcome) silently vanish
from the "Sorties" breakdown without appearing in any bucket.
"""
import uuid
from datetime import datetime, timezone

import database as db


def _unique_symbol(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:8]}"


def _insert_closed_trade(symbol: str, exit_reason: str, pnl: float = 10.0):
    now = datetime.now(timezone.utc).isoformat()
    return db.insert_trade({
        "symbol": symbol, "direction": "long", "session": "London",
        "entry_time": now, "exit_time": now,
        "entry_price": 2000.0, "exit_price": 2001.0, "stop_loss": 1995.0,
        "take_profit1": 2001.4, "take_profit2": 2003.6,
        "volume": 0.1, "risk_amount": 10.0, "pnl": pnl, "pnl_pct": 0.5,
        "duration_min": 10.0, "status": "closed", "exit_reason": exit_reason,
        "mode": "paper", "meta": {},
    })


def test_weekly_report_accounts_for_every_exit_reason():
    db.init_db()
    symbol = _unique_symbol("TESTSYM_EXITS_WK")
    reasons = ["sl", "sl_realtime", "sl_after_tp1", "sl_after_tp1",
               "tp1", "tp2", "timeout", "early_exit", "manual"]
    for r in reasons:
        _insert_closed_trade(symbol, r)

    report = db.get_weekly_report(week_offset=0, symbol=symbol)
    ex = report["exit_reasons"]
    total = report["stats"]["total"]
    assert total == len(reasons)

    assert ex["sl_direct"] == 2       # "sl" + "sl_realtime"
    assert ex["sl_after_tp1"] == 2
    assert ex["tp1_only"] == 1
    assert ex["tp2"] == 1
    assert ex["timeout"] == 1
    assert ex["early_exit"] == 1
    assert ex["other"] == 1           # "manual"

    accounted = (ex["sl_direct"] + ex["sl_after_tp1"] + ex["tp1_only"]
                 + ex["tp2"] + ex["timeout"] + ex["early_exit"] + ex["other"])
    assert accounted == total


def test_monthly_report_accounts_for_every_exit_reason():
    db.init_db()
    symbol = _unique_symbol("TESTSYM_EXITS_MO")
    reasons = ["sl", "sl_realtime", "sl_after_tp1", "sl_after_tp1",
               "tp1", "tp2", "timeout", "early_exit", "manual"]
    for r in reasons:
        _insert_closed_trade(symbol, r)

    report = db.get_monthly_report(month_offset=0, symbol=symbol)
    ex = report["exit_reasons"]
    total = report["stats"]["total"]
    assert total == len(reasons)

    assert ex["sl_direct"] == 2
    assert ex["sl_after_tp1"] == 2
    assert ex["tp1_only"] == 1
    assert ex["tp2"] == 1
    assert ex["timeout"] == 1
    assert ex["early_exit"] == 1
    assert ex["other"] == 1

    accounted = (ex["sl_direct"] + ex["sl_after_tp1"] + ex["tp1_only"]
                 + ex["tp2"] + ex["timeout"] + ex["early_exit"] + ex["other"])
    assert accounted == total
