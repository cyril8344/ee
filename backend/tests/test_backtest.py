"""End-to-end backtest tests (synthetic offline data)."""
from datetime import datetime, timedelta

import pytest

from backtest import BacktestConfig, run_backtest


def _recent_cfg(days=45):
    end = datetime.utcnow()
    start = end - timedelta(days=days)
    return BacktestConfig(start=start.strftime("%Y-%m-%d"),
                          end=end.strftime("%Y-%m-%d"))


def test_backtest_produces_full_report():
    rep = run_backtest(_recent_cfg(45))
    assert "summary" in rep, rep.get("error")
    s = rep["summary"]
    for key in ("trades", "winrate", "profit_factor", "net_profit",
                "max_drawdown_usd", "max_drawdown_pct", "expectancy"):
        assert key in s
    assert "equity_curve" in rep and len(rep["equity_curve"]) >= 1
    assert "by_session" in rep
    assert "heatmap" in rep


def test_backtest_winrate_in_range():
    rep = run_backtest(_recent_cfg(45))
    wr = rep["summary"]["winrate"]
    assert 0.0 <= wr <= 100.0


def test_backtest_equity_curve_is_consistent():
    rep = run_backtest(_recent_cfg(45))
    s = rep["summary"]
    final = rep["equity_curve"][-1]["equity"]
    assert final == pytest.approx(s["final_equity"], abs=0.01)


def test_backtest_respects_max_trades_per_day():
    cfg = _recent_cfg(20)
    cfg.max_trades_per_day = 2
    rep = run_backtest(cfg)
    # group trades by entry day and assert cap
    by_day = {}
    for t in rep.get("trades", []):
        day = t["entry_time"][:10]
        by_day[day] = by_day.get(day, 0) + 1
    assert all(c <= 2 for c in by_day.values())
