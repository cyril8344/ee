"""Tests for wf_monitor_runs persistence (Walk-Forward auto surveillance history).

Uses a unique symbol name per test (uuid suffix) so repeated runs against the
shared on-disk test DB (see conftest.py — fixed tempfile path, not wiped
between runs) never collide with rows from a previous invocation.
"""
import uuid

import database as db


def _unique_symbol(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:8]}"


def test_wf_monitor_log_and_read_history():
    db.init_db()
    symbol = _unique_symbol("TESTSYM_WFHIST")
    db.wf_monitor_log_run(symbol, "2024-01-01 → 2026-01-01", 1.25, 0.18, 80.0, True)
    db.wf_monitor_log_run(symbol, "2024-01-08 → 2026-01-08", 1.05, 0.45, 50.0, False)

    history = db.wf_monitor_history(symbol, limit=10)
    assert len(history) == 2
    # most recent first
    assert history[0]["avg_pf"] == 1.05
    assert history[0]["is_robust"] is False
    assert history[1]["avg_pf"] == 1.25
    assert history[1]["is_robust"] is True


def test_wf_monitor_history_respects_limit_and_symbol_scope():
    db.init_db()
    symbol = _unique_symbol("TESTSYM_WFHIST2")
    other_symbol = _unique_symbol("SOME_OTHER_SYMBOL")
    for i in range(5):
        db.wf_monitor_log_run(symbol, f"period{i}", 1.0 + i * 0.1, 0.2, 60.0, True)
    other_symbol_history = db.wf_monitor_history(other_symbol, limit=10)
    assert other_symbol_history == []

    limited = db.wf_monitor_history(symbol, limit=2)
    assert len(limited) == 2
    assert limited[0]["period"] == "period4"
