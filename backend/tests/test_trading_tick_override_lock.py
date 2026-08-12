"""Regression test: trading_tick() must never evaluate a live signal while a
pretrain/walk-forward/Optuna run (manual, or the weekly wf_monitor surveillance)
holds pretrain._STRATEGY_OVERRIDE_LOCK — that lock guards a window where
strategy.py/strategy_ict.py module attributes are temporarily setattr'd to test
values. Without this guard, a real (paper or live) trade could be opened or
managed using test thresholds instead of the live ones, since trading_tick()
and pretrain's background thread run concurrently in the same process and
share the same module-level globals.
"""
import os

os.environ.setdefault("XAU_DATA_PROVIDER", "synthetic")

import main as main_module


def test_trading_tick_skips_evaluation_when_override_lock_held(monkeypatch):
    called = {"n": 0}
    monkeypatch.setattr(main_module, "_trading_tick_locked",
                         lambda: called.__setitem__("n", called["n"] + 1) or {"evaluated": True})

    main_module._pretrain_module._STRATEGY_OVERRIDE_LOCK.acquire()
    try:
        result = main_module.trading_tick()
    finally:
        main_module._pretrain_module._STRATEGY_OVERRIDE_LOCK.release()

    assert called["n"] == 0
    assert isinstance(result, dict)
    assert "evaluated" not in result


def test_trading_tick_runs_normally_when_lock_free(monkeypatch):
    called = {"n": 0}
    monkeypatch.setattr(main_module, "_trading_tick_locked",
                         lambda: called.__setitem__("n", called["n"] + 1) or {"evaluated": True})

    result = main_module.trading_tick()

    assert called["n"] == 1
    assert result == {"evaluated": True}
