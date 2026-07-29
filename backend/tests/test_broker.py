"""Tests for PaperBroker.close_position — see the BE_BUFFER_R / stale-price fix."""
from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest

from broker import PaperBroker, Position


def _make_position(direction="short", entry=4044.34, stop_loss=4048.0,
                    tp1=4040.0, tp2=4030.0, volume=0.5):
    return Position(
        ticket=1, direction=direction, entry=entry, volume=volume,
        stop_loss=stop_loss, take_profit1=tp1, take_profit2=tp2,
        open_time=datetime.now(timezone.utc), remaining=volume,
    )


def test_close_position_uses_given_price_not_stale_cache():
    """A real-time TP2/SL trigger must fill at the level that triggered it, not at
    whatever self.get_price() (cached M5 close, up to 300s stale) happens to return."""
    broker = PaperBroker(spread_pips=0.0, slippage_pips=0.0, symbol="XAUUSD")

    def _boom():
        raise AssertionError("get_price() should not be called when price= is given")
    broker.get_price = _boom

    pos = _make_position()
    info = broker.close_position(pos, "tp2_realtime", price=pos.take_profit2)
    assert info["closed"] is True
    assert info["exit_price"] == pytest.approx(pos.take_profit2)


def test_close_position_falls_back_to_get_price_when_no_price_given():
    """Manual / timeout closes have no specific target — current market price is correct."""
    broker = PaperBroker(spread_pips=0.0, slippage_pips=0.0, symbol="XAUUSD")
    broker.get_price = lambda: 4041.0

    pos = _make_position()
    info = broker.close_position(pos, "manual")
    assert info["exit_price"] == pytest.approx(4041.0)


def _bar(ts, o, h, l, c):
    return pd.DataFrame(
        {"open": [o], "high": [h], "low": [l], "close": [c], "volume": [1.0]},
        index=pd.DatetimeIndex([ts], name="time"),
    )


def test_update_position_does_not_recheck_be_sl_on_same_tp1_bar():
    """Regression: the candle that triggers TP1 can open above the entry (future BE
    level) then sell off through TP1 within the same 5min bar. Re-polling that same
    cached bar must NOT be interpreted as price coming back up to hit the breakeven
    stop — only a genuinely later bar may do that (mirrors backtest.py::_try_exit,
    which already guards this via `return None` on the TP1 bar)."""
    broker = PaperBroker(spread_pips=0.0, slippage_pips=0.0, symbol="XAUUSD")
    pos = _make_position(direction="short", entry=4036.77, stop_loss=4039.46,
                          tp1=4034.89, tp2=4031.0, volume=0.2)

    t1 = datetime(2026, 7, 29, 9, 0, tzinfo=timezone.utc)
    # Candle opens above entry (>= future BE stop) then sells off hard through TP1.
    broker.data.get_m5 = lambda bars=2: _bar(t1, o=4037.50, h=4037.60, l=4034.50, c=4035.00)

    info = broker.update_position(pos)
    assert info["reason"] == "tp1_partial"
    assert pos.tp1_done is True
    assert pos.stop_loss == pytest.approx(4036.77)  # BE_BUFFER_R = 0.0 par défaut

    # Same bar polled again (still cached) — must NOT trigger sl_after_tp1 even
    # though this bar's high (4037.60) is above the new breakeven stop.
    info2 = broker.update_position(pos)
    assert info2 is None

    # A genuinely later bar that stays below breakeven — still no trigger.
    t2 = t1 + timedelta(minutes=5)
    broker.data.get_m5 = lambda bars=2: _bar(t2, o=4035.00, h=4035.80, l=4034.00, c=4034.50)
    assert broker.update_position(pos) is None

    # A later bar that actually reaches breakeven — now it must trigger.
    t3 = t2 + timedelta(minutes=5)
    broker.data.get_m5 = lambda bars=2: _bar(t3, o=4034.50, h=4036.90, l=4034.20, c=4036.00)
    info3 = broker.update_position(pos)
    assert info3["closed"] is True
    assert info3["reason"] == "sl_after_tp1"
    assert info3["exit_price"] == pytest.approx(4036.77)
