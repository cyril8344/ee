"""Tests for PaperBroker.close_position — see the BE_BUFFER_R / stale-price fix."""
from datetime import datetime, timezone

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
