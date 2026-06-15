"""Tests for the strict risk manager."""
import pytest

from risk_manager import RiskManager, CONTRACT_SIZE, MIN_LOT


def test_position_sizing_risks_one_percent():
    rm = RiskManager(capital=10000, risk_per_trade_pct=1.0)
    # entry 2000, stop 1995 -> 5.0 distance; loss/lot = 5*100 = 500
    # 1% of 10000 = 100 -> 0.2 lots
    vol, risk, dist = rm.compute_position(2000.0, 1995.0)
    assert dist == pytest.approx(5.0)
    assert vol == pytest.approx(0.2)
    assert risk == pytest.approx(vol * dist * CONTRACT_SIZE)
    assert risk == pytest.approx(100.0, abs=1.0)


def test_zero_stop_distance_rejected():
    rm = RiskManager(capital=10000)
    dec = rm.can_open_trade(2000.0, 2000.0)
    assert dec.allowed is False


def test_max_trades_per_day():
    rm = RiskManager(capital=10000, max_trades_per_day=4)
    for _ in range(4):
        assert rm.can_open_trade(2000.0, 1995.0).allowed is True
        rm.register_open()
    assert rm.can_open_trade(2000.0, 1995.0).allowed is False
    assert "Max trades" in rm.can_open_trade(2000.0, 1995.0).reason


def test_daily_stop_blocks_bot():
    rm = RiskManager(capital=10000, daily_stop_pct=2.0)
    rm.start_new_day(10000)
    # lose 2% -> -200 should block
    rm.register_close(-200.0)
    assert rm.blocked is True
    assert rm.can_open_trade(2000.0, 1995.0).allowed is False


def test_daily_stop_not_triggered_below_limit():
    rm = RiskManager(capital=10000, daily_stop_pct=2.0)
    rm.start_new_day(10000)
    rm.register_close(-150.0)  # -1.5%, under the -2% limit
    assert rm.blocked is False
    assert rm.can_open_trade(2000.0, 1995.0).allowed is True


def test_risk_pct_change_requires_confirmation():
    rm = RiskManager(capital=10000, risk_per_trade_pct=1.0)
    assert rm.set_risk_pct(3.0, confirmed=False) is False
    assert rm.risk_per_trade_pct == 1.0
    assert rm.set_risk_pct(2.0, confirmed=True) is True
    assert rm.risk_per_trade_pct == 2.0


def test_register_close_updates_capital():
    rm = RiskManager(capital=10000)
    rm.start_new_day(10000)
    rm.register_close(250.0)
    assert rm.capital == pytest.approx(10250.0)
    assert rm.realised_pnl_today == pytest.approx(250.0)
