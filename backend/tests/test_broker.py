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


# --------------------------------------------------------------------------- #
# MarketData : data_symbol / price_scale (proxy ES via SPY, voir pretrain_es.py)
# --------------------------------------------------------------------------- #
def test_market_data_fetches_data_symbol_not_internal_symbol(monkeypatch):
    from broker import MarketData
    seen = {}

    def _fake_get_m5(bars, symbol):
        seen["symbol"] = symbol
        return _bar(datetime(2026, 1, 1, tzinfo=timezone.utc), 580, 582, 579, 581), "twelvedata"

    import data_provider
    monkeypatch.setattr(data_provider, "get_m5", _fake_get_m5)

    md = MarketData(symbol="ES", data_symbol="SPY", price_scale=10.0)
    md.get_m5(2)
    assert seen["symbol"] == "SPY"


def test_market_data_scales_prices_but_not_volume(monkeypatch):
    from broker import MarketData

    def _fake_get_m5(bars, symbol):
        df = _bar(datetime(2026, 1, 1, tzinfo=timezone.utc), 580.0, 582.0, 579.0, 581.0)
        df["volume"] = 100000
        return df, "twelvedata"

    import data_provider
    monkeypatch.setattr(data_provider, "get_m5", _fake_get_m5)

    md = MarketData(symbol="ES", data_symbol="SPY", price_scale=10.0)
    df = md.get_m5(2)
    row = df.iloc[-1]
    assert row["open"] == pytest.approx(5800.0)
    assert row["high"] == pytest.approx(5820.0)
    assert row["low"] == pytest.approx(5790.0)
    assert row["close"] == pytest.approx(5810.0)
    assert row["volume"] == pytest.approx(100000)


def test_market_data_default_scale_is_a_no_op_for_existing_symbols(monkeypatch):
    """Regression guard: XAU/EUR must be completely unaffected by this feature."""
    from broker import MarketData

    def _fake_get_m5(bars, symbol):
        return _bar(datetime(2026, 1, 1, tzinfo=timezone.utc), 2000, 2001, 1999, 2000.5), "twelvedata"

    import data_provider
    monkeypatch.setattr(data_provider, "get_m5", _fake_get_m5)

    md = MarketData(symbol="XAUUSD")
    df = md.get_m5(2)
    assert df.iloc[-1]["close"] == pytest.approx(2000.5)
    assert md.data_symbol == "XAUUSD"


# --------------------------------------------------------------------------- #
# ESPaperBroker : logique de sortie fidèle à pretrain_es.py::_try_exit_es
# --------------------------------------------------------------------------- #
def _make_es_position(direction="long", entry=5800.0, stop_loss=5790.0,
                       tp1=5807.0, tp2=5814.0, volume=5):
    return Position(
        ticket=1, direction=direction, entry=entry, volume=volume,
        stop_loss=stop_loss, take_profit1=tp1, take_profit2=tp2,
        open_time=datetime.now(timezone.utc), remaining=volume,
    )


def test_es_broker_tp1_moves_sl_to_breakeven_unconditionally():
    from broker import ESPaperBroker
    broker = ESPaperBroker(spread_pips=0.0, slippage_pips=0.0, symbol="ES",
                           contract_size=50.0, pip_size=0.25)
    pos = _make_es_position()

    t1 = datetime(2026, 1, 1, 15, 0, tzinfo=timezone.utc)
    broker.data.get_m5 = lambda bars=2: _bar(t1, o=5802, h=5808, l=5801, c=5807)

    info = broker.update_position(pos)
    assert info["reason"] == "tp1_partial"
    assert pos.tp1_done is True
    assert pos.stop_loss == pytest.approx(5800.0)  # BE inconditionnel, pile l'entrée


def test_es_broker_splits_contracts_as_whole_numbers():
    from broker import ESPaperBroker
    broker = ESPaperBroker(spread_pips=0.0, slippage_pips=0.0, symbol="ES",
                           contract_size=50.0, pip_size=0.25)
    pos = _make_es_position(volume=5)  # 5 contrats, impair

    t1 = datetime(2026, 1, 1, 15, 0, tzinfo=timezone.utc)
    broker.data.get_m5 = lambda bars=2: _bar(t1, o=5802, h=5808, l=5801, c=5807)
    broker.update_position(pos)

    assert pos.remaining == pytest.approx(round(5 - round(5 * 0.5)))
    assert float(pos.remaining).is_integer()


def test_es_broker_sl_after_tp1_closes_at_breakeven_not_a_loss():
    from broker import ESPaperBroker
    broker = ESPaperBroker(spread_pips=0.0, slippage_pips=0.0, symbol="ES",
                           contract_size=50.0, pip_size=0.25)
    pos = _make_es_position()

    t1 = datetime(2026, 1, 1, 15, 0, tzinfo=timezone.utc)
    broker.data.get_m5 = lambda bars=2: _bar(t1, o=5802, h=5808, l=5801, c=5807)
    broker.update_position(pos)

    t2 = t1 + timedelta(minutes=5)
    broker.data.get_m5 = lambda bars=2: _bar(t2, o=5807, h=5807.5, l=5799, c=5800)
    info = broker.update_position(pos)
    assert info["closed"] is True
    assert info["reason"] == "sl_after_tp1"
    assert info["exit_price"] == pytest.approx(5800.0)  # = entry, pas de perte sur le reliquat


def test_es_broker_ignores_same_bar_as_tp1_for_sl_check():
    """Même correctif que le broker générique (cf. test SL-après-TP1 XAU) : une
    bougie qui ouvre au-dessus de l'entrée puis chute d'un trait à travers TP1
    ne doit pas déclencher un faux SL-après-TP1 sur cette même bougie."""
    from broker import ESPaperBroker
    broker = ESPaperBroker(spread_pips=0.0, slippage_pips=0.0, symbol="ES",
                           contract_size=50.0, pip_size=0.25)
    pos = _make_es_position(direction="short", entry=5800.0, stop_loss=5810.0,
                            tp1=5793.0, tp2=5786.0, volume=4)

    t1 = datetime(2026, 1, 1, 15, 0, tzinfo=timezone.utc)
    # Bougie ouvre au-dessus de l'entrée (>= futur BE) puis chute à travers TP1.
    broker.data.get_m5 = lambda bars=2: _bar(t1, o=5801, h=5801.5, l=5791, c=5793)
    info = broker.update_position(pos)
    assert info["reason"] == "tp1_partial"
    assert pos.stop_loss == pytest.approx(5800.0)

    # Même bougie repolled (cache) — ne doit pas déclencher malgré high >= BE.
    assert broker.update_position(pos) is None


def test_es_broker_tp2_closes_remaining_contracts():
    from broker import ESPaperBroker
    broker = ESPaperBroker(spread_pips=0.0, slippage_pips=0.0, symbol="ES",
                           contract_size=50.0, pip_size=0.25)
    pos = _make_es_position()

    t1 = datetime(2026, 1, 1, 15, 0, tzinfo=timezone.utc)
    broker.data.get_m5 = lambda bars=2: _bar(t1, o=5802, h=5808, l=5801, c=5807)
    broker.update_position(pos)

    t2 = t1 + timedelta(minutes=5)
    broker.data.get_m5 = lambda bars=2: _bar(t2, o=5808, h=5815, l=5807, c=5814)
    info = broker.update_position(pos)
    assert info["closed"] is True
    assert info["reason"] == "tp2"
    assert info["exit_price"] == pytest.approx(5814.0)
