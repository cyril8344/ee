"""Tests for the ATR-trailing alternative to the single BE jump after TP1
(strategy.TRAIL_AFTER_TP1_ENABLED / TRAIL_AFTER_TP1_ATR_MULT), added to compare
against the diagnostic showing a much lower TP1->TP2 conversion rate for LONG
(41.7%) than SHORT (60.8%) despite an almost identical TP1 hit rate — the SL
fixed exactly at breakeven may be absorbing normal post-TP1 noise indiscriminately.
Disabled by default: must not change existing behaviour unless explicitly enabled
via extra_overrides in a pretrain/walk-forward run.

TP2 is deliberately set far out of reach of the test bars below so each test
exercises only the trailing/SL interaction, not an incidental TP2 exit.
"""
import datetime

import pandas as pd

import strategy
from backtest import _try_exit


def _make_long_trade(entry=2000.0, sl=1995.0, tp1=2001.4, tp2=2100.0, risk=5.0):
    return {
        "direction": "long", "entry": entry, "stop_loss": sl, "tp1": tp1, "tp2": tp2,
        "volume": 0.1, "tp1_done": False, "be_after_tp1": True, "tp1_close_all": False,
        "remaining": 0.1, "realised": 0.0,
        "max_exit_time": datetime.datetime(2024, 1, 1, 12, 0),
        "entry_time": datetime.datetime(2024, 1, 1, 10, 0),
        "risk": risk, "mae": 0.0, "mfe": 0.0,
    }


def _make_short_trade(entry=2000.0, sl=2005.0, tp1=1998.6, tp2=1900.0, risk=5.0):
    return {
        "direction": "short", "entry": entry, "stop_loss": sl, "tp1": tp1, "tp2": tp2,
        "volume": 0.1, "tp1_done": False, "be_after_tp1": True, "tp1_close_all": False,
        "remaining": 0.1, "realised": 0.0,
        "max_exit_time": datetime.datetime(2024, 1, 1, 12, 0),
        "entry_time": datetime.datetime(2024, 1, 1, 10, 0),
        "risk": risk, "mae": 0.0, "mfe": 0.0,
    }


def _bar(high, low, close, atr=5.0):
    return {"high": high, "low": low, "close": close, "atr": atr}


_TS = pd.Timestamp(2024, 1, 1, 10, 5)


def setup_function(_):
    strategy.TRAIL_AFTER_TP1_ENABLED = False
    strategy.TRAIL_AFTER_TP1_ATR_MULT = 1.0


def teardown_function(_):
    strategy.TRAIL_AFTER_TP1_ENABLED = False
    strategy.TRAIL_AFTER_TP1_ATR_MULT = 1.0


def test_trailing_disabled_by_default_sl_stays_at_be_after_tp1():
    t = _make_long_trade()
    # bougie qui touche TP1 -> saut unique SL a BE, return None
    assert _try_exit(t, _bar(2001.5, 2000.5, 2001.0), _TS, 0.0, 1.0) is None
    assert t["stop_loss"] == t["entry"]

    # bougie suivante qui pousse tres haut : sans trailing, le SL ne doit PAS bouger
    exit_info = _try_exit(t, _bar(2010.0, 2009.0, 2009.5), _TS, 0.0, 1.0)
    assert exit_info is None
    assert t["stop_loss"] == t["entry"]


def test_trailing_ratchets_sl_up_for_long_after_tp1():
    strategy.TRAIL_AFTER_TP1_ENABLED = True
    strategy.TRAIL_AFTER_TP1_ATR_MULT = 1.0
    t = _make_long_trade()
    assert _try_exit(t, _bar(2001.5, 2000.5, 2001.0, atr=5.0), _TS, 0.0, 1.0) is None
    assert t["stop_loss"] == t["entry"]  # plancher BE juste apres TP1

    # high=2008, ATR=5 -> candidate = 2008 - 5 = 2003, au-dessus de BE ; low=2006
    # reste au-dessus du nouveau stop -> pas de sortie
    exit_info = _try_exit(t, _bar(2008.0, 2006.0, 2007.0, atr=5.0), _TS, 0.0, 1.0)
    assert exit_info is None
    assert t["stop_loss"] == 2003.0


def test_trailing_never_loosens_on_a_lower_high_bar():
    strategy.TRAIL_AFTER_TP1_ENABLED = True
    strategy.TRAIL_AFTER_TP1_ATR_MULT = 1.0
    t = _make_long_trade()
    _try_exit(t, _bar(2001.5, 2000.5, 2001.0, atr=5.0), _TS, 0.0, 1.0)
    _try_exit(t, _bar(2008.0, 2006.0, 2007.0, atr=5.0), _TS, 0.0, 1.0)
    assert t["stop_loss"] == 2003.0

    # bougie suivante avec un high plus bas (candidate = 2006 - 5 = 2001 < 2003) :
    # le stop ne doit jamais redescendre ; low reste au-dessus du stop courant
    exit_info = _try_exit(t, _bar(2006.0, 2004.0, 2005.0, atr=5.0), _TS, 0.0, 1.0)
    assert exit_info is None
    assert t["stop_loss"] == 2003.0


def test_trailing_stop_can_be_hit_intrabar_once_tightened():
    strategy.TRAIL_AFTER_TP1_ENABLED = True
    strategy.TRAIL_AFTER_TP1_ATR_MULT = 1.0
    t = _make_long_trade()
    _try_exit(t, _bar(2001.5, 2000.5, 2001.0, atr=5.0), _TS, 0.0, 1.0)

    # bougie qui pousse a 2010 (candidate SL = 2005) puis redescend a 2004 dans la
    # meme bougie -> le trailing fraichement resserre doit etre touche
    exit_info = _try_exit(t, _bar(2010.0, 2004.0, 2004.5, atr=5.0), _TS, 0.0, 1.0)
    assert exit_info is not None
    pnl, exit_price, reason = exit_info
    assert reason == "sl_after_tp1"
    assert exit_price == 2005.0


def test_trailing_ratchets_sl_down_for_short_after_tp1():
    strategy.TRAIL_AFTER_TP1_ENABLED = True
    strategy.TRAIL_AFTER_TP1_ATR_MULT = 1.0
    t = _make_short_trade()
    assert _try_exit(t, _bar(2000.0, 1998.5, 1999.0, atr=5.0), _TS, 0.0, 1.0) is None
    assert t["stop_loss"] == t["entry"]

    # low=1990, ATR=5 -> candidate = 1990 + 5 = 1995, en dessous de BE ; high=1993
    # reste en dessous du nouveau stop -> pas de sortie
    exit_info = _try_exit(t, _bar(1993.0, 1990.0, 1991.0, atr=5.0), _TS, 0.0, 1.0)
    assert exit_info is None
    assert t["stop_loss"] == 1995.0


def test_trailing_skipped_when_atr_is_zero():
    strategy.TRAIL_AFTER_TP1_ENABLED = True
    strategy.TRAIL_AFTER_TP1_ATR_MULT = 1.0
    t = _make_long_trade()
    _try_exit(t, _bar(2001.5, 2000.5, 2001.0, atr=5.0), _TS, 0.0, 1.0)
    exit_info = _try_exit(t, _bar(2010.0, 2009.0, 2009.5, atr=0.0), _TS, 0.0, 1.0)
    assert exit_info is None
    assert t["stop_loss"] == t["entry"]  # pas d'ATR dispo -> pas de mise a jour du trailing
