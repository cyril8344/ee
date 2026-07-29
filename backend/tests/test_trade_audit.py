"""Tests for trade_audit._replay_sl_after_tp1 — distinguishing the fixed
same-bar bug from a genuine breakeven retracement, using controlled M5 data."""
from unittest.mock import patch

import pandas as pd
import pytest

from trade_audit import _replay_sl_after_tp1


def _m5(rows):
    idx = pd.DatetimeIndex([r[0] for r in rows], name="time", tz="UTC")
    df = pd.DataFrame({
        "open":  [r[1] for r in rows],
        "high":  [r[2] for r in rows],
        "low":   [r[3] for r in rows],
        "close": [r[4] for r in rows],
    }, index=idx)
    df.attrs["provider"] = "twelvedata"
    return df


def _trade(entry=4036.77, tp1=4034.887, tp2=4030.0, direction="short"):
    return {
        "direction": direction,
        "entry_time": "2026-07-29T09:00:00+00:00",
        "exit_time": "2026-07-29T09:05:00+00:00",
        "entry_price": entry,
        "take_profit1": tp1,
        "take_profit2": tp2,
    }


def test_flags_bug_signature_when_same_bar_opens_above_entry_then_dives_through_tp1():
    # Bougie qui ouvre au-dessus de l'entrée puis chute d'un trait à travers TP1 —
    # exactement le cas qui déclenchait le bug (high de la bougie de TP1 déjà >= BE).
    m5 = _m5([
        ("2026-07-29T09:00:00Z", 4037.50, 4037.60, 4034.50, 4035.00),
        ("2026-07-29T09:05:00Z", 4035.00, 4035.80, 4034.00, 4034.50),
    ])
    with patch("trade_audit.load_m5_data", return_value=m5):
        result = _replay_sl_after_tp1(_trade(), "XAUUSD")
    assert result["status"] == "ok"
    assert result["bug_signature"] is True


def test_no_bug_signature_when_tp1_bar_stays_below_entry():
    # La bougie de TP1 n'a jamais dépassé l'entrée (high < entry) — un vrai
    # retour au breakeven nécessiterait une bougie ultérieure distincte.
    m5 = _m5([
        ("2026-07-29T09:00:00Z", 4036.00, 4036.20, 4034.50, 4034.80),
        ("2026-07-29T09:05:00Z", 4034.80, 4036.90, 4034.20, 4036.50),  # retour réel plus tard
    ])
    with patch("trade_audit.load_m5_data", return_value=m5):
        result = _replay_sl_after_tp1(_trade(), "XAUUSD")
    assert result["status"] == "ok"
    assert result["bug_signature"] is False
    assert result["genuine_be_touch_later"] is True


def test_would_reach_tp2_detected_after_tp1_bar():
    m5 = _m5([
        ("2026-07-29T09:00:00Z", 4036.00, 4036.20, 4034.50, 4034.80),
        ("2026-07-29T09:05:00Z", 4034.80, 4035.00, 4029.50, 4029.80),  # touche TP2 (4030.0)
    ])
    with patch("trade_audit.load_m5_data", return_value=m5):
        result = _replay_sl_after_tp1(_trade(), "XAUUSD")
    assert result["status"] == "ok"
    assert result["would_reach_tp2"] is True


def test_synthetic_data_marked_unavailable_not_silently_trusted():
    m5 = _m5([("2026-07-29T09:00:00Z", 4036.00, 4036.20, 4034.50, 4034.80)])
    m5.attrs["provider"] = "synthetic"
    with patch("trade_audit.load_m5_data", return_value=m5):
        result = _replay_sl_after_tp1(_trade(), "XAUUSD")
    assert result["status"] == "data_unavailable"
